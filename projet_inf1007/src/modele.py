"""Les modèles : entraînement, évaluation, prédiction et sauvegarde."""

import json
import os
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from donnees import COLONNES_ENTREE, GRAINE, encoder_categories, verifier_fichier
from exceptions import DonneeInvalideError, ModeleNonEntraineError

NOM_FICHIER_MODELE = "modele_genre.joblib"
NOM_FICHIER_METADONNEES = "metadonnees.json"


def chronometrer(fonction: Callable) -> Callable:
    """Décorateur : mesure combien de temps dure une méthode.

    Un décorateur « enveloppe » une fonction : il ajoute un comportement
    (ici, un chronomètre) sans modifier le code de la fonction.
    """
    @wraps(fonction)
    def enveloppe(self, *args, **kwargs):
        """Appelle la méthode, puis garde sa durée dans ``self._duree``."""
        debut = time.perf_counter()
        resultat = fonction(self, *args, **kwargs)
        self._duree = time.perf_counter() - debut
        return resultat
    return enveloppe


def exactitude_numpy(y_vrai, y_pred) -> float:
    """Exactitude = part des prédictions correctes, calculée avec NumPy.

    ``vrai == pred`` compare toutes les paires d'un seul coup (sans boucle) et
    donne des True/False ; la moyenne des True/False est la part de True.

    Raises:
        DonneeInvalideError: si les deux séries n'ont pas la même taille ou sont vides.
    """
    vrai, pred = np.asarray(y_vrai), np.asarray(y_pred)
    if vrai.shape != pred.shape or vrai.size == 0:
        raise DonneeInvalideError("Les deux séries doivent avoir la même taille, non nulle.")
    return float(np.mean(vrai == pred))


class Modele(ABC):
    """Classe parente : tout ce qui est commun à nos trois modèles.

    Les classes enfants ne changent qu'une chose : ``creer_estimateur``,
    c'est-à-dire l'algorithme de prédiction utilisé.
    """

    NOM = "modèle"

    def __init__(self) -> None:
        self._pipeline = None   # le modèle entraîné (vide tant qu'on n'a pas entraîné)
        self._duree = 0.0       # durée du dernier entraînement, en secondes

    @abstractmethod
    def creer_estimateur(self):
        """Retourne l'algorithme de prédiction (redéfini dans chaque enfant)."""

    @chronometrer
    def entrainer(self, X: pd.DataFrame, y: pd.Series) -> "Modele":
        """Apprend à partir des chansons d'entraînement.

        Le pipeline enchaîne deux étapes : ``StandardScaler`` met toutes les
        mesures à la même échelle (une durée en millisecondes est un très gros
        nombre comparé à une énergie entre 0 et 1), puis l'algorithme apprend.
        """
        self._pipeline = make_pipeline(StandardScaler(), self.creer_estimateur())
        self._pipeline.fit(X, y)
        return self

    @classmethod
    def charger(cls, chemin: str) -> "Modele":
        """Crée un modèle à partir d'un fichier .joblib déjà entraîné (sans ré-entraîner).

        Raises:
            FichierIntrouvableError: si le fichier n'existe pas.
        """
        modele = cls()
        modele._pipeline = charger_pipeline(chemin)
        return modele

    def _modele_pret(self):
        """Retourne le pipeline entraîné, ou refuse s'il n'existe pas encore."""
        if self._pipeline is None:
            raise ModeleNonEntraineError(f"Le modèle « {self.NOM} » n'est pas entraîné.")
        return self._pipeline

    def predire(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit le genre de chaque chanson de ``X``.

        Raises:
            ModeleNonEntraineError: si le modèle n'est pas entraîné.
        """
        return self._modele_pret().predict(X)

    def evaluer(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Exactitude sur des chansons que le modèle n'a jamais vues."""
        return exactitude_numpy(y, self.predire(X))

    @property
    def classes(self) -> list[str]:
        """Les genres que le modèle connaît."""
        return list(self._modele_pret().classes_)

    @property
    def pipeline(self):
        """Le modèle entraîné, prêt à être sauvegardé."""
        return self._modele_pret()

    @property
    def duree(self) -> float:
        """Durée du dernier entraînement, en secondes."""
        return self._duree

    def predire_une_chanson(self, valeurs: dict) -> tuple[str, dict[str, float]]:
        """Prédit le genre d'une seule chanson (utilisé par l'interface).

        Args:
            valeurs: une valeur par colonne d'entrée ; ``key`` est une note
                (ex. ``"C#"``) et ``mode`` vaut ``"Major"`` ou ``"Minor"``.

        Returns:
            Le genre le plus probable et la probabilité de chaque genre.

        Raises:
            DonneeInvalideError: si une valeur manque ou si la note/le mode est inconnu.
            ModeleNonEntraineError: si le modèle n'est pas entraîné.
        """
        manquantes = [colonne for colonne in COLONNES_ENTREE if colonne not in valeurs]
        if manquantes:
            raise DonneeInvalideError(f"Valeurs manquantes : {manquantes}")
        ligne = encoder_categories(pd.DataFrame([valeurs]))[COLONNES_ENTREE]
        if ligne[["key", "mode"]].isna().any(axis=None):
            raise DonneeInvalideError("Note (key) ou mode inconnu.")
        probabilites = self._modele_pret().predict_proba(ligne)[0]
        genre = self.classes[int(np.argmax(probabilites))]
        return genre, dict(zip(self.classes, probabilites.tolist()))

    def __repr__(self) -> str:
        etat = "entraîné" if self._pipeline is not None else "non entraîné"
        return f"{type(self).__name__}({self.NOM}, {etat})"


class ModeleReference(Modele):
    """Modèle trivial : répond toujours le genre le plus fréquent (point de comparaison)."""

    NOM = "référence (genre majoritaire)"

    def creer_estimateur(self) -> DummyClassifier:
        """Un classifieur qui répond toujours le genre le plus fréquent."""
        return DummyClassifier(strategy="most_frequent")


class ModeleLogistique(Modele):
    """Régression logistique : calcule un score par genre avec une somme pondérée des mesures."""

    NOM = "régression logistique"

    def creer_estimateur(self) -> LogisticRegression:
        """Une régression logistique (jusqu'à 1000 itérations pour converger)."""
        return LogisticRegression(max_iter=1000)


class ModeleForet(Modele):
    """Forêt aléatoire : 100 arbres de décision qui votent, le genre le plus voté gagne."""

    NOM = "forêt aléatoire"

    def creer_estimateur(self) -> RandomForestClassifier:
        """Une forêt de 100 arbres."""
        # min_samples_leaf=8 : arbres plus petits, donc fichier plus léger (~13 Mo)
        return RandomForestClassifier(n_estimators=100, min_samples_leaf=8,
                                      random_state=GRAINE, n_jobs=-1)

    def importances(self) -> dict[str, float]:
        """Importance de chaque mesure dans les décisions des arbres (la somme fait 1)."""
        foret = self.pipeline[-1]      # la dernière étape du pipeline est la forêt
        return dict(zip(COLONNES_ENTREE, foret.feature_importances_.tolist()))


def reperes_principaux(valeurs: dict, genre: str, metadonnees: dict,
                       combien: int = 2) -> list[tuple[str, float, float]]:
    """Choisit les mesures numériques les plus importantes pour expliquer une prédiction.

    Args:
        valeurs: les valeurs saisies par l'utilisateur.
        genre: le genre prédit.
        metadonnees: contenu de ``metadonnees.json``.
        combien: nombre de mesures à retourner.

    Returns:
        Une liste de ``(nom de la mesure, valeur saisie, valeur typique du genre)``.
    """
    importances = metadonnees["importances"]
    numeriques = list(metadonnees["intervalles"])
    plus_importantes = sorted(numeriques, key=importances.get, reverse=True)[:combien]
    typiques = metadonnees["medianes"][genre]
    return [(nom, valeurs[nom], typiques[nom]) for nom in plus_importantes]


def sauvegarder_modele(modele: Modele, chemin: str) -> None:
    """Enregistre le modèle entraîné dans un fichier .joblib (format binaire, compressé)."""
    pipeline = modele.pipeline
    os.makedirs(os.path.dirname(chemin) or ".", exist_ok=True)
    joblib.dump(pipeline, chemin, compress=3)


def charger_pipeline(chemin: str):
    """Recharge un modèle sauvegardé.

    Raises:
        FichierIntrouvableError: si le fichier n'existe pas.
    """
    verifier_fichier(chemin)
    return joblib.load(chemin)


def ecrire_json(donnees: dict, chemin: str) -> None:
    """Écrit un dictionnaire dans un fichier JSON lisible."""
    os.makedirs(os.path.dirname(chemin) or ".", exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(donnees, fichier, ensure_ascii=False, indent=2)


def lire_json(chemin: str) -> dict:
    """Lit un fichier JSON.

    Raises:
        FichierIntrouvableError: si le fichier n'existe pas.
        DonneeInvalideError: si le contenu n'est pas du JSON valide.
    """
    verifier_fichier(chemin)
    with open(chemin, encoding="utf-8") as fichier:
        try:
            return json.load(fichier)
        except json.JSONDecodeError as erreur:
            raise DonneeInvalideError(f"Fichier JSON illisible : {erreur}") from erreur
