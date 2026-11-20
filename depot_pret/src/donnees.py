"""Lecture, nettoyage et séparation des données.

Une ligne du CSV = une chanson décrite par 13 mesures (le tempo, l'énergie...)
et son genre musical (la réponse que le modèle doit apprendre à deviner).
"""

import os
from typing import Iterator

import pandas as pd
from sklearn.model_selection import train_test_split

from exceptions import DonneeInvalideError, FichierIntrouvableError

CIBLE = "music_genre"
COLONNES_NUMERIQUES = [
    "popularity", "acousticness", "danceability", "duration_ms", "energy",
    "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence",
]
COLONNES_ENTREE = COLONNES_NUMERIQUES + ["key", "mode"]

# Un modèle ne comprend que des nombres : on remplace chaque note et chaque
# mode par sa position dans ces listes (C -> 0, C# -> 1, ... B -> 11).
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MODES = ["Minor", "Major"]

GRAINE = 42  # même graine = même séparation train/test à chaque exécution


def verifier_fichier(chemin: str) -> None:
    """Vérifie qu'un fichier existe.

    Raises:
        FichierIntrouvableError: si le fichier n'existe pas.
    """
    if not os.path.isfile(chemin):
        raise FichierIntrouvableError(f"Fichier introuvable : {chemin}")


def lire_csv_par_blocs(chemin: str, taille_bloc: int = 10_000) -> Iterator[pd.DataFrame]:
    """Générateur : lit le CSV par morceaux de ``taille_bloc`` lignes.

    Au lieu de tout charger d'un coup, ``yield`` donne un bloc à la fois.

    Raises:
        FichierIntrouvableError: si le fichier n'existe pas.
    """
    verifier_fichier(chemin)
    with open(chemin, encoding="utf-8") as fichier:
        for bloc in pd.read_csv(fichier, chunksize=taille_bloc):
            yield bloc


def lire_csv(chemin: str) -> pd.DataFrame:
    """Lit tout le CSV en recollant les blocs.

    Raises:
        FichierIntrouvableError: si le fichier n'existe pas.
        DonneeInvalideError: si le fichier est vide.
    """
    blocs = list(lire_csv_par_blocs(chemin))
    if not blocs:
        raise DonneeInvalideError(f"Le fichier est vide : {chemin}")
    return pd.concat(blocs, ignore_index=True)


def encoder_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace ``key`` et ``mode`` (du texte) par des nombres.

    Une note ou un mode inconnu devient une valeur manquante (NaN).

    Returns:
        Une copie du DataFrame, l'original n'est pas modifié.
    """
    copie = df.copy()
    copie["key"] = copie["key"].astype(str).str.strip().map(
        {note: position for position, note in enumerate(NOTES)})
    copie["mode"] = copie["mode"].astype(str).str.strip().map(
        {mode: position for position, mode in enumerate(MODES)})
    return copie


def nettoyer(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données brutes.

    Étapes : garder les colonnes utiles, transformer le texte en nombres,
    traiter les codes « inconnu » (``?`` pour le tempo, ``-1`` pour la durée),
    encoder ``key`` et ``mode``, puis supprimer les lignes incomplètes.

    Raises:
        DonneeInvalideError: si une colonne nécessaire est absente.
    """
    colonnes = COLONNES_ENTREE + [CIBLE]
    absentes = [colonne for colonne in colonnes if colonne not in df.columns]
    if absentes:
        raise DonneeInvalideError(f"Colonnes manquantes dans les données : {absentes}")

    propre = df.dropna(how="all")[colonnes].copy()           # lignes vides et colonnes inutiles
    for colonne in COLONNES_NUMERIQUES:
        propre[colonne] = pd.to_numeric(propre[colonne], errors="coerce")   # "?" devient NaN
    propre.loc[propre["duration_ms"] == -1, "duration_ms"] = float("nan")   # -1 = inconnu
    propre = encoder_categories(propre)
    propre[CIBLE] = propre[CIBLE].astype(str).str.strip()
    return propre.dropna().reset_index(drop=True)             # supprime les lignes incomplètes


def calculer_intervalles(df: pd.DataFrame) -> dict[str, list[float]]:
    """Donne ``[minimum, maximum]`` de chaque colonne numérique (pour les curseurs)."""
    return {colonne: [float(df[colonne].min()), float(df[colonne].max())]
            for colonne in COLONNES_NUMERIQUES}


def calculer_medianes(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Valeur typique (médiane) de chaque mesure numérique, genre par genre."""
    medianes = df.groupby(CIBLE)[COLONNES_NUMERIQUES].median()
    return {genre: {colonne: float(valeur) for colonne, valeur in ligne.items()}
            for genre, ligne in medianes.iterrows()}


class JeuDeDonnees:
    """Le jeu de données : sait se charger, se nettoyer et se séparer en deux."""

    def __init__(self, chemin: str) -> None:
        """Prépare l'objet sans lire le fichier (``charger`` le fera)."""
        self._chemin = chemin
        self._df: pd.DataFrame | None = None

    def charger(self) -> "JeuDeDonnees":
        """Lit le CSV et le nettoie. Retourne l'objet pour pouvoir enchaîner."""
        self._df = nettoyer(lire_csv(self._chemin))
        return self

    @property
    def df(self) -> pd.DataFrame:
        """Les données nettoyées.

        Raises:
            DonneeInvalideError: si ``charger`` n'a pas encore été appelé.
        """
        if self._df is None:
            raise DonneeInvalideError("Les données ne sont pas chargées : appelez charger().")
        return self._df

    def separer(self, taille_test: float = 0.2):
        """Sépare en entraînement (pour apprendre) et test (pour évaluer honnêtement).

        ``stratify`` garde la même proportion de chaque genre des deux côtés.

        Returns:
            ``X_train, X_test, y_train, y_test`` (X = mesures, y = genres).
        """
        return train_test_split(self.df[COLONNES_ENTREE], self.df[CIBLE],
                                test_size=taille_test, random_state=GRAINE,
                                stratify=self.df[CIBLE])

    def __len__(self) -> int:
        return 0 if self._df is None else len(self._df)

    def __repr__(self) -> str:
        etat = "non chargé" if self._df is None else f"{len(self._df)} chansons"
        return f"JeuDeDonnees({self._chemin!r}, {etat})"
