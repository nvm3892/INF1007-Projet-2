"""Point d'entrée du projet.

    python src/main.py entrainer    entraîne, évalue et sauvegarde le modèle
    python src/main.py              ouvre l'interface graphique
"""

import argparse
import os
import sys

from donnees import NOTES, MODES, JeuDeDonnees, calculer_intervalles, calculer_medianes
from exceptions import DonneeInvalideError, FichierIntrouvableError
from graphiques import enregistrer_figure, figure_exploration, figure_matrice_confusion
from modele import (NOM_FICHIER_METADONNEES, NOM_FICHIER_MODELE, ModeleForet,
                    ModeleLogistique, ModeleReference, ecrire_json, sauvegarder_modele)

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PAR_DEFAUT = os.path.join(RACINE, "donnees", "music_genre_clean.csv")
DOSSIER_MODELES = os.path.join(RACINE, "modeles")
DOSSIER_FIGURES = os.path.join(RACINE, "figures")
CHEMIN_HISTORIQUE = os.path.join(RACINE, "historique_predictions.csv")


def entrainer(chemin_csv: str) -> None:
    """Toute la chaîne, étape par étape : données, modèles, sauvegarde, graphiques."""
    # 1. Charger et nettoyer, puis séparer : 80 % pour apprendre, 20 % pour tester
    jeu = JeuDeDonnees(chemin_csv).charger()
    X_train, X_test, y_train, y_test = jeu.separer()
    print(f"{len(jeu)} chansons : {len(X_train)} pour entraîner, {len(X_test)} pour tester")

    # 2. Entraîner les trois modèles et les évaluer sur le même ensemble de test
    reference, logistique, foret = ModeleReference(), ModeleLogistique(), ModeleForet()
    scores = {}
    for modele in (reference, logistique, foret):
        modele.entrainer(X_train, y_train)
        scores[modele.NOM] = modele.evaluer(X_test, y_test)
        print(f"  {modele.NOM:<32} exactitude {scores[modele.NOM]:.1%}  ({modele.duree:.1f} s)")

    # 3. Sauvegarder la forêt (le modèle retenu) et ses métadonnées pour l'interface
    sauvegarder_modele(foret, os.path.join(DOSSIER_MODELES, NOM_FICHIER_MODELE))
    ecrire_json({
        "genres": foret.classes,
        "notes": NOTES,
        "modes": MODES,
        "intervalles": calculer_intervalles(jeu.df),
        "medianes": calculer_medianes(jeu.df),
        "importances": foret.importances(),
        "exactitude": scores[foret.NOM],
        "exactitude_reference": scores[ModeleReference.NOM],
        "scores": scores,
    }, os.path.join(DOSSIER_MODELES, NOM_FICHIER_METADONNEES))

    # 4. Graphiques
    enregistrer_figure(figure_exploration(jeu.df),
                       os.path.join(DOSSIER_FIGURES, "exploration_genres.png"))
    enregistrer_figure(figure_matrice_confusion(y_test, foret.predire(X_test), foret.classes),
                       os.path.join(DOSSIER_FIGURES, "matrice_confusion.png"))
    print(f"Modèle et métadonnées dans {DOSSIER_MODELES}, graphiques dans {DOSSIER_FIGURES}")


def lancer_interface() -> int:
    """Ouvre la fenêtre de l'application. Retourne 1 si le modèle est introuvable."""
    from interface import Application      # importé ici : l'entraînement n'a pas besoin de Tkinter
    try:
        Application(DOSSIER_MODELES, CHEMIN_HISTORIQUE).demarrer()
    except (FichierIntrouvableError, DonneeInvalideError) as erreur:
        print(f"Erreur : {erreur}\nEntraînez d'abord le modèle : python src/main.py entrainer")
        return 1
    return 0


def main(arguments: list[str] | None = None) -> int:
    """Lit la ligne de commande. Retourne 0 si tout va bien, 1 en cas d'erreur."""
    parseur = argparse.ArgumentParser(description="Projet INF1007 : genre musical")
    parseur.add_argument("commande", nargs="?", choices=["entrainer"])
    parseur.add_argument("--donnees", default=CSV_PAR_DEFAUT, help="chemin du CSV nettoyé")
    args = parseur.parse_args(arguments)

    if args.commande is None:
        return lancer_interface()
    try:
        entrainer(args.donnees)
    except (FichierIntrouvableError, DonneeInvalideError) as erreur:
        print(f"Erreur : {erreur}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
