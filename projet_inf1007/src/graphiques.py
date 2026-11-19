"""Graphiques Matplotlib : exploration des données et évaluation du modèle.

Chaque fonction retourne une figure ; ``enregistrer_figure`` l'écrit en PNG
(et l'interface pourra afficher la même figure dans sa fenêtre).
"""

import os

import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from donnees import CIBLE

# Couleurs de l'interface (celles de la maquette)
FOND = "#5f6b6d"
CLAIR = "#b5cbd0"
TEXTE = "#e3eef0"
ATTENUE = "#93a7ac"
FOND_CHAMP = "#4a5557"
ERREUR = "#ffd6c7"

# (colonne, titre) des quatre graphiques d'exploration
CARACTERISTIQUES = [
    ("tempo", "Tempo (battements par minute)"),
    ("instrumentalness", "Instrumentalness (0 = voix, 1 = sans voix)"),
    ("energy", "Énergie (0 = calme, 1 = intense)"),
    ("speechiness", "Speechiness (présence de parole)"),
]


def figure_exploration(df: pd.DataFrame) -> Figure:
    """Compare les 10 genres sur 4 mesures avec des boîtes à moustaches.

    Une boîte contient la moitié centrale des chansons du genre ; le trait
    dans la boîte est la médiane.
    """
    genres = df.groupby(CIBLE)["tempo"].median().sort_values().index.tolist()
    figure = Figure(figsize=(14, 10))
    graphiques = figure.subplots(2, 2).ravel()
    for graphique, (colonne, titre) in zip(graphiques, CARACTERISTIQUES):
        valeurs = [df.loc[df[CIBLE] == genre, colonne] for genre in genres]
        graphique.boxplot(valeurs, orientation="horizontal", tick_labels=genres,
                          showfliers=False)
        graphique.set_title(titre)
        graphique.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    return figure


def figure_matrice_confusion(y_vrai, y_pred, genres: list[str]) -> Figure:
    """Matrice de confusion : une ligne par genre réel, une colonne par genre prédit.

    Chaque case donne la part des chansons du genre réel que le modèle a rangées
    dans le genre prédit. Une bonne matrice a de grosses valeurs sur la diagonale.
    """
    comptes = confusion_matrix(y_vrai, y_pred, labels=genres)
    parts = comptes / comptes.sum(axis=1, keepdims=True)   # chaque ligne totalise 100 %
    figure = Figure(figsize=(9, 8))
    graphique = figure.subplots()
    graphique.imshow(parts, cmap="Blues", vmin=0, vmax=1)
    graphique.set_xticks(range(len(genres)), labels=genres, rotation=45, ha="right")
    graphique.set_yticks(range(len(genres)), labels=genres)
    graphique.set_xlabel("Genre prédit")
    graphique.set_ylabel("Genre réel")
    graphique.set_title("Matrice de confusion")
    for ligne in range(len(genres)):
        for colonne in range(len(genres)):
            graphique.text(colonne, ligne, f"{parts[ligne, colonne]:.0%}",
                           ha="center", va="center", fontsize=8)
    figure.tight_layout()
    return figure


def figure_probabilites(probabilites: dict[str, float]) -> Figure:
    """Barres horizontales : la probabilité de chaque genre (celle de l'interface).

    Le genre le plus probable est en clair et en haut. Si toutes les
    probabilités sont à zéro (avant la première prédiction), aucun genre n'est mis en avant.
    """
    genres = sorted(probabilites, key=probabilites.get)        # le plus probable en dernier = en haut
    pourcentages = [probabilites[genre] * 100 for genre in genres]
    en_avant = bool(pourcentages) and max(pourcentages) > 0
    couleurs = [CLAIR if (en_avant and genre == genres[-1]) else ATTENUE for genre in genres]

    figure = Figure(figsize=(4.7, 3.0), facecolor=FOND)
    graphique = figure.subplots()
    graphique.set_facecolor(FOND)
    graphique.barh(genres, pourcentages, color=couleurs)
    graphique.set_xlim(0, 100)
    graphique.set_title("Probabilité de chaque genre", loc="left", color=CLAIR, fontweight="bold")
    graphique.set_xlabel("Probabilité (%)", color=TEXTE)
    graphique.set_ylabel("Genre", color=TEXTE)
    graphique.tick_params(colors=TEXTE, labelsize=8)
    for cote in ("top", "right"):
        graphique.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        graphique.spines[cote].set_color(CLAIR)
    figure.tight_layout()
    return figure


def enregistrer_figure(figure: Figure, chemin: str) -> None:
    """Enregistre une figure en PNG (crée le dossier au besoin)."""
    os.makedirs(os.path.dirname(chemin) or ".", exist_ok=True)
    figure.savefig(chemin, dpi=120)
