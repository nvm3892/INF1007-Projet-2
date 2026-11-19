"""Tests du module graphiques."""

from matplotlib.figure import Figure

from donnees import CIBLE, nettoyer
from graphiques import enregistrer_figure, figure_exploration, figure_matrice_confusion


def test_figure_exploration_a_quatre_graphiques(donnees_brutes):
    figure = figure_exploration(nettoyer(donnees_brutes))
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 4


def test_matrice_confusion_et_enregistrement(tmp_path, donnees_brutes):
    genres = sorted(donnees_brutes[CIBLE].unique())
    figure = figure_matrice_confusion(donnees_brutes[CIBLE], donnees_brutes[CIBLE], genres)

    assert figure.axes[0].get_title() == "Matrice de confusion"

    chemin = tmp_path / "figures" / "matrice.png"
    enregistrer_figure(figure, str(chemin))
    assert chemin.is_file() and chemin.stat().st_size > 0


def test_figure_probabilites_met_en_avant_le_genre_le_plus_probable():
    from graphiques import CLAIR, figure_probabilites
    figure = figure_probabilites({"Rock": 0.2, "Jazz": 0.7, "Rap": 0.1})
    graphique = figure.axes[0]
    textes = [t.get_text() for t in graphique.get_yticklabels()]
    assert textes[-1] == "Jazz"                       # le plus probable est en haut
    assert len(graphique.patches) == 3
    assert graphique.get_title(loc="left") == "Probabilité de chaque genre"
    from matplotlib.colors import to_hex
    assert to_hex(graphique.patches[-1].get_facecolor()) == CLAIR
