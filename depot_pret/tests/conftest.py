"""Données de test fabriquées à la main : trois genres faciles à distinguer.

Les tests n'ont donc pas besoin du vrai CSV.
"""

import numpy as np
import pandas as pd
import pytest

from donnees import CIBLE, COLONNES_NUMERIQUES


@pytest.fixture
def donnees_brutes() -> pd.DataFrame:
    """300 chansons : 100 par genre, avec key et mode écrits en texte comme dans le CSV."""
    generateur = np.random.default_rng(0)
    blocs = []
    for genre, energie, tempo in [("Calme", 0.2, 80), ("Moyen", 0.5, 120), ("Fort", 0.9, 160)]:
        bloc = pd.DataFrame({c: generateur.uniform(0, 1, 100) for c in COLONNES_NUMERIQUES})
        bloc["energy"] = generateur.normal(energie, 0.05, 100)
        bloc["tempo"] = generateur.normal(tempo, 5, 100)
        bloc["duration_ms"] = generateur.integers(120_000, 300_000, 100)
        bloc["key"] = generateur.choice(["C", "D", "A"], 100)
        bloc["mode"] = generateur.choice(["Major", "Minor"], 100)
        bloc[CIBLE] = genre
        blocs.append(bloc)
    return pd.concat(blocs, ignore_index=True)


@pytest.fixture
def une_chanson(donnees_brutes) -> dict:
    """Les valeurs d'une chanson, comme l'interface les enverrait."""
    return donnees_brutes.drop(columns=[CIBLE]).iloc[0].to_dict()


@pytest.fixture
def metadonnees() -> dict:
    """Un petit contenu de metadonnees.json, écrit à la main."""
    return {
        "genres": ["Calme", "Fort"],
        "notes": ["C", "C#", "D"],
        "modes": ["Minor", "Major"],
        "intervalles": {"energy": [0.0, 1.0], "tempo": [50.0, 200.0], "popularity": [0.0, 100.0]},
        "medianes": {"Calme": {"energy": 0.2, "tempo": 80.0, "popularity": 30.0},
                     "Fort": {"energy": 0.9, "tempo": 160.0, "popularity": 60.0}},
        "importances": {"energy": 0.6, "tempo": 0.3, "popularity": 0.1},
    }
