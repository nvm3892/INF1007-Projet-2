"""Tests du module donnees."""

import numpy as np
import pandas as pd
import pytest

from donnees import (CIBLE, COLONNES_ENTREE, JeuDeDonnees, calculer_intervalles,
                     encoder_categories, lire_csv, lire_csv_par_blocs, nettoyer)
from exceptions import DonneeInvalideError, FichierIntrouvableError


def test_nettoyer_supprime_codes_inconnus_et_lignes_vides(donnees_brutes):
    brut = donnees_brutes.copy()
    brut["tempo"] = brut["tempo"].astype(str)
    brut.loc[0, "tempo"] = "?"            # tempo inconnu
    brut.loc[1, "duration_ms"] = -1       # durée inconnue
    ligne_vide = pd.DataFrame([[np.nan] * brut.shape[1]], columns=brut.columns)
    brut = pd.concat([brut, ligne_vide], ignore_index=True)

    propre = nettoyer(brut)

    assert len(propre) == len(donnees_brutes) - 2
    assert not propre.isna().any().any()
    assert propre["tempo"].dtype == float


def test_nettoyer_garde_seulement_les_colonnes_utiles(donnees_brutes):
    propre = nettoyer(donnees_brutes.assign(artist_name="x", instance_id=1))
    assert list(propre.columns) == COLONNES_ENTREE + [CIBLE]


def test_nettoyer_colonne_manquante_leve_erreur(donnees_brutes):
    with pytest.raises(DonneeInvalideError, match="music_genre"):
        nettoyer(donnees_brutes.drop(columns=[CIBLE]))


def test_encoder_categories_remplace_texte_par_nombres():
    df = pd.DataFrame({"key": ["C", "B", "C#"], "mode": ["Minor", "Major", "Major"]})
    resultat = encoder_categories(df)
    assert resultat["key"].tolist() == [0, 11, 1]
    assert resultat["mode"].tolist() == [0, 1, 1]
    assert df["key"].tolist() == ["C", "B", "C#"]        # l'original n'a pas changé


def test_encoder_categories_note_inconnue_devient_manquante():
    df = pd.DataFrame({"key": ["H"], "mode": ["Major"]})
    assert encoder_categories(df)["key"].isna().all()


def test_lire_csv_par_blocs_donne_toutes_les_lignes(tmp_path, donnees_brutes):
    chemin = tmp_path / "donnees.csv"
    donnees_brutes.head(25).to_csv(chemin, index=False)

    blocs = list(lire_csv_par_blocs(str(chemin), taille_bloc=10))

    assert [len(bloc) for bloc in blocs] == [10, 10, 5]
    assert len(lire_csv(str(chemin))) == 25


def test_lire_csv_fichier_inexistant(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        lire_csv(str(tmp_path / "absent.csv"))


def test_calculer_intervalles(donnees_brutes):
    minimum, maximum = calculer_intervalles(nettoyer(donnees_brutes))["tempo"]
    assert minimum == pytest.approx(donnees_brutes["tempo"].min())
    assert maximum == pytest.approx(donnees_brutes["tempo"].max())


def test_jeu_de_donnees_avant_chargement(tmp_path):
    jeu = JeuDeDonnees(str(tmp_path / "x.csv"))
    assert len(jeu) == 0
    with pytest.raises(DonneeInvalideError):
        _ = jeu.df


def test_jeu_de_donnees_separation(tmp_path, donnees_brutes):
    chemin = tmp_path / "donnees.csv"
    donnees_brutes.to_csv(chemin, index=False)
    jeu = JeuDeDonnees(str(chemin)).charger()

    X_train, X_test, y_train, y_test = jeu.separer(taille_test=0.2)
    X_train_bis, X_test_bis, _, _ = jeu.separer(taille_test=0.2)

    assert len(X_test) / len(jeu) == pytest.approx(0.2, abs=0.01)
    assert set(X_train.index).isdisjoint(X_test.index)             # aucune chanson des deux côtés
    assert X_test.index.equals(X_test_bis.index)                    # reproductible
    assert (y_test == "Calme").mean() == pytest.approx(1 / 3, abs=0.02)   # genres proportionnels
    assert "300 chansons" in repr(jeu)


def test_calculer_medianes_par_genre(donnees_brutes):
    from donnees import calculer_medianes
    medianes = calculer_medianes(nettoyer(donnees_brutes))
    assert set(medianes) == {"Calme", "Moyen", "Fort"}
    assert medianes["Calme"]["energy"] < medianes["Fort"]["energy"]
