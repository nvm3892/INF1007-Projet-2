"""Tests du module modele."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from donnees import CIBLE, nettoyer
from exceptions import DonneeInvalideError, FichierIntrouvableError, ModeleNonEntraineError
from modele import (Modele, ModeleForet, ModeleLogistique, ModeleReference, charger_pipeline,
                    ecrire_json, exactitude_numpy, lire_json, reperes_principaux,
                    sauvegarder_modele)
from donnees import COLONNES_ENTREE


@pytest.fixture
def jeu(donnees_brutes):
    """Données nettoyées, séparées à la main : 240 pour entraîner, 60 pour tester."""
    propre = nettoyer(donnees_brutes).sample(frac=1, random_state=0)
    entrainement, test = propre.iloc[:240], propre.iloc[240:]
    return (entrainement[COLONNES_ENTREE], test[COLONNES_ENTREE],
            entrainement[CIBLE], test[CIBLE])


def test_exactitude_numpy_coincide_avec_scikit_learn():
    vrai = ["Rock", "Rap", "Jazz", "Rock", "Rap", "Jazz", "Rock"]
    predit = ["Rock", "Jazz", "Jazz", "Rock", "Rap", "Rap", "Rap"]
    assert exactitude_numpy(vrai, predit) == pytest.approx(accuracy_score(vrai, predit))
    assert exactitude_numpy(vrai, predit) == pytest.approx(4 / 7)


@pytest.mark.parametrize("vrai, predit", [(["a", "b"], ["a"]), ([], [])])
def test_exactitude_numpy_entrees_invalides(vrai, predit):
    with pytest.raises(DonneeInvalideError):
        exactitude_numpy(vrai, predit)


def test_classe_parente_abstraite():
    with pytest.raises(TypeError):
        Modele()


def test_polymorphisme_chaque_enfant_a_son_algorithme():
    assert isinstance(ModeleReference().creer_estimateur(), DummyClassifier)
    assert isinstance(ModeleLogistique().creer_estimateur(), LogisticRegression)
    assert isinstance(ModeleForet().creer_estimateur(), RandomForestClassifier)


def test_predire_avant_entrainement(jeu):
    with pytest.raises(ModeleNonEntraineError):
        ModeleForet().predire(jeu[1])


def test_reference_vaut_la_part_du_genre_majoritaire(jeu):
    X_train, _, y_train, _ = jeu
    exactitude = ModeleReference().entrainer(X_train, y_train).evaluer(X_train, y_train)
    assert exactitude == pytest.approx(y_train.value_counts(normalize=True).max())


def test_foret_bat_largement_la_reference(jeu):
    X_train, X_test, y_train, y_test = jeu
    foret = ModeleForet().entrainer(X_train, y_train).evaluer(X_test, y_test)
    reference = ModeleReference().entrainer(X_train, y_train).evaluer(X_test, y_test)
    assert foret > 0.9
    assert foret > reference + 0.4


def test_le_decorateur_enregistre_la_duree(jeu):
    X_train, _, y_train, _ = jeu
    modele = ModeleReference()
    assert modele.duree == 0.0
    modele.entrainer(X_train, y_train)
    assert modele.duree > 0.0


def test_predire_une_chanson(jeu, une_chanson):
    X_train, _, y_train, _ = jeu
    modele = ModeleForet().entrainer(X_train, y_train)

    genre, probabilites = modele.predire_une_chanson(une_chanson)

    assert genre in modele.classes
    assert sum(probabilites.values()) == pytest.approx(1.0)
    assert probabilites[genre] == max(probabilites.values())


def test_predire_une_chanson_valeur_absente(jeu, une_chanson):
    X_train, _, y_train, _ = jeu
    modele = ModeleForet().entrainer(X_train, y_train)
    del une_chanson["tempo"]
    with pytest.raises(DonneeInvalideError, match="tempo"):
        modele.predire_une_chanson(une_chanson)


def test_predire_une_chanson_note_inconnue(jeu, une_chanson):
    X_train, _, y_train, _ = jeu
    modele = ModeleForet().entrainer(X_train, y_train)
    une_chanson["key"] = "H"
    with pytest.raises(DonneeInvalideError):
        modele.predire_une_chanson(une_chanson)


def test_sauvegarde_puis_rechargement(tmp_path, jeu):
    X_train, X_test, y_train, _ = jeu
    modele = ModeleForet().entrainer(X_train, y_train)
    chemin = str(tmp_path / "sous_dossier" / "modele.joblib")

    sauvegarder_modele(modele, chemin)

    assert np.array_equal(charger_pipeline(chemin).predict(X_test), modele.predire(X_test))


def test_charger_pipeline_fichier_absent(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        charger_pipeline(str(tmp_path / "absent.joblib"))


def test_json_aller_retour_et_erreurs(tmp_path):
    chemin = str(tmp_path / "dossier" / "meta.json")
    ecrire_json({"genres": ["Rock", "Rap"], "exactitude": 0.5}, chemin)
    assert lire_json(chemin) == {"genres": ["Rock", "Rap"], "exactitude": 0.5}

    with pytest.raises(FichierIntrouvableError):
        lire_json(str(tmp_path / "absent.json"))
    abime = tmp_path / "abime.json"
    abime.write_text("{pas du json", encoding="utf-8")
    with pytest.raises(DonneeInvalideError):
        lire_json(str(abime))


def test_repr_indique_l_etat(jeu):
    X_train, _, y_train, _ = jeu
    modele = ModeleLogistique()
    assert "non entraîné" in repr(modele)
    assert "non" not in repr(modele.entrainer(X_train, y_train))


def test_charger_donne_le_meme_modele_sans_reentrainer(jeu, tmp_path):
    X_train, X_test, y_train, _ = jeu
    foret = ModeleForet().entrainer(X_train, y_train)
    chemin = str(tmp_path / "m.joblib")
    sauvegarder_modele(foret, chemin)
    recharge = ModeleForet.charger(chemin)
    assert list(recharge.predire(X_test)) == list(foret.predire(X_test))


def test_charger_fichier_absent(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        ModeleForet.charger(str(tmp_path / "absent.joblib"))


def test_importances_totalisent_un(jeu):
    X_train, _, y_train, _ = jeu
    importances = ModeleForet().entrainer(X_train, y_train).importances()
    assert set(importances) == set(COLONNES_ENTREE)
    assert sum(importances.values()) == pytest.approx(1.0)


def test_reperes_principaux_choisit_les_mesures_les_plus_importantes(metadonnees):
    valeurs = {"energy": 0.85, "tempo": 150.0, "popularity": 55.0}
    reperes = reperes_principaux(valeurs, "Fort", metadonnees)
    assert reperes == [("energy", 0.85, 0.9), ("tempo", 150.0, 160.0)]
