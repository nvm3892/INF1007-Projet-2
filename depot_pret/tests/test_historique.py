"""Tests du module historique : fichier CSV des prédictions."""

import pytest

from historique import Historique, proteger

COLONNES = ["energy", "tempo"]


@pytest.fixture
def historique(tmp_path) -> Historique:
    return Historique(str(tmp_path / "dossier" / "historique.csv"), COLONNES)


def test_fichier_absent_donne_une_liste_vide(historique):
    assert historique.lire() == []
    assert len(historique) == 0


def test_ajouter_puis_lire(historique):
    historique.ajouter("Circus", "Pop", 0.876, {"energy": 0.5, "tempo": 120.0})
    ligne = historique.lire()[0]
    assert (ligne["titre"], ligne["genre"], ligne["confiance"]) == ("Circus", "Pop", "87.6")
    assert (ligne["energy"], ligne["tempo"]) == ("0.5", "120.0")
    assert len(historique) == 1


def test_entete_ecrit_une_seule_fois(historique):
    historique.ajouter("A", "Rock", 0.5, {"energy": 0.1, "tempo": 90.0})
    historique.ajouter("B", "Jazz", 0.6, {"energy": 0.2, "tempo": 100.0})
    with open(historique._chemin, encoding="utf-8") as fichier:
        assert sum(ligne.startswith("date,") for ligne in fichier) == 1
    assert [ligne["titre"] for ligne in historique.lire()] == ["A", "B"]


def test_virgules_et_guillemets_dans_le_titre(historique):
    historique.ajouter('Moi, "toi"', "Rock", 0.5, {"energy": 0.1, "tempo": 90.0})
    assert historique.lire()[0]["titre"] == 'Moi, "toi"'


def test_proteger_contre_les_formules_excel():
    assert proteger("=1+1") == "'=1+1"
    assert proteger("@x") == "'@x"
    assert proteger("Circus") == "Circus"


def test_repr(historique):
    assert "0 prédictions" in repr(historique)
