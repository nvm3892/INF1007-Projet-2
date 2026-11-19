"""Tests du module variables : validation des saisies de l'utilisateur."""

import pytest

from exceptions import DonneeInvalideError
from variables import (Variable, VariableCategorielle, VariableNumerique, construire_variables,
                       formater_nombre, valider_titre)


@pytest.fixture
def energie() -> VariableNumerique:
    return VariableNumerique("energy", "Énergie", "0 calme, 1 intense", 0.0, 1.0)


@pytest.mark.parametrize("saisie, attendu", [("0.5", 0.5), (" 0,5 ", 0.5), ("1", 1.0), ("0", 0.0)])
def test_numerique_accepte_les_saisies_valides(energie, saisie, attendu):
    assert energie.valider(saisie) == attendu


@pytest.mark.parametrize("saisie", ["", "   "])
def test_numerique_refuse_le_vide(energie, saisie):
    with pytest.raises(DonneeInvalideError, match="Énergie"):
        energie.valider(saisie)


def test_numerique_refuse_le_texte(energie):
    with pytest.raises(DonneeInvalideError, match="nombre"):
        energie.valider("abc")


@pytest.mark.parametrize("saisie", ["1.5", "-0.1", "nan", "inf"])
def test_numerique_refuse_hors_intervalle(energie, saisie):
    with pytest.raises(DonneeInvalideError, match="compris entre"):
        energie.valider(saisie)


def test_categorielle_accepte_et_refuse():
    tonalite = VariableCategorielle("key", "Tonalité", "", ["C", "D"])
    assert tonalite.valider("C") == "C"
    with pytest.raises(DonneeInvalideError):
        tonalite.valider("Z")
    with pytest.raises(DonneeInvalideError):
        tonalite.valider("")


def test_variable_abstraite_ne_peut_pas_etre_creee():
    with pytest.raises(TypeError):
        Variable("x", "X")


def test_polymorphisme_meme_methode_deux_comportements(energie):
    mode = VariableCategorielle("mode", "Mode", "", ["Minor", "Major"])
    assert isinstance(energie.valider("0.3"), float)
    assert isinstance(mode.valider("Major"), str)


def test_repr_donne_le_type_et_le_nom(energie):
    assert repr(energie) == "VariableNumerique('energy')"


def test_construire_variables(metadonnees):
    variables = construire_variables(metadonnees)
    assert [v.nom for v in variables] == ["energy", "tempo", "popularity", "key", "mode"]
    assert isinstance(variables[0], VariableNumerique)
    assert variables[0].libelle == "Énergie"
    assert isinstance(variables[-1], VariableCategorielle)
    assert variables[-1].choix == ["Minor", "Major"]


def test_formater_nombre():
    assert formater_nombre(4497994.0) == "4 497 994"
    assert formater_nombre(0.5) == "0.5"


def test_valider_titre():
    assert valider_titre("  ") == "Ma chanson"
    assert valider_titre(" Circus ") == "Circus"
    with pytest.raises(DonneeInvalideError):
        valider_titre("x" * 61)
