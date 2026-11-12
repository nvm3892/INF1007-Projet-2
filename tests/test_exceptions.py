"""Tests du module exceptions (premier test de la CI)."""

import pytest

from exceptions import DonneeInvalideError, verifier_non_vide


def test_verifier_non_vide_retourne_valeur_nettoyee():
    assert verifier_non_vide("  42  ", "age") == "42"


def test_verifier_non_vide_refuse_chaine_vide():
    with pytest.raises(DonneeInvalideError):
        verifier_non_vide("", "age")


def test_verifier_non_vide_refuse_espaces_seulement():
    with pytest.raises(DonneeInvalideError, match="age"):
        verifier_non_vide("   ", "age")
