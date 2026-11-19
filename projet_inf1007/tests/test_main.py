"""Tests du point d'entrée : les erreurs doivent donner un message, pas un traceback."""

from main import main


def test_entrainer_avec_un_fichier_absent_retourne_1(tmp_path, capsys):
    code = main(["entrainer", "--donnees", str(tmp_path / "absent.csv")])
    assert code == 1
    assert "introuvable" in capsys.readouterr().out
