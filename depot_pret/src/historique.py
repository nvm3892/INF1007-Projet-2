"""Historique des prédictions, enregistré dans un fichier CSV."""

import csv
import os
from datetime import datetime

COLONNES_FIXES = ["date", "titre", "genre", "confiance"]


def proteger(texte: str) -> str:
    """Empêche Excel de prendre un titre comme « =1+1 » pour une formule."""
    return "'" + texte if texte[:1] in ("=", "+", "-", "@") else texte


class Historique:
    """Un fichier CSV où chaque prédiction ajoute une ligne."""

    def __init__(self, chemin: str, colonnes_variables: list[str]) -> None:
        """
        Args:
            chemin: fichier CSV de l'historique (créé au besoin).
            colonnes_variables: noms des variables, ajoutés comme colonnes.
        """
        self._chemin = chemin
        self._colonnes = COLONNES_FIXES + list(colonnes_variables)

    def ajouter(self, titre: str, genre: str, confiance: float, valeurs: dict) -> None:
        """Ajoute une prédiction à la fin du fichier.

        Args:
            titre: titre donné par l'utilisateur.
            genre: genre prédit.
            confiance: probabilité du genre prédit, entre 0 et 1.
            valeurs: les valeurs saisies, une par variable.

        Raises:
            OSError: si le fichier ne peut pas être écrit.
        """
        os.makedirs(os.path.dirname(self._chemin) or ".", exist_ok=True)
        nouveau = not os.path.isfile(self._chemin)
        ligne = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "titre": proteger(titre),
            "genre": genre,
            "confiance": round(confiance * 100, 1),
            **valeurs,
        }
        with open(self._chemin, "a", newline="", encoding="utf-8") as fichier:
            ecrivain = csv.DictWriter(fichier, fieldnames=self._colonnes, extrasaction="ignore")
            if nouveau:
                ecrivain.writeheader()
            ecrivain.writerow(ligne)

    def lire(self) -> list[dict]:
        """Retourne toutes les prédictions (la plus ancienne en premier), ou ``[]`` s'il n'y en a pas."""
        if not os.path.isfile(self._chemin):
            return []
        with open(self._chemin, newline="", encoding="utf-8") as fichier:
            return list(csv.DictReader(fichier))

    def __len__(self) -> int:
        return len(self.lire())

    def __repr__(self) -> str:
        return f"Historique({self._chemin!r}, {len(self)} prédictions)"
