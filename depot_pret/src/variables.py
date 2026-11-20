"""Les variables que l'utilisateur remplit dans l'interface, et la validation de leurs saisies.

Une « variable » = une case du formulaire (le tempo, l'énergie, la tonalité...).
Chaque type de variable sait vérifier ce qu'on y écrit : c'est le rôle de ``valider``.
"""

from abc import ABC, abstractmethod

from exceptions import DonneeInvalideError, verifier_non_vide

LONGUEUR_MAX_TITRE = 60
TITRE_PAR_DEFAUT = "Ma chanson"

# nom de la colonne -> (libellé affiché, petite aide pour un non-informaticien)
TEXTES = {
    "popularity": ("Popularité", "0 à 100"),
    "acousticness": ("Acoustique", "0 électro, 1 acoustique"),
    "danceability": ("Dansabilité", "0 à 1"),
    "duration_ms": ("Durée", "en ms (60 000 = 1 min)"),
    "energy": ("Énergie", "0 calme, 1 intense"),
    "instrumentalness": ("Instrumental", "0 voix, 1 sans voix"),
    "liveness": ("Public", "0 studio, 1 concert"),
    "loudness": ("Volume", "en dB (négatif)"),
    "speechiness": ("Parole", "0 chant, 1 parole"),
    "tempo": ("Tempo", "battements par minute"),
    "valence": ("Positivité", "0 triste, 1 joyeux"),
    "key": ("Tonalité", "note principale"),
    "mode": ("Mode", "Major ou Minor"),
}


def formater_nombre(nombre: float) -> str:
    """Écrit un nombre lisiblement : 4497994.0 -> « 4 497 994 », 0.5 -> « 0.5 »."""
    texte = f"{nombre:,.3f}".rstrip("0").rstrip(".")
    return texte.replace(",", " ")


class Variable(ABC):
    """Classe parente : une case du formulaire, avec un nom, un libellé et une aide."""

    def __init__(self, nom: str, libelle: str, aide: str = "") -> None:
        """
        Args:
            nom: nom de la colonne dans les données (ex. ``"energy"``).
            libelle: texte affiché à l'utilisateur (ex. ``"Énergie"``).
            aide: courte explication affichée à côté.
        """
        self._nom = nom
        self._libelle = libelle
        self._aide = aide

    @property
    def nom(self) -> str:
        """Nom de la colonne dans les données."""
        return self._nom

    @property
    def libelle(self) -> str:
        """Texte affiché à l'utilisateur."""
        return self._libelle

    @property
    def aide(self) -> str:
        """Courte explication affichée à côté du champ."""
        return self._aide

    @abstractmethod
    def valider(self, texte: str):
        """Vérifie ce que l'utilisateur a écrit et retourne la valeur propre.

        Raises:
            DonneeInvalideError: avec un message clair si la saisie est refusée.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._nom!r})"


class VariableNumerique(Variable):
    """Une mesure numérique : on vérifie que c'est un nombre dans l'intervalle connu."""

    def __init__(self, nom: str, libelle: str, aide: str, minimum: float, maximum: float) -> None:
        super().__init__(nom, libelle, aide)
        self._minimum = minimum
        self._maximum = maximum

    @property
    def minimum(self) -> float:
        """Plus petite valeur acceptée."""
        return self._minimum

    @property
    def maximum(self) -> float:
        """Plus grande valeur acceptée."""
        return self._maximum

    def valider(self, texte: str) -> float:
        """Retourne la saisie en nombre.

        Raises:
            DonneeInvalideError: si la saisie est vide, n'est pas un nombre
                ou sort de l'intervalle observé dans les données.
        """
        texte = verifier_non_vide(texte, self._libelle)
        try:
            valeur = float(texte.replace(" ", "").replace(",", "."))
        except ValueError:
            raise DonneeInvalideError(
                f"Le champ « {self._libelle} » doit être un nombre (par exemple 0.5).") from None
        if not self._minimum <= valeur <= self._maximum:    # faux aussi pour nan
            raise DonneeInvalideError(
                f"Le champ « {self._libelle} » doit être compris entre "
                f"{formater_nombre(self._minimum)} et {formater_nombre(self._maximum)} "
                f"(valeur saisie : {texte}).")
        return valeur


class VariableCategorielle(Variable):
    """Un choix dans une liste : on vérifie que la valeur fait partie des choix."""

    def __init__(self, nom: str, libelle: str, aide: str, choix: list[str]) -> None:
        super().__init__(nom, libelle, aide)
        self._choix = list(choix)

    @property
    def choix(self) -> list[str]:
        """Les valeurs permises."""
        return list(self._choix)

    def valider(self, texte: str) -> str:
        """Retourne le choix tel quel.

        Raises:
            DonneeInvalideError: si rien n'est choisi ou si la valeur n'est pas dans la liste.
        """
        texte = verifier_non_vide(texte, self._libelle)
        if texte not in self._choix:
            raise DonneeInvalideError(
                f"Le champ « {self._libelle} » doit être l'une de ces valeurs : "
                f"{', '.join(self._choix)}.")
        return texte


def valider_titre(texte: str) -> str:
    """Le titre est facultatif : vide = « Ma chanson ». Refuse les titres trop longs."""
    texte = texte.strip()
    if len(texte) > LONGUEUR_MAX_TITRE:
        raise DonneeInvalideError(
            f"Le titre ne doit pas dépasser {LONGUEUR_MAX_TITRE} caractères.")
    return texte or TITRE_PAR_DEFAUT


def construire_variables(metadonnees: dict) -> list[Variable]:
    """Construit la liste des variables du formulaire à partir des métadonnées du modèle.

    Une variable numérique par intervalle, puis la tonalité et le mode.
    """
    variables = []
    for nom, (minimum, maximum) in metadonnees["intervalles"].items():
        libelle, aide = TEXTES.get(nom, (nom, ""))
        variables.append(VariableNumerique(nom, libelle, aide, minimum, maximum))
    for nom, choix in (("key", metadonnees["notes"]), ("mode", metadonnees["modes"])):
        libelle, aide = TEXTES.get(nom, (nom, ""))
        variables.append(VariableCategorielle(nom, libelle, aide, choix))
    return variables
