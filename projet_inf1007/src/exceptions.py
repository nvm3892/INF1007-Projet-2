"""Exceptions personnalisées du projet."""


class DonneeInvalideError(Exception):
    """Levée lorsqu'une saisie ou une donnée ne respecte pas les règles attendues.

    Exemples : champ vide, valeur non numérique, valeur hors intervalle,
    colonne manquante dans le CSV, catégorie inconnue.
    """


class FichierIntrouvableError(Exception):
    """Levée lorsqu'un fichier nécessaire (CSV, modèle, métadonnées) n'existe pas."""


class ModeleNonEntraineError(Exception):
    """Levée lorsqu'on demande une prédiction à un modèle pas encore entraîné."""


def verifier_non_vide(valeur: str, nom_champ: str) -> str:
    """Retourne la valeur sans espaces superflus si elle n'est pas vide.

    Args:
        valeur: texte saisi par l'utilisateur.
        nom_champ: nom du champ, utilisé dans le message d'erreur.

    Returns:
        La valeur nettoyée (sans espaces au début et à la fin).

    Raises:
        DonneeInvalideError: si la valeur est vide ou ne contient que des espaces.
    """
    valeur_nettoyee = valeur.strip()
    if not valeur_nettoyee:
        raise DonneeInvalideError(f"Le champ « {nom_champ} » est obligatoire.")
    return valeur_nettoyee
