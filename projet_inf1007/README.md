# INF1007 — Projet 2 : de la donnée à l'application

> Équipe : _Prénom Nom_ et _Prénom Nom_ — Session A2026

## Description

_À compléter : domaine, ce que fait l'application, ce qu'un utilisateur peut en faire._

## Jeu de données

- Nom et lien : _à confirmer après validation avec l'auxiliaire (piste actuelle : 30,000 Spotify Songs sur Kaggle)_
- Source et licence : _à compléter (ex. UCI Machine Learning Repository, CC BY 4.0)_
- Citation : _à compléter_
- Fichier dans `donnees/` s'il fait moins de 25 Mo et si la licence permet de le redistribuer ; sinon instructions de téléchargement ici.

## Installation et lancement

```bash
git clone https://github.com/nvm3892/INF1007-Projet-2.git
cd INF1007-Projet-2
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python src/main.py               # lance l'application (à venir)
pytest                           # lance les tests
```

## Structure du code

```
INF1007-Projet-2/
├── README.md, DECISIONS.md, JOURNAL_IA.md, requirements.txt
├── .github/workflows/tests.yml   # CI : pytest à chaque push
├── donnees/                      # jeu de données (CSV)
├── modeles/                      # modèle sauvegardé + métadonnées JSON
├── src/
│   ├── donnees.py                # lecture, nettoyage, classes de variables
│   ├── modele.py                 # entraînement, évaluation, prédiction
│   ├── exceptions.py             # exceptions personnalisées
│   ├── interface.py              # interface graphique
│   └── main.py                   # point d'entrée
└── tests/
```

## Résultat du modèle

_À compléter : variable cible, métrique, score du modèle vs modèle de référence, limites._

## Couverture des tests

_À compléter avec le résultat de `coverage run -m pytest` puis `coverage report`._

## Outil du chapitre 7 utilisé

_À compléter : décorateur, générateur ou lambda, expliqué en une phrase._

## Documentation consultée

- pandas : _page consultée_
- scikit-learn : _page consultée_
- Technologie d'interface : _page consultée_
