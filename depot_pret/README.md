# INF1007 — Projet 2 : de la donnée à l'application

> Équipe : _Prénom Nom_ et _Prénom Nom_ — Session A2026

## Description

Application qui devine le **genre musical** d'une chanson (Alternative, Anime, Blues, Classical, Country, Electronic, Hip-Hop, Jazz, Rap ou Rock) à partir de mesures audio : tempo, énergie, danceability, tonalité, etc. L'utilisateur règle les caractéristiques d'une chanson et obtient le genre le plus probable, avec la probabilité de chaque genre.

**Interface (« WatsDatBeat? »)** : fenêtre Tkinter inspirée de la maquette. Un « DJ » conversationnel dit quoi remplir et annonce le genre ; un disque vinyle affiche le genre prédit ; trois carrés montrent les « genres proches » ; un graphique Matplotlib intégré montre la probabilité de chaque genre. Chaque saisie est validée (vide, texte, hors intervalle) avec un message clair en français, et chaque prédiction est ajoutée à `historique_predictions.csv` (bouton « Historique »). L'interface lit `modeles/metadonnees.json` et le modèle sauvegardé : **aucun ré-entraînement**.

Adaptations par rapport à la maquette : le modèle prend 13 mesures audio (pas un titre de chanson), donc le formulaire a ces champs ; les pochettes d'albums sont remplacées par les genres les plus proches (pas de données d'albums, ni de droits d'auteur à gérer).

## Jeu de données

- Nom : **Prediction of Music Genre** (Kaggle, utilisateur `vicsuperman`) — https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre
- Contenu : 50 000 chansons, 10 genres de 5 000 chansons, 11 variables numériques et 2 catégorielles (`key`, `mode`).
- **Source d'origine et licence : non précisées sur la page Kaggle du jeu de données** (vérifié en lisant la page). Les colonnes ressemblent aux « audio features » de Spotify, mais l'auteur ne le confirme pas : nous ne l'affirmons donc pas. Le fichier nettoyé (3,4 Mo) est inclus dans le dépôt pour que l'entraînement soit reproductible ; il serait à retirer si une licence interdisant la redistribution apparaissait.
- Nettoyage appliqué (`src/donnees.py`) : suppression des 5 lignes vides, des lignes dont le tempo vaut `?` ou la durée `-1` (codes de valeur inconnue, environ 20 % des lignes) et des colonnes inutiles (identifiant, nom de l'artiste, nom du titre, date). Résultat : **40 560 lignes**, environ 4 040 par genre. Les chansons avec `instrumentalness = 0` sont conservées : c'est une vraie valeur (chanson chantée), pas une valeur inconnue.
- Le fichier nettoyé se place dans `donnees/music_genre_clean.csv`.

## Installation et lancement

```bash
git clone https://github.com/nvm3892/INF1007-Projet-2.git
cd INF1007-Projet-2
python -m venv .venv
.venv\Scripts\activate           # Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
python src/main.py entrainer     # entraîne, évalue et sauvegarde le modèle (inutile si modeles/ est fourni)
python src/main.py               # ouvre l'interface graphique
pytest                           # lance les tests
```

Le modèle sauvegardé (`modeles/modele_genre.joblib`) est un fichier binaire : il ne s'ouvre pas dans un éditeur de texte. S'il ne se charge pas à cause d'une version différente de scikit-learn, `python src/main.py entrainer` le régénère en quelques secondes.

## Structure du code

```
INF1007-Projet-2/
├── README.md, DECISIONS.md, JOURNAL_IA.md, requirements.txt
├── .github/workflows/tests.yml   # CI : pytest et couverture à chaque push
├── donnees/                      # music_genre_clean.csv
├── modeles/                      # modele_genre.joblib + metadonnees.json
├── figures/                      # graphique d'exploration, matrice de confusion
├── src/
│   ├── exceptions.py             # DonneeInvalideError, FichierIntrouvableError, ModeleNonEntraineError
│   ├── donnees.py                # lecture par blocs, nettoyage, classe JeuDeDonnees
│   ├── modele.py                 # Modele (abstraite) → ModeleReference, ModeleLogistique, ModeleForet
│   ├── variables.py              # Variable (abstraite) → VariableNumerique, VariableCategorielle
│   ├── historique.py             # classe Historique (fichier CSV des prédictions)
│   ├── graphiques.py             # exploration, matrice de confusion, probabilités
│   ├── interface.py              # classe Application (Tkinter)
│   └── main.py                   # point d'entrée (entraînement ou interface)
└── tests/                        # tests pytest de la logique
```

L'interface se construit à partir de `modeles/metadonnees.json` (genres, notes, modes, minimum, maximum et médiane de chaque mesure) et utilise `Modele.predire_une_chanson`, qui renvoie le genre et la probabilité de chaque genre. `Application` **contient** (composition) le modèle, les variables et l'historique.

## Résultat du modèle

- **Cible :** `music_genre` (classification, 10 genres). **Métrique :** exactitude (part des chansons dont le genre est bien deviné). Séparation 80 % / 20 %, stratifiée, graine 42. Les mesures sont mises à la même échelle (`StandardScaler`) ; `key` et `mode` sont remplacés par leur position (C = 0, C# = 1…).
- Scores sur l'ensemble de test, jamais vu à l'entraînement :

| Modèle | Exactitude |
|---|---|
| Référence (toujours le genre majoritaire) | 10,1 % |
| Régression logistique | 53,5 % |
| **Forêt aléatoire (retenue)** | **56,4 %** |

- La forêt fait plus de cinq fois mieux que le hasard, mais se trompe encore presque une fois sur deux. Les confusions sont logiques (voir `figures/matrice_confusion.png`) : la moitié des chansons Rap sont classées Hip-Hop (et inversement), Alternative est le genre le moins bien reconnu (31 %, souvent pris pour du Rock), tandis que Classical (85 %), Rock (75 %) et Anime (74 %) sont les mieux reconnus.
- Limites : le genre est une étiquette humaine souvent floue, les mesures audio ne décrivent ni les paroles ni la culture d'un genre, et l'origine des données n'est pas documentée.

## Couverture des tests

62 tests pytest sur la logique ; `coverage report` indique **93 %** sur `src/`. L'affichage Tkinter (`interface.py`) est exclu de la couverture : il est vérifié à la main, la logique qu'il appelle (validation, historique, prédiction) est testée.

## Outils du chapitre 7 utilisés

- **Décorateur** `chronometrer` : mesure la durée de chaque entraînement sans modifier le code de la méthode.
- **Générateur** `lire_csv_par_blocs` : lit le CSV par blocs de 10 000 lignes avec `yield`, au lieu de tout charger d'un coup.

## Documentation consultée

Pages consultées pour ce projet :
- pandas : `read_csv` (paramètre `chunksize`) — https://pandas.pydata.org/docs/
- scikit-learn : `train_test_split`, `RandomForestClassifier` — https://scikit-learn.org/stable/
- Tkinter : https://docs.python.org/3/library/tkinter.html et `tkinter.ttk` (Combobox, Treeview) — https://docs.python.org/3/library/tkinter.ttk.html
- Matplotlib dans Tkinter : `FigureCanvasTkAgg` — https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html
