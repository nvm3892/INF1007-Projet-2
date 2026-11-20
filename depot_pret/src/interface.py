"""Interface graphique « WatsDatBeat? » (Tkinter).

Ce fichier ne contient que l'affichage : le calcul est dans les autres modules
(``variables`` valide les saisies, ``modele`` prédit, ``historique`` enregistre).
"""

import os
import random
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from exceptions import DonneeInvalideError, ModeleNonEntraineError
from graphiques import CLAIR, ERREUR, FOND, FOND_CHAMP, TEXTE, figure_probabilites
from historique import Historique
from modele import (NOM_FICHIER_METADONNEES, NOM_FICHIER_MODELE, ModeleForet, lire_json,
                    reperes_principaux)
from variables import (TEXTES, VariableCategorielle, construire_variables, formater_nombre,
                       valider_titre)

COLONNES_FORMULAIRE = 3                   # le formulaire s'affiche sur 3 colonnes
POLICES_TITRE = ["Bahnschrift SemiBold", "Arial Black", "Segoe UI Black", "DejaVu Sans"]
POLICES_TEXTE = ["Bahnschrift", "Segoe UI", "DejaVu Sans"]


class Application:
    """La fenêtre principale.

    Elle CONTIENT un modèle, la liste des variables et l'historique (composition) :
    elle ne fait que les relier à l'écran.
    """

    def __init__(self, dossier_modeles: str, chemin_historique: str) -> None:
        """Charge le modèle sauvegardé et construit la fenêtre (sans l'afficher).

        Args:
            dossier_modeles: dossier contenant le modèle .joblib et le JSON.
            chemin_historique: fichier CSV de l'historique.

        Raises:
            FichierIntrouvableError: si le modèle ou les métadonnées sont absents.
        """
        # Le formulaire se construit à partir des métadonnées du modèle sauvegardé
        self._metadonnees = lire_json(os.path.join(dossier_modeles, NOM_FICHIER_METADONNEES))
        self._modele = ModeleForet.charger(os.path.join(dossier_modeles, NOM_FICHIER_MODELE))
        self._variables = construire_variables(self._metadonnees)
        self._historique = Historique(chemin_historique, [v.nom for v in self._variables])

        self._champs = {}              # nom de variable -> case de saisie
        self._etiquettes = {}          # nom de variable -> texte au-dessus de la case
        self._canvas_graphique = None
        self._fenetre = tk.Tk()
        self._construire_fenetre()

    # ------------------------------------------------------------------ construction

    def _construire_fenetre(self) -> None:
        """Crée la fenêtre et assemble ses deux colonnes (conversation et formulaire à gauche, disque et graphique à droite)."""
        fenetre = self._fenetre
        fenetre.title("WatsDatBeat?")
        fenetre.configure(bg=FOND)
        fenetre.geometry("1360x860")
        fenetre.minsize(1280, 840)
        self._preparer_polices_et_styles()

        gauche = tk.Frame(fenetre, bg=FOND)
        gauche.grid(row=0, column=0, sticky="nsew", padx=(30, 10), pady=(10, 10))
        droite = tk.Frame(fenetre, bg=FOND)
        droite.grid(row=0, column=1, sticky="n", padx=(10, 30), pady=(10, 10))
        fenetre.columnconfigure(0, weight=1)
        fenetre.rowconfigure(0, weight=1)

        tk.Label(gauche, text="WatsDatBeat?", font=self._police_titre, bg=FOND,
                 fg=CLAIR).pack(anchor="w")
        self._construire_conversation(gauche)
        self._construire_formulaire(gauche)
        self._construire_boutons(gauche)
        self._construire_genres_proches(gauche)
        self._construire_disque(droite)
        self._afficher_graphique({genre: 0.0 for genre in self._metadonnees["genres"]}, droite)

    def _preparer_polices_et_styles(self) -> None:
        """Choisit les polices et le style des listes déroulantes et du tableau."""
        disponibles = set(tkfont.families(self._fenetre))
        titre = next((p for p in POLICES_TITRE if p in disponibles), "Helvetica")
        texte = next((p for p in POLICES_TEXTE if p in disponibles), "Helvetica")
        self._police_titre = tkfont.Font(family=titre, size=40, weight="bold")
        self._police_section = tkfont.Font(family=titre, size=14, weight="bold")
        self._police_disque = tkfont.Font(family=titre, size=24, weight="bold")
        self._police_chat = tkfont.Font(family=texte, size=12, weight="bold")
        self._police_label = tkfont.Font(family=texte, size=10, weight="bold")
        self._police_aide = tkfont.Font(family=texte, size=8)
        self._police_champ = tkfont.Font(family=texte, size=10)

        style = ttk.Style(self._fenetre)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=FOND_CHAMP, background=CLAIR,
                        foreground=TEXTE, arrowcolor=FOND, bordercolor=FOND_CHAMP)
        style.map("TCombobox", fieldbackground=[("readonly", FOND_CHAMP)],
                  foreground=[("readonly", TEXTE)])
        style.configure("Treeview", background=FOND_CHAMP, fieldbackground=FOND_CHAMP,
                        foreground=TEXTE, rowheight=26)
        style.configure("Treeview.Heading", background=CLAIR, foreground=FOND)
        self._fenetre.option_add("*TCombobox*Listbox.background", FOND_CHAMP)
        self._fenetre.option_add("*TCombobox*Listbox.foreground", TEXTE)

    def _construire_conversation(self, parent: tk.Frame) -> None:
        """Zone de texte où le DJ parle (messages normaux et erreurs)."""
        self._conversation = tk.Text(parent, height=8, width=64, bg=FOND, fg=TEXTE, bd=0,
                                     highlightthickness=0, wrap="word", font=self._police_chat,
                                     state="disabled", cursor="arrow")
        self._conversation.tag_configure("erreur", foreground=ERREUR)
        self._conversation.pack(anchor="w", pady=(6, 6))
        self._dire("DJ: Quel genre on cherche?")
        self._dire("DJ: Décris ta chanson avec les cases ci-dessous, puis clique sur Prédire.")

    def _construire_formulaire(self, parent: tk.Frame) -> None:
        """Un champ par variable : case de texte pour un nombre, liste déroulante pour la tonalité et le mode."""
        formulaire = tk.Frame(parent, bg=FOND)
        formulaire.pack(anchor="w", pady=(4, 4))

        # première case : le titre (facultatif), qui ne passe pas dans le modèle
        self._champ_titre = self._creer_cellule(formulaire, 0, "Titre", "facultatif", None)
        for rang, variable in enumerate(self._variables, start=1):
            choix = variable.choix if isinstance(variable, VariableCategorielle) else None
            self._champs[variable.nom] = self._creer_cellule(
                formulaire, rang, variable.libelle, variable.aide, choix, variable.nom)

    def _creer_cellule(self, parent, rang, libelle, aide, choix, nom=None):
        """Crée une case du formulaire : une liste déroulante si ``choix`` existe, sinon un champ."""
        ligne, colonne = divmod(rang, COLONNES_FORMULAIRE)
        cellule = tk.Frame(parent, bg=FOND)
        cellule.grid(row=ligne, column=colonne, sticky="w", padx=(0, 18), pady=3)

        entete = tk.Frame(cellule, bg=FOND)
        entete.pack(anchor="w")
        etiquette = tk.Label(entete, text=libelle, font=self._police_label, bg=FOND, fg=TEXTE)
        etiquette.pack(side="left")
        tk.Label(entete, text=f"  {aide}", font=self._police_aide, bg=FOND,
                 fg=TEXTE).pack(side="left")

        if choix is None:
            case = tk.Entry(cellule, width=24, font=self._police_champ, bg=FOND_CHAMP, fg=TEXTE,
                            insertbackground=TEXTE, relief="flat", highlightthickness=1,
                            highlightbackground=FOND_CHAMP, highlightcolor=CLAIR)
        else:
            case = ttk.Combobox(cellule, values=choix, width=22, state="readonly",
                                font=self._police_champ)
        case.pack(anchor="w", ipady=3)
        if nom is not None:
            self._etiquettes[nom] = etiquette
        return case

    def _construire_boutons(self, parent: tk.Frame) -> None:
        """Boutons Prédire, Exemple et Historique."""
        boutons = tk.Frame(parent, bg=FOND)
        boutons.pack(anchor="w", pady=(8, 6))
        for texte, commande in (("Prédire", self._predire),
                                ("Exemple", self._remplir_exemple),
                                ("Historique", self._ouvrir_historique)):
            tk.Button(boutons, text=texte, command=commande, font=self._police_label,
                      bg=CLAIR, fg=FOND, activebackground=TEXTE, activeforeground=FOND,
                      relief="flat", cursor="hand2", padx=18, pady=6).pack(side="left", padx=(0, 10))

    def _construire_genres_proches(self, parent: tk.Frame) -> None:
        """Trois carrés qui montrent les genres suivants les plus probables."""
        tk.Label(parent, text="Genres proches", font=self._police_section, bg=FOND,
                 fg=CLAIR).pack(anchor="w", pady=(10, 4))
        rangee = tk.Frame(parent, bg=FOND)
        rangee.pack(anchor="w")
        self._carres = []
        for _ in range(3):
            cadre = tk.Frame(rangee, bg=FOND)
            cadre.pack(side="left", padx=(0, 18))
            carre = tk.Frame(cadre, width=84, height=84, bg=CLAIR)
            carre.pack_propagate(False)
            carre.pack()
            pourcentage = tk.Label(carre, text="—", font=self._police_section, bg=CLAIR, fg=FOND)
            pourcentage.pack(expand=True)
            nom_genre = tk.Label(cadre, text=" ", font=self._police_label, bg=FOND, fg=TEXTE)
            nom_genre.pack()
            self._carres.append((pourcentage, nom_genre))

    def _construire_disque(self, parent: tk.Frame) -> None:
        """Titre « Now playing », disque vinyle et graphique des probabilités."""
        self._now_playing = tk.Label(parent, text="Now playing: —", font=self._police_section,
                                     bg=FOND, fg=CLAIR, wraplength=460, justify="right")
        self._now_playing.pack(anchor="e", pady=(24, 0))
        disque = tk.Canvas(parent, width=400, height=390, bg=FOND, highlightthickness=0)
        disque.pack()
        centre_x, centre_y = 215, 195
        disque.create_oval(centre_x - 185, centre_y - 185, centre_x + 185, centre_y + 185,
                           fill=CLAIR, outline="")
        for rayon in range(130, 185, 12):                       # les sillons du vinyle
            disque.create_oval(centre_x - rayon, centre_y - rayon, centre_x + rayon,
                               centre_y + rayon, outline="#a4bcc2")
        disque.create_oval(centre_x - 110, centre_y - 110, centre_x + 110, centre_y + 110,
                           fill=FOND, outline="")
        disque.create_oval(centre_x - 104, centre_y - 104, centre_x + 104, centre_y + 104,
                           outline=CLAIR, width=2)
        self._trou_disque = disque.create_oval(centre_x - 34, centre_y - 34, centre_x + 34,
                                               centre_y + 34, outline=CLAIR, width=3)
        self._texte_disque = disque.create_text(centre_x, centre_y, text="", fill=CLAIR,
                                                font=self._police_disque, width=190,
                                                justify="center")
        self._disque = disque

    def _afficher_graphique(self, probabilites: dict, parent: tk.Frame | None = None) -> None:
        """Dessine (ou redessine) le graphique Matplotlib des probabilités."""
        if parent is not None:
            self._cadre_graphique = tk.Frame(parent, bg=FOND)
            self._cadre_graphique.pack(pady=(6, 0))
        if self._canvas_graphique is not None:
            self._canvas_graphique.get_tk_widget().destroy()
        self._canvas_graphique = FigureCanvasTkAgg(figure_probabilites(probabilites),
                                                   master=self._cadre_graphique)
        self._canvas_graphique.draw()
        self._canvas_graphique.get_tk_widget().pack()

    # ------------------------------------------------------------------ actions

    def _dire(self, texte: str, etiquette: str = "") -> None:
        """Ajoute une ligne à la conversation avec le DJ."""
        self._conversation.configure(state="normal")
        self._conversation.insert("end", texte + "\n", etiquette)
        self._conversation.see("end")
        self._conversation.configure(state="disabled")

    @staticmethod
    def _ecrire(case, texte: str) -> None:
        """Remplace le contenu d'une case de texte."""
        if isinstance(case, ttk.Combobox):
            case.set(texte)
        else:
            case.delete(0, "end")
            case.insert(0, texte)

    def _predire(self) -> None:
        """Valide les saisies, prédit le genre et affiche le résultat."""
        for etiquette in self._etiquettes.values():
            etiquette.configure(fg=TEXTE)

        erreurs, valeurs = [], {}
        try:
            titre = valider_titre(self._champ_titre.get())
        except DonneeInvalideError as erreur:
            titre = ""
            erreurs.append(str(erreur))
        for variable in self._variables:
            try:
                valeurs[variable.nom] = variable.valider(self._champs[variable.nom].get())
            except DonneeInvalideError as erreur:
                erreurs.append(str(erreur))
                self._etiquettes[variable.nom].configure(fg=ERREUR)
        if erreurs:
            self._dire("DJ: Je ne peux pas lancer le disque, corrigeons ça :", "erreur")
            for message in erreurs:
                self._dire("  • " + message, "erreur")
            return

        try:
            genre, probabilites = self._modele.predire_une_chanson(valeurs)
        except (DonneeInvalideError, ModeleNonEntraineError) as erreur:
            self._dire(f"DJ: Je n'arrive pas à deviner : {erreur}", "erreur")
            return
        confiance = probabilites[genre]

        self._dire(f"You: {titre}")
        self._dire(f"DJ: Cette chanson entre dans la catégorie {genre} "
                   f"(confiance : {confiance:.0%}).")
        reperes = [f"{TEXTES[nom][0].lower()} {formater_nombre(saisie)} "
                   f"(typique du {genre} : {formater_nombre(typique)})"
                   for nom, saisie, typique in reperes_principaux(valeurs, genre, self._metadonnees)]
        self._dire("DJ: Repères : " + " ; ".join(reperes) + ".")

        self._now_playing.configure(text=f"Now playing: {titre}")
        self._disque.itemconfigure(self._texte_disque, text=genre)
        self._disque.itemconfigure(self._trou_disque, state="hidden")
        autres = sorted((g for g in probabilites if g != genre), key=probabilites.get, reverse=True)
        for (pourcentage, nom_genre), proche in zip(self._carres, autres[:3]):
            pourcentage.configure(text=f"{probabilites[proche]:.0%}")
            nom_genre.configure(text=proche)
        self._afficher_graphique(probabilites)

        try:
            self._historique.ajouter(titre, genre, confiance, valeurs)
        except OSError as erreur:
            self._dire(f"DJ: Prédiction faite, mais l'historique n'a pas pu être écrit ({erreur}).",
                       "erreur")

    def _remplir_exemple(self) -> None:
        """Remplit le formulaire avec une chanson typique d'un genre pris au hasard."""
        genre = random.choice(self._metadonnees["genres"])
        typiques = self._metadonnees["medianes"][genre]
        for variable in self._variables:
            if variable.nom in typiques:
                self._ecrire(self._champs[variable.nom], formater_nombre(typiques[variable.nom]))
        self._ecrire(self._champs["key"], random.choice(self._metadonnees["notes"]))
        self._ecrire(self._champs["mode"], random.choice(self._metadonnees["modes"]))
        self._ecrire(self._champ_titre, f"Exemple {genre}")
        self._dire(f"DJ: Voici une chanson typique du {genre}. Clique sur Prédire : "
                   "est-ce que je la retrouve ?")

    def _ouvrir_historique(self) -> None:
        """Ouvre une fenêtre avec toutes les prédictions déjà faites."""
        fenetre = tk.Toplevel(self._fenetre)
        fenetre.title("Historique des prédictions")
        fenetre.configure(bg=FOND)
        fenetre.geometry("760x420")
        lignes = self._historique.lire()
        if not lignes:
            tk.Label(fenetre, text="Aucune prédiction pour l'instant.", font=self._police_chat,
                     bg=FOND, fg=CLAIR).pack(expand=True)
            return
        colonnes = ("date", "titre", "genre", "confiance")
        tableau = ttk.Treeview(fenetre, columns=colonnes, show="headings")
        for colonne, texte, largeur in (("date", "Date", 170), ("titre", "Titre", 260),
                                        ("genre", "Genre", 140), ("confiance", "Confiance", 100)):
            tableau.heading(colonne, text=texte)
            tableau.column(colonne, width=largeur, anchor="w")
        for ligne in reversed(lignes):                      # la plus récente en premier
            tableau.insert("", "end", values=(ligne["date"], ligne["titre"], ligne["genre"],
                                              f"{ligne['confiance']} %"))
        tableau.pack(fill="both", expand=True, padx=12, pady=12)

    def demarrer(self) -> None:
        """Affiche la fenêtre et attend les clics de l'utilisateur."""
        self._fenetre.mainloop()
