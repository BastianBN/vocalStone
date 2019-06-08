import json
import platform
import threading
import tkinter
from tkinter import filedialog, ttk
from tkinter.ttk import Progressbar
from typing import Optional, Callable

import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from serial.tools import list_ports

from classificateur import *

SEUIL_DETECTION = 500
NOMBRE_FFT_ENREGISTREMENT = 30
NOMBRE_FFT_RECONNAISSANCE = 10  # matrice de (x,64) coefficients de fourier
VERBOSE = False  # pour afficher dans la console les données reçues


def print_debug(s: str, *args, **kwargs) -> None:
    if VERBOSE:
        print(s, *args, **kwargs)


class P2I(object):
    reconnaissance_active = True

    waterfall = [np.linspace(0, 100, 64)]
    waterfall_index = 0
    serial_port: serial.Serial
    ml: DetecteurDeVoix

    graph_change = False

    def __init__(self):
        self.setup_serial()
        self.lancer_reconnaissance_vocale()

    def plot(self, X, Y, *args, **kwargs):
        plt.plot(X, Y, *args, **kwargs)

    def afficher_nom(self, nom: str, autorise: Optional[bool]):
        if autorise is not None:
            print("{} {} autorisé(e)".format(nom, "est" if autorise else "n'est pas"))
        else:
            print(nom)

    def plot_fft(self, coefs_fft):
        X, Y, i = [], [], 0
        for coef in coefs_fft:
            X.append(i)
            X.append(i)
            X.append(i)
            Y.append(0)
            Y.append(coef)
            Y.append(0)
            i += 1
        self.add_plot(X, Y)

    def add_plot(self, X, Y):
        self.plot(X, Y)

    def lancer_reconnaissance_vocale(self):
        print("Reconnaissance vocale dans le terminal")
        if self.serial_port is not None:
            if self.ml is not None:
                self.ml = DetecteurDeVoix()
            if self.serial_port.isOpen():
                self.reconnaitre_voix()
                # self.after(1000, self.afficher_fft_realtime)
            # self.after(2000, self.reset_graph_loop)
            else:
                print("Port série non disponible")
        else:
            print("port série non config")

    def reconnaitre_voix(self):
        """
        read_serial récupère les données du port série, et une fois un nombre suffisant de vecteurs dépassant
        le seuil d'intensité atteint, il exécute la méthode 'analyse_detection' avec ces données
        """
        self.read_serial(self.analyse_detection, repeter=True)

    #    morceau_fft = None
    #    self.coefs_ffts = []
    #    while self.reconnaissance_active:
    #        ligne = self.serial_port.readline().replace(b'\r\n', b'')
    #        print(ligne, end=" ; ")
    #        if ligne == b'restart':
    #            self.waterfall, self.waterfall_index = [], 0
    #            print("Remise à zéro des tableaux, parlez maintenant")
    #            self.coefs_ffts = []
    #            morceau_fft = []
    #            continue
    #        if ligne == b"begin":
    #            morceau_fft = []  # une transformée de Fourier
    #            continue
    #        # if ligne != b'end' and ligne != b'begin' and ligne !=b'\n' and ligne != b'' and ligne != b'restart' and morceau_fft is not None:
    #        # print(ligne)
    #        try:
    #            nombre = float(ligne.decode('utf-8'))
    #            if ligne != 'end' and morceau_fft is not None:
    #                morceau_fft.append(nombre/100)
    #        except (UnicodeDecodeError, ValueError):
    #            pass
    #        if ligne == b'end' and morceau_fft is not None:
    #            print("\nlongeur: {}".format(len(morceau_fft)))
    #            if len(morceau_fft) == 62:
    #                fft_array = np.array(morceau_fft)
    #                if fft_array.max() > 1:#SEUIL_DETECTION:
    #                    self.coefs_ffts.append(fft_array)
    #                    print("nouveau morceau dans coefs_ffts")
    #                    if len(self.waterfall) <= NOMBRE_FFT_RECONNAISSANCE:
    #                        self.waterfall.append(fft_array)
    #                    else:
    #                        if self.waterfall_index >= len(self.waterfall) - 1:
    #                            self.waterfall_index = 0
    #                        else:
    #                            self.waterfall_index += 1
    #                        self.waterfall[self.waterfall_index] = fft_array
    #                    self.graph_change = True
    #                else:
    #                    print(fft_array.max())
    #            else:
    #                print("erreur de taille"+str(len(morceau_fft)))
    #                morceau_fft = None
    #                continue
    #            if len(
    #                    self.coefs_ffts) > NOMBRE_FFT_RECONNAISSANCE:  # on attend d'avoir quelques échantillons pour éviter de valier un seul faux positif
    #                self.donnees = np.array(self.coefs_ffts)
    #                if self.donnees.max() > 1:#900:
    #                    print("prédiction")
    #                    # classe_pred, probas = ml.predire_classe_probas(self.donnees)
    #                    classe_pred, probas, autorise = self.ml.autoriser_personne_probas(self.donnees)
    #                    if autorise:
    #                        self.serial_port.write(1)
    #                        pers = Personne.get(Personne.nom == classe_pred)
    #                        Entree.create(personne=pers, pourcentage_confiance=probas[classe_pred]) #on enregistre le passage de la personne
    #                    self.afficher_nom(classe_pred, autorise)
    #                    self.afficher_probas(probas)
    #                    # if classe_pred in classes_valides:
    #                    #    print("Personne autorisée à entrer !")
    #                    #    print(f.renderText(classe_pred))
    #                    # self.coefs_fft_mean = [np.mean(x) for x in self.donnees.transpose()]
    #                    self.coefs_ffts = []  # on reset
    #                else:
    #                    print("le maximum d'amplitude est inférieur à 900, on considère que personne n'a parlé")
    #            morceau_fft = None  # pour bien faire sortir les erreurs
    def analyse_detection(self, donnees):
        print("prédiction")
        # classe_pred, probas = ml.predire_classe_probas(self.donnees)
        classe_pred, probas, autorise = self.ml.autoriser_personne_probas(donnees)
        if autorise:
            self.serial_port.write(1)
            pers = Personne.get(Personne.nom == classe_pred)
            Entree.create(personne=pers,
                          pourcentage_confiance=probas[classe_pred])  # on enregistre le passage de la personne
        self.afficher_nom(classe_pred, autorise)
        self.afficher_probas(probas)

    def read_serial(self, analyse: Callable, repeter=True):
        if repeter:
            NOMBRE_FFT_REQUIS = NOMBRE_FFT_RECONNAISSANCE
        else:
            NOMBRE_FFT_REQUIS = NOMBRE_FFT_ENREGISTREMENT
        morceau_fft = None
        self.coefs_ffts = []
        loop = True
        self.reconnaissance_active = True
        while self.reconnaissance_active and loop:
            ligne = self.serial_port.readline().replace(b'\r\n', b'')
            print_debug(ligne, end=" ; ")
            if ligne == b'restart':
                self.waterfall, self.waterfall_index = [], 0
                print_debug("Remise à zéro des tableaux, parlez maintenant")
                self.coefs_ffts = []
                morceau_fft = []
                continue
            if ligne == b"begin":
                morceau_fft = []  # une transformée de Fourier
                continue
            # if ligne != b'end' and ligne != b'begin' and ligne !=b'\n' and ligne != b'' and ligne != b'restart' and morceau_fft is not None:
            # print(ligne)
            try:
                nombre = float(ligne.decode('utf-8'))
                if ligne != 'end' and morceau_fft is not None:
                    morceau_fft.append(nombre / 100)
            except (UnicodeDecodeError, ValueError):
                pass
            if ligne == b'end' and morceau_fft is not None:
                print_debug("\nlongeur: {}".format(len(morceau_fft)))
                if len(morceau_fft) == 62:
                    fft_array = np.array(morceau_fft)
                    if fft_array.max() > SEUIL_DETECTION:  # SEUIL_DETECTION:
                        print("\r[OUI]", end='')
                        self.coefs_ffts.append(fft_array)
                        print_debug("nouveau morceau dans coefs_ffts")
                        array_pour_waterfall = transformation_coefs(fft_array)
                        if len(self.waterfall) <= NOMBRE_FFT_RECONNAISSANCE:
                            self.waterfall.append(array_pour_waterfall)
                        else:
                            if self.waterfall_index >= len(self.waterfall) - 1:
                                self.waterfall_index = 0
                            else:
                                self.waterfall_index += 1
                            self.waterfall[self.waterfall_index] = array_pour_waterfall
                        self.graph_change = True
                    else:
                        print("\r[NON]", end='')
                        print_debug(fft_array.max())
                else:
                    print("erreur de taille" + str(len(morceau_fft)))
                    morceau_fft = None
                    continue
                if len(
                        self.coefs_ffts) > NOMBRE_FFT_REQUIS:  # on attend d'avoir quelques échantillons pour éviter de valier un seul faux positif
                    self.donnees = np.array(self.coefs_ffts)
                    analyse(self.donnees)
                    self.coefs_ffts = []
                    loop = repeter  # pour finir la boucle si pas besoin de repeter
                morceau_fft = None  # pour bien faire sortir les erreurs

    def stop_reconnaissance_vocale(self):
        self.reconnaissance_active = False

    def afficher_graphique(self):
        plt.matshow(self.waterfall)

    def setup_serial(self):
        try:
            ports = list_ports.comports()
            for port in ports:
                if "Arduino" in port.description:
                    print("Configuration de la carte {} branchée sur le port {}".format(port.description, port.device))
                    self.serial_port = serial.Serial(port=port.device, baudrate=115200, timeout=1, writeTimeout=1)
                    print(self.serial_port)
                    return None  # on sort de la boucle car on va pas configurer plusieurs ports série
            print("Configuration automatique du port série échouée, essai de configuration manuelle")
            if platform.system() == 'Linux':  # Linux
                self.serial_port = serial.Serial(port="/dev/ttyACM0", baudrate=115200, timeout=1, writeTimeout=1)
            elif platform.system() == 'Darwin':  # macOS
                self.serial_port = serial.Serial(port='/dev/cu.usbmodem1A161', baudrate=115200, timeout=1,
                                                 writeTimeout=1)
            else:  # Windows
                self.serial_port = serial.Serial(port="COM4", baudrate=115200, timeout=1, writeTimeout=1)
        except serial.serialutil.SerialException:
            self.serial_port = None
            self.reconnaissance_active = False
            print("Port série non configuré")

    def afficher_probas(self, probas: dict):
        print("  ".join(["{}: {}".format(k, round(v)) for k, v in probas.items()]))

    def voir_matrice_ffts(self, coefs_fft: np.array, nom: str):
        plt.matshow(coefs_fft)

    #  def lancer_enregistrementOLD(self, callback: Optional[Callable]):
    #      morceau_fft = None
    #      self.coefs_ffts = []
    #      if self.serial_port.isOpen():
    #          print("Début enregistrement")
    #          while len(self.coefs_ffts) <= NOMBRE_FFT_RECONNAISSANCE:
    #              ligne = self.serial_port.readline().replace(b'\r\n', b'')
    #              if ligne == b"begin":
    #                  morceau_fft = []  # une transformée de Fourier
    #                  # progessbar.step(len(coefs_ffts))
    #                  continue
    #              if ligne == b'end' and morceau_fft is not None:
    #                  if len(morceau_fft) == 64:
    #                      fft_array = np.array(morceau_fft)
    #                      if fft_array.max() > SEUIL_DETECTION:
    #                          self.coefs_ffts.append(fft_array)
    #                      else:
    #                          print(fft_array.max())
    #                  else:
    #                      morceau_fft = None
    #                      continue
    #              try:
    #                  nombre = float(ligne.decode('utf-8'))
    #                  if morceau_fft is not None:
    #                      morceau_fft.append(nombre)
    #              except (UnicodeDecodeError, ValueError):
    #                  pass
    #          print("Fin enregistrement")
    #          callback()

    def lancer_enregistrement(self, callback: Optional[Callable]):
        def analyse(donnees: list):  # en fait on l'utilise pas
            return None

        if self.serial_port.isOpen():
            print("Début enregistrement")
            self.read_serial(analyse, repeter=False)
            print("Fin enregistrement")
            callback(self.donnees)

    def plot_mfcc_fft(self, coefs_fft):
        for coefs in coefs_fft:
            self.add_plot(np.linspace(1, 13, 13), mfcc(coefs, freq_ech)[0])


class GUI(P2I, tkinter.Tk):  # héritage multiple :)
    morceau_fft = []
    reconnaissance_active = True
    axes = None

    def __init__(self, *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)
        self.title("Reconnaissance vocale GUI")

        # root.geometry("150x50+0+0")
        # main bar
        self.menu_bar = tkinter.Menu(self)
        # Create the submenu (tearoff is if menu can pop out)
        self.file_menu = tkinter.Menu(self.menu_bar, tearoff=0)
        self.detection_menu = tkinter.Menu(self.menu_bar, tearoff=0)
        self.graph_menu = tkinter.Menu(self.menu_bar, tearoff=0)
        self.bdd_menu = tkinter.Menu(self.menu_bar, tearoff=0)

        self.menu_bar.add_cascade(label="Fichier", menu=self.file_menu)
        self.menu_bar.add_cascade(label="Détection", menu=self.detection_menu)
        self.menu_bar.add_cascade(label="Graphique", menu=self.graph_menu)
        self.menu_bar.add_cascade(label="Base de données", menu=self.bdd_menu)
        self.config(menu=self.menu_bar)
        # Add commands to submenu
        self.file_menu.add_command(label="Analyser un ficher audio WAV", command=self.choisir_fichier_et_analyser)
        self.file_menu.add_command(label="Charger un modèle de reconnaissance vocale",
                                   command=self.ouvrir_modele_reconnaissance)
        self.file_menu.add_command(label="Sauvegarder le modèle de reconnaissance vocale",
                                   command=self.enregistrer_modele_reconnaissance)
        self.file_menu.add_command(label="Quitter", command=self.destroy)

        self.detection_menu.add_command(label="Détecter un loctuer avec Arduino",
                                        command=self.lancer_reconnaissance_vocale)
        self.detection_menu.add_command(label="Arrêter la détection", command=self.stop_reconnaissance_vocale)

        self.graph_menu.add_command(label="Afficher le graph tampon", command=self.plot_data)
        self.graph_menu.add_command(label="Effacer le graphique", command=self.reset_graph)

        self.bdd_menu.add_command(label="Gérer", command=self.gerer_bdd)
        self.bdd_menu.add_command(label="Enregistrer un locuteur", command=self.enregistrer_echantillon)
        self.bdd_menu.add_command(label="Récapitulatif", command=self.recap_bdd)
        self.bdd_menu.add_command(label="Statistiques contrôle d'accès avec noms", command=self.stats_sql_historique)

        self.serial_frame = tkinter.Frame(master=self)
        self.serial_frame.pack(fill=tkinter.BOTH)  # Conteneur pour les infos liées à la détéction des voix
        self.graph_frame = tkinter.Frame(master=self)
        self.graph_frame.pack()  # conteneur pour les graphiques

        self.nom = tkinter.Message(self.serial_frame, text="Lancez la reconnaisance vocale", font=('sans-serif', 30),
                                   width=400)  # family="sans-serif"
        self.nom.pack(side=tkinter.TOP, fill=tkinter.BOTH)
        tkinter.Button(master=self.serial_frame, text="Enregistrer un nouvel échantillon",
                       command=self.enregistrer_echantillon).pack(side=tkinter.LEFT)
        tkinter.Button(master=self.serial_frame, text="Oublier la matrice actuelle", command=self.reset_ecoute).pack(
            side=tkinter.LEFT)
        tkinter.Button(master=self.serial_frame, text="Lancer la reconnaissance vocale",
                       command=self.lancer_reconnaissance_vocale).pack(side=tkinter.LEFT)

        self.setup_matplotlib_figure()

        self.affichage_probas = tkinter.Label(master=self, text="probas ici")
        self.affichage_probas.pack(fill=tkinter.BOTH)

        self.config(menu=self.menu_bar)
        self.setup_serial()
        self.mainloop()

    def add_plot(self, X, Y, *args, **kwargs):
        # print("add plot")
        self.fig.add_subplot(111).plot(X, Y, linewidth=1, *args, **kwargs)
        self.canvas.draw()
        self.graph_frame.update()

    def plot(self, X, Y, *args, **kwargs):
        print("clear")
        self.fig.clear()
        self.add_plot(X, Y, *args, **kwargs)

    def setup_matplotlib_figure(self):
        self.fig = Figure(figsize=(5, 3), dpi=120)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        # self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        # self.toolbar.update()
        # self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    def afficher_nom(self, nom: str, autorise: bool):
        if autorise:
            self.nom.configure(text=nom, fg='green')
        else:
            self.nom.configure(text=nom, fg='red')

    def choisir_fichier_et_analyser(self):
        nom_fichier = filedialog.askopenfilename(
            initialdir="./echantillons-test/bastian", title="Choisir un fichier",
            filetypes=(("WAV files", "*.wav"),)
        )
        print(nom_fichier)
        X, Y = [], []
        for coefs in wav_coefs_morceaux(nom_fichier, 2 * 64, T=0.5):
            i = 0
            self.plot_fft(coefs)
            time.sleep(0.5)
        self.data = (X, Y)
        print("plot fini")

    def plot_data(self):
        self.add_plot(self.data[0], self.data[1])

    def reset_graph(self):
        self.fig.clear()
        self.canvas.draw()
        self.graph_frame.update()

    def lancer_reconnaissance_vocale(self):
        classes_valides = ['bastian',
                           'jean']  # les numéros des dossiers avec le nom des personnes à reconnaître 0=bastian, 1=jean
        if self.ml is None:
            self.ml = DetecteurDeVoix()
        print("ml")
        if self.serial_port is not None:
            if self.serial_port.isOpen():
                t = threading.Thread(target=self.reconnaitre_voix)
                t.start()
                self.after(3000, self.afficher_graphique)
            #       self.after(5000, self.reset_graph_loop)
            # self.after(1000, self.afficher_fft_realtime)
            else:
                print("Port série non disponible")
        else:
            print("port série non configuré")

    def stop_reconnaissance_vocale(self):
        self.reconnaissance_active = False

    def afficher_graphique(self):
        #      if self.donnees is None:
        #          return
        #      for coefs in mfcc(self.donnees, freq_ech):
        #          self.add_plot(np.linspace(1, 13, 13), coefs)
        if self.graph_change:
            self.fig.clear()
            self.fig.add_subplot(111).matshow(np.array(self.waterfall))
            self.canvas.draw()
            self.graph_frame.update()
            self.graph_change = not self.graph_change
        self.after(300, self.afficher_graphique)

    def enregistrer_echantillon(self):
        self.coefs_ffts = []
        self.reconnaissance_active = False  # pour arrêter la reconnaisance avant l'enregistrement
        fenetre_rec = tkinter.Toplevel()
        fenetre_rec.title("Enregistrement d'un nouvel échantillon")
        fenetre_rec.pack_propagate()

        progessbar = Progressbar(master=fenetre_rec, mode='determinate')
        # progessbar.pack()
        nom_input = tkinter.Entry(master=fenetre_rec)
        nom_input.pack()
        personne: Personne = None

        def handle_save():
            nom = nom_input.get()
            print([x.nom for x in Personne.select().where(Personne.nom == nom)])
            with open(nom + ".json", "w+") as f:  # enregistrement et fin du prgm
                json.dump([x.tolist() for x in self.coefs_ffts], f)
            f.close()
            personne, b = Personne.get_or_create(nom=nom)
            lt = time.localtime()
            maintenant = str(lt.tm_hour) + ":" + str(lt.tm_min)
            echantillon = Echantillon.create(personne=personne, nom_echantillon=maintenant)
            for tab in self.coefs_ffts:
                morceau = Morceau(echantillon=echantillon)
                morceau.coefs = numpy.array(tab)
                morceau.save()
            fenetre_rec.destroy()  # fini !

        bouton_save = tkinter.Button(master=fenetre_rec, text="Ajouter à la BDD", command=handle_save, state='disabled')
        bouton_save.pack()

        def handle_fin_rec(coefs_ffts: np.array):
            print("finalisation enregistrement")
            progessbar.stop()
            bouton_save.configure(state='normal')
            self.voir_matrice_ffts(coefs_ffts, nom="")
            self.coefs_ffts = coefs_ffts.copy()
            print("données copiées")

        def handle_rec():
            bouton_rec.configure(state='disabled')
            # progessbar.start()
            self.lancer_enregistrement(handle_fin_rec)

        bouton_rec = tkinter.Button(fenetre_rec, text="Démarrer l'enregistrement", command=handle_rec)
        bouton_rec.pack()

        echantillons_view = tkinter.Listbox(master=fenetre_rec, selectmode=tkinter.SINGLE)

        def handle_select(ev):
            nom_input.delete(0, tkinter.END)
            # nom_input.insert(0, Personne.get(Personne.id==echantillons_view.curselection()[0]).nom) #c'est moche
            nom_input.insert(0, echantillons_view.get(echantillons_view.curselection()[0]))
            nom_input.setvar('text', echantillons_view.curselection())

        echantillons_view.bind("<Double-Button-1>", handle_select)
        # echantillons_view.bind("<Button-1>", handle_select)
        echantillons_view.pack()
        for p in Personne.select():
            echantillons_view.insert(tkinter.END, p.nom)
        # fenetre_rec.pack_slaves()
        fenetre_rec.focus()
        # nom = simpledialog.askstring(title="Enregistrement d'un nouvel échantillon", prompt="Nom du locuteur", parent=self)
        # time.sleep(3)
        fenetre_rec.pack_slaves()
        progessbar.stop()

    def gerer_bdd(self):
        fenetre = tkinter.Toplevel()
        self.var_id_personne = tkinter.IntVar()
        fr = tkinter.LabelFrame(master=fenetre, text="Sélectionnez un locuteur")
        fr.pack(fill=tkinter.BOTH)
        for personne in Personne.select():
            r = tkinter.Radiobutton(master=fr, variable=self.var_id_personne, text=personne.nom, value=personne.id)
            r.pack(side='left', expand=1)

        def suppr_pers():
            personne = Personne.get(Personne.id == self.var_id_personne.get())
            for e in personne.echantillons:
                e.delete_instance(recursive=True)
            personne.delete_instance()
            fenetre.destroy()
            self.gerer_bdd()

        bouton_suppr = tkinter.Button(master=fenetre, text="Supprimer", command=suppr_pers)
        bouton_suppr.pack(side=tkinter.LEFT)
        bouton_ech = tkinter.Button(master=fenetre, text="Échantillons >>", command=self.gerer_echantillons)
        bouton_ech.pack(side=tkinter.RIGHT)

        def autoriser():
            personne = Personne.get(Personne.id == self.var_id_personne.get())
            personne.autorisee = not personne.autorisee
            personne.save()

        bouton_autor = tkinter.Button(master=fenetre, text="Autoriser", command=autoriser)
        bouton_autor.pack(side=tkinter.LEFT)

        def lire_personne():
            personne = Personne.get(Personne.id == self.var_id_personne.get())
            audio(personne.nom)

        bouton_play = tkinter.Button(master=fenetre, text="Lire", command=lire_personne)
        bouton_play.pack()
        fenetre.pack_slaves()
        fenetre.focus()

    def gerer_echantillons(self):
        fenetre = tkinter.Toplevel()
        var_id_echantillon = tkinter.IntVar()
        fr = tkinter.LabelFrame(master=fenetre, text="Sélectionnez un échantillon")
        fr.pack(fill=tkinter.BOTH)
        for echantilon in Personne.get(Personne.id == self.var_id_personne.get()).echantillons:
            r = tkinter.Radiobutton(master=fr, variable=var_id_echantillon, value=echantilon.id,
                                    text=echantilon.nom_echantillon)
            r.pack(side=tkinter.LEFT)

        def suppr_ech():
            e = Echantillon.get(Echantillon.id == var_id_echantillon.get())
            e.delete_instance(recursive=True)
            fenetre.destroy()
            self.gerer_echantillons()

        bouton_suppr = tkinter.Button(master=fenetre, text="Supprimer", command=suppr_ech)
        bouton_suppr.pack(side=tkinter.RIGHT)

        nom_ech = tkinter.Entry(master=fenetre, text="Nom")

        def enregistrer_ech():
            # print(nom_ech.get())
            e: Echantillon = Echantillon.get(Echantillon.id == var_id_echantillon.get())
            e.nom_echantillon = nom_ech.get()
            e.save()
            fenetre.destroy()
            self.gerer_echantillons()

        bouton_save = tkinter.Button(master=fenetre, command=enregistrer_ech, text="Enregistrer")

        def reveal_modif():
            bouton_reveal_modif.destroy()
            nom_ech.pack(side=tkinter.LEFT)
            bouton_save.pack(side=tkinter.LEFT)

        bouton_reveal_modif = tkinter.Button(master=fenetre, command=reveal_modif, text="Modifier")
        bouton_reveal_modif.pack(side=tkinter.LEFT)

        def afficher_ech_mat():
            echantilon: Echantillon = Echantillon.get(Echantillon.id == var_id_echantillon.get())
            # coefs_fft = []
            # for morceau in echantilon.morceaux:
            #    coefs_fft.append(morceau.coefs)
            self.voir_matrice_ffts(echantilon.matrice, echantilon.personne.nom)

        #        self.donnees = coefs_fft
        #        self.afficher_graphique()

        bouton_voir_mat = tkinter.Button(master=fenetre, text="Voir matrice FFT", command=afficher_ech_mat)
        bouton_voir_mat.pack()

        def voir_mfcc():
            echantilon: Echantillon = Echantillon.get(Echantillon.id == var_id_echantillon.get())
            coefs_fft = []
            for morceau in echantilon.morceaux:
                coefs_fft.append(morceau.coefs)
            self.voir_matrice_mfcc(coefs_fft, echantilon.personne.nom)

        mfcc_bouton = tkinter.Button(master=fenetre, command=voir_mfcc, text="Voir MFCC")
        mfcc_bouton.pack()

        def lire_echantillon():
            echantilon: Echantillon = Echantillon.get(Echantillon.id == var_id_echantillon.get())
            play_morceau(echantilon.matrice)

        play_button = tkinter.Button(master=fenetre, command=lire_echantillon, text="Synthétiser l'Audio")
        play_button.pack()

    def recap_bdd(self):
        fenetre = tkinter.Toplevel()
        tableau = ttk.Treeview(fenetre)
        tableau['columns'] = ["count(e.id)"]
        tableau.heading(column='#0', text="Nom")
        tableau.heading(column='count(e.id)', text="Nombre d'échantillons")
        # for row in RawQuery("select p.nom, count(e.id) from personne as p, echantillon as e where e.personne_id = p.id group by p.id;").bind(maBDD).execute():
        #    tableau.insert("", 1, "", text=row['nom'], values=[row['id)']])
        for p in Personne.select():
            tp = tableau.insert("", 1, text=p.nom, values=[p.echantillons.count(), ""])
            for e in p.echantillons:
                tableau.insert(tp, "end", text=e.nom_echantillon)
        tableau.pack()

    def afficher_probas(self, probas: dict):
        s = "  ".join(["{}: {}".format(k, round(v)) for k, v in probas.items()])
        self.affichage_probas.configure(text=s)

    def voir_matrice_ffts(self, coefs_fft: np.array, nom: str):
        fenetre = tkinter.Toplevel()
        nom_aff = tkinter.Label(master=fenetre, text=nom)
        nom_aff.pack(fill=tkinter.BOTH)
        fig = Figure(figsize=(5, 4), dpi=100)
        # for coefs in coefs_fft:
        #    fig.add_subplot(111).add_plot(np.linspace(1, 13, 13), mfcc(coefs, freq_ech)[0])
        fig.add_subplot(111).matshow(coefs_fft)
        canvas = FigureCanvasTkAgg(fig, master=fenetre)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        canvas.draw()

        def voir_mfcc():
            self.voir_matrice_mfcc(coefs_fft, nom)

        mfcc_bouton = tkinter.Button(master=fenetre, command=voir_mfcc, text="Voir MFCC")
        mfcc_bouton.pack()

        def lire_echantillon():
            play_morceau(coefs_fft)

        play_button = tkinter.Button(master=fenetre, command=lire_echantillon, text="Synthétiser l'Audio")
        play_button.pack()

    def reset_graph_loop(self):
        """Pas nécessaire pour le waterfall"""
        self.reset_graph()
        self.after(3000, self.reset_graph_loop)

    def reset_ecoute(self):
        self.coefs_ffts = []
        self.donnees = []
        self.waterfall = []
        self.waterfall_index = 0

    def voir_matrice_mfcc(self, coefs_fft: np.array, nom: str):
        fenetre = tkinter.Toplevel()
        nom_aff = tkinter.Label(master=fenetre, text=nom)
        nom_aff.pack(fill=tkinter.BOTH)
        fig = Figure(figsize=(5, 4), dpi=100)
        # for coefs in coefs_fft:
        #    fig.add_subplot(111).add_plot(np.linspace(1, 13, 13), mfcc(coefs, freq_ech)[0])
        fig.add_subplot(111).matshow([transformation_coefs(cs) for cs in coefs_fft])
        canvas = FigureCanvasTkAgg(fig, master=fenetre)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        canvas.draw()

    def stats_sql_historique(self):
        fenetre = tkinter.Toplevel()
        tableau = ttk.Treeview(fenetre)
        tableau['columns'] = ["nom", "n_entrees", "conf_min", "conf_avg"]
        tableau.heading(column='#0', text="Jour")
        tableau.heading(column='nom', text="Nom")
        tableau.heading(column='n_entrees', text="Entrées")
        tableau.heading(column='conf_min', text="Confiance minimale")
        tableau.heading(column='conf_avg', text="Confiance Moyenne")

        tree_map = {}
        donnees = historique_jour_et_nom_rollup()
        for row in donnees:
            if row['jour'] is not None:
                if row['nom'] == None:
                    mr = tableau.insert("", 1, text=str(row['jour']),
                                        values=["", row['n_entrees'], row['conf_min'], row['conf_avg']])
                    tree_map[row['jour']] = mr
        for row in donnees:
            if row['jour'] is not None:
                maitre = tree_map[row['jour']]
                if maitre is not None and row['nom'] is not None:
                    w = row['jour']
                    x = [row['nom'], row['n_entrees'], "", row['conf_avg']]
                    tableau.insert(maitre, "end", text=w, values=x)
        tableau.pack()

    def ouvrir_modele_reconnaissance(self):
        nom_fichier = filedialog.askopenfilename(
            initialdir=".", title="Choisir un fichier Pickle",
            filetypes=(("application/python-pickle", "*.pickle"),)
        )
        print("Chargement du modèle de classification depuis le fichier {}".format(nom_fichier))
        if nom_fichier is not None:  # sinon avec 'None' comme argument, ça  va initialiser le modèle
            self.ml = DetecteurDeVoix(fichier_modele=nom_fichier)

    def enregistrer_modele_reconnaissance(self):
        fichier = tkinter.filedialog.asksaveasfile(mode='wb', defaultextension=".pickle")
        print(fichier)
        if fichier is not None:
            if self.ml is not None:
                self.ml.enregistrer_modele(fichier=fichier)
            else:
                print("Erreur: aucun modèle de reconnaissance vocale en mémoire - lancez la reconnaissance vocale")
        fichier.close()
