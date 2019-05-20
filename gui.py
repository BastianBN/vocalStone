import json
import platform
import threading
import tkinter
from tkinter import filedialog, ttk
from tkinter.ttk import Progressbar

import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from serial.tools import list_ports

from classificateur import *


def commande_test():
    print("test")


class P2IGUI(tkinter.Tk):
    coefs_fft_mean: List = []
    fft_time_series = [[], [], [], [], [], [], [], [], [], []]
    morceau_fft = []
    reconnaissance_active = True

    waterfall = [np.linspace(0, 100, 64)]
    waterfall_index=0
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
        self.test_menu = tkinter.Menu(self.menu_bar, tearoff=0)
        self.bdd_menu = tkinter.Menu(self.menu_bar, tearoff=0)

        self.menu_bar.add_cascade(label="Fichier", menu=self.file_menu)
        self.menu_bar.add_cascade(label="Détection", menu=self.detection_menu)
        self.menu_bar.add_cascade(label="Test", menu=self.test_menu)
        self.menu_bar.add_cascade(label="Graphique", menu=self.graph_menu)
        self.menu_bar.add_cascade(label="Base de données", menu=self.bdd_menu)
        self.config(menu=self.menu_bar)
        # Add commands to submenu
        self.file_menu.add_command(label="Analyser un ficher audio WAV", command=self.choisir_fichier_et_analyser)
        self.file_menu.add_command(label="Quitter", command=self.destroy)

        self.detection_menu.add_command(label="Détecter un loctuer avec Arduino", command=self.reconnaitre_voix)
        self.detection_menu.add_command(label="Arrêter la détection", command=self.stop_reconnaissance_vocale)

        self.test_menu.add_command(label="Exécuter commande_test", command=commande_test)
        self.test_menu.add_command(label="Courbe 2", command=self.courbe2)
        self.test_menu.add_command(label="Courbe 3", command=self.courbe3)
        self.test_menu.add_command(label="Afficher Bastian", command=self.afficher_bastian)
        self.test_menu.add_command(label="Thread test", command=self.long_test)

        self.graph_menu.add_command(label="Afficher le graph tampon", command=self.plot_data)
        self.graph_menu.add_command(label="Effacer le graphique", command=self.reset_graph)

        self.bdd_menu.add_command(label="Gérer", command=self.gerer_bdd)
        self.bdd_menu.add_command(label="Enregistrer un locuteur", command=self.enregistrer_echantillon)
        self.bdd_menu.add_command(label="Récapitulatif", command=self.recap_bdd)

        self.serial_frame = tkinter.Frame(master=self)
        self.serial_frame.pack(fill=tkinter.BOTH)  # Conteneur pour les infos liées à la détéction des voix
        self.graph_frame = tkinter.Frame(master=self)
        self.graph_frame.pack()  # conteneur pour les graphiques

        self.nom = tkinter.Message(self.serial_frame, text="Lancez la reconnaisance vocale", font=('sans-serif', 30),
                                   width=400)  # family="sans-serif"
        self.nom.pack(side=tkinter.TOP, fill=tkinter.BOTH)
        tkinter.Button(master=self.serial_frame, text="Enregistrer un nouvel échantillon",
                       command=self.enregistrer_echantillon).pack(side=tkinter.LEFT)
        tkinter.Button(master=self.serial_frame, text="Modifier la base de données", command=self.gerer_bdd).pack(
            side=tkinter.LEFT)
        tkinter.Button(master=self.serial_frame, text="Lancer la reconnaissance vocale",
                       command=self.reconnaitre_voix).pack(side=tkinter.LEFT)

        self.setup_matplotlib_figure()

        self.affichage_probas = tkinter.Label(master=self, text="probas ici")
        self.affichage_probas.pack(fill=tkinter.BOTH)

        self.config(menu=self.menu_bar)
        self.setup_serial()
        self.mainloop()

    def courbe2(self):
        x = np.linspace(1, 10)
        y = np.exp(x)
        self.plot(x, y)
        pass

    def courbe3(self):
        x = np.linspace(1, 5)
        y = np.power(x, 2)
        self.add_plot(x, y)
        # self.fig.add_subplot(111).plot(x, y)
        # self.canvas.draw()
        # self.graph_frame.update()

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
        self.fig = Figure(figsize=(3,4), dpi=120)
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

    def afficher_bastian(self):
        self.afficher_nom("bastian")

    def long_test(self):
        def callback():
            print("thread start")
            time.sleep(3)
            print("thread end")
            self.afficher_nom("thread finished")

        t = threading.Thread(target=callback)
        t.start()
        print("thread started")
        self.afficher_nom("thread started")

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

    def reconnaitre_voix(self):
        classes_valides = ['bastian',
                           'jean']  # les numéros des dossiers avec le nom des personnes à reconnaître 0=bastian, 1=jean

        def callback():
            morceau_fft = None
            coefs_ffts = []
            while self.reconnaissance_active:
                ligne = self.serial_port.readline().replace(b'\r\n', b'')
                if ligne == b'restart':
                    print("Remise à zéro des tableaux, parlez maintenant")
                    coefs_ffts = []
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
                        morceau_fft.append(nombre)
                except (UnicodeDecodeError, ValueError):
                    pass
                if ligne == b'end' and morceau_fft is not None:
                    if len(morceau_fft) == 64:
                        coefs_ffts.append(np.array(morceau_fft))
                        self.morceau_fft = np.array(morceau_fft)
                        if len(self.waterfall) <= 41:
                            self.waterfall.append(self.morceau_fft)
                        else:
                            if self.waterfall_index >= len(self.waterfall)-1:
                                self.waterfall_index = 0
                            else:
                                self.waterfall_index += 1
                            self.waterfall[self.waterfall_index] = morceau_fft
                        # self.plot_fft(morceau_fft)
                    else:
                        morceau_fft = None
                        continue
                    if len(
                            coefs_ffts) > 40:  # on attend d'avoir quelques échantillons pour éviter de valier un seul faux positif
                        self.donnees = np.array(coefs_ffts)
                        #classe_pred, probas = ml.predire_classe_probas(self.donnees)
                        classe_pred, probas, autorise = ml.autoriser_personne_probas(self.donnees)
                        print(classe_pred)
                        self.afficher_nom(classe_pred, autorise)
                        self.afficher_probas(probas)
                        # if classe_pred in classes_valides:
                        #    print("Personne autorisée à entrer !")
                        #    print(f.renderText(classe_pred))

                        # self.coefs_fft_mean = [np.mean(x) for x in self.donnees.transpose()]
                        coefs_ffts = []  # on reset
                    morceau_fft = None  # pour bien faire sortir les erreurs

        if self.serial_port is not None:
            ml = DetecteurDeVoix()
            if self.serial_port.isOpen():
                t = threading.Thread(target=callback)
                self.reconnaissance_active = True
                t.start()
                self.after(300, self.afficher_waterfall)
                # self.after(1000, self.afficher_fft_realtime)
            # self.after(2000, self.reset_graph_loop)
            else:
                print("Port série non disponible")
        else:
            print("port série non config")

    def stop_reconnaissance_vocale(self):
        self.reconnaissance_active = False

    def afficher_fft_realtime(self):
        if self.donnees is not None:
            # self.fig.clear()

            # self.coefs_fft_mean[63]=100
            # self.plot_fft(self.coefs_fft_mean)
            # to_display=[]
            # for ffts in np.array(self.fft_time_series).transpose():
            #    to_display.append(np.mean(ffts))
            # to_display = [np.mean(ffts) for ffts in np.array(self.fft_time_series).transpose()]
            # to_display[63]=100
            # self.plot_fft(to_display)

            # self.morceau_fft[63] = 100
            self.plot_fft(self.morceau_fft)
            self.fig.clear()

            self.after(100, self.afficher_fft_realtime)

    def reset_graph_loop(self):
        self.fig.clear()
        self.after(1500, self.reset_graph_loop)

    def afficher_waterfall(self):
        self.fig.clear()
        self.fig.add_subplot(111).matshow(np.array(self.waterfall))
        self.canvas.draw()
        self.graph_frame.update()
        self.after(50, self.afficher_waterfall)

    def enregistrer_echantillon(self):
        coefs_ffts = []
        fenetre_rec = tkinter.Toplevel()
        fenetre_rec.title("Enregistrement d'un nouvel échantillon")
        fenetre_rec.pack_propagate()

        progessbar = Progressbar(master=fenetre_rec, mode='determinate')
        # progessbar.pack()
        nom_input = tkinter.Entry(master=fenetre_rec)
        nom_input.pack()

        def handle_save():
            nom = nom_input.get()
            print([x.nom for x in Personne.select().where(Personne.nom == nom)])
            with open(nom + ".json", "w+") as f:  # enregistrement et fin du prgm
                json.dump(coefs_ffts, f)
            f.close()
            personne, b = Personne.get_or_create(nom=nom)
            lt = time.localtime()
            maintenant = str(lt.tm_hour) + ":" + str(lt.tm_min)
            echantillon = Echantillon.create(personne=personne, nom_echantillon=maintenant)
            for tab in coefs_ffts:
                morceau = Morceau(echantillon=echantillon)
                morceau.coefs = numpy.array(tab)
                morceau.save()
            fenetre_rec.destroy()  # fini !
            self.voir_matrice_ffts(np.array(coefs_ffts), personne.nom)

        bouton_save = tkinter.Button(master=fenetre_rec, text="Ajouter à la BDD", command=handle_save, state='disabled')
        bouton_save.pack()

        def handle_fin_rec():
            progessbar.stop()
            bouton_save.configure(state='normal')

        def handle_rec():
            bouton_rec.configure(state='disabled')
            # progessbar.start()
            morceau_fft = None
            if self.serial_port.isOpen():
                print("Début enregistrement")
                while len(coefs_ffts) <= 40:
                    ligne = self.serial_port.readline().replace(b'\r\n', b'')
                    if ligne == b"begin":
                        morceau_fft = []  # une transformée de Fourier
                        # progessbar.step(len(coefs_ffts))
                        continue
                    if ligne != b'end' and ligne != b'begin' and ligne != b'\n' and ligne != b'' and morceau_fft is not None:
                        nombre = float(ligne.decode('utf-8'))
                        if ligne != 'end':
                            morceau_fft.append(nombre)
                    if ligne == b'end' and morceau_fft is not None:
                        if len(morceau_fft) == 64:
                            coefs_ffts.append(morceau_fft)
                        else:
                            morceau_fft = None
                            continue

            fenetre_rec.after(4000, handle_fin_rec)

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

    def setup_serial(self):
        ports = list_ports.comports()
        for port in ports:
            if "Arduino" in port.description:
                print("Configuration de la carte {} branchée sur le port {}".format(port.description, port.device))
                self.serial_port = serial.Serial(port=port.device, baudrate=115200, timeout=1, writeTimeout=1)
                print(self.serial_port)
                return None  # on sort de la boucle car on va pas configurer plusieurs ports série
        print("Configuration automatique du port série échouée, essai de configuration manuelle")
        try:
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
            coefs_fft = []
            for morceau in echantilon.morceaux:
                coefs_fft.append(morceau.coefs)
            self.voir_matrice_ffts(np.array(coefs_fft), echantilon.personne.nom)

        bouton_voir_mat = tkinter.Button(master=fenetre, text="Voir matrice FFT", command=afficher_ech_mat)
        bouton_voir_mat.pack()

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
                tableau.insert(tp, "end", "", text=e.nom_echantillon)
        tableau.pack()

    def afficher_probas(self, probas: dict):
        s = "  ".join(["{}: {}".format(k, round(v)) for k, v in probas.items()])
        self.affichage_probas.configure(text=s)

    def voir_matrice_ffts(self, coefs_fft: np.array, nom: str):
        fenetre = tkinter.Toplevel()
        nom_aff = tkinter.Label(master=fenetre, text=nom)
        nom_aff.pack(fill=tkinter.BOTH)
        fig = Figure(figsize=(5, 4), dpi=100)
        fig.add_subplot(111).matshow(coefs_fft)
        canvas = FigureCanvasTkAgg(fig, master=fenetre)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        canvas.draw()

    #    self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)  # A tk.DrawingArea.
    #    self.canvas.draw()
    #    self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


g = P2IGUI()
