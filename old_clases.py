import platform
import threading
import tkinter
from tkinter import filedialog
from typing import Optional, Callable

import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from serial.tools import list_ports

from classificateur import *

SEUIL_DETECTION = 500
NOMBRE_FFT_RECONNAISSANCE = 40  # matrice de (x,64) coefficients de fourier
SEUIL_VECTEUR_DETECTION = 600
VERBOSE = False

class P2I(object):
    reconnaissance_active = True

    waterfall = [np.linspace(0, 100, 64)]
    waterfall_index = 0
    serial_port: serial.Serial

    graph_change = False

    def __init__(self):
        self.setup_serial()
        self.lancer_reconnaissance_vocale()

    def plot(self, X, Y, *args, **kwargs):
        plt.plot(X, Y, *args, **kwargs)

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
        morceau_fft = None
        self.coefs_ffts = []
        while self.reconnaissance_active:
            ligne = self.serial_port.readline().replace(b'\r\n', b'')
      #      print(ligne, end=" ; ")
            if ligne == b'restart':
                self.waterfall, self.waterfall_index = [], 0
                print("Remise à zéro des tableaux, parlez maintenant")
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
     #           print("\nlongeur: {}".format(len(morceau_fft)))
                if len(morceau_fft) == 62:
                    fft_array = np.array(morceau_fft)
                    print("\rmax: {}                    ".format(fft_array.max()), end='')
                    if fft_array.max() > SEUIL_VECTEUR_DETECTION:  # SEUIL_DETECTION:
                        self.coefs_ffts.append(fft_array)
         #               print("nouveau morceau dans coefs_ffts")
                        if len(self.waterfall) <= NOMBRE_FFT_RECONNAISSANCE:
                            self.waterfall.append(fft_array)
                        else:
                            if self.waterfall_index >= len(self.waterfall) - 1:
                                self.waterfall_index = 0
                            else:
                                self.waterfall_index += 1
                            self.waterfall[self.waterfall_index] = fft_array
                        self.graph_change = True
                    else:
                        pass
                else:
      #              print("erreur de taille" + str(len(morceau_fft)))
                    morceau_fft = None
                    continue
                if len(
                        self.coefs_ffts) > NOMBRE_FFT_RECONNAISSANCE:  # on attend d'avoir quelques échantillons pour éviter de valier un seul faux positif
                    self.donnees = np.array(self.coefs_ffts)
                    if self.donnees.max() > 900:  # 900:
                        pass
                    else:
                        print()
                        print("le maximum d'amplitude est inférieur à 900, on considère que personne n'a parlé")
                    self.coefs_ffts = []  # on reset
    morceau_fft = None  # pour bien faire sortir les erreurs

    def read_serial(self, analyse: Callable, repeter=True):
        morceau_fft = None
        self.coefs_ffts = []
        loop = True
        while self.reconnaissance_active and loop:
            ligne = self.serial_port.readline().replace(b'\r\n', b'')
        #    print(ligne, end=" ; ")
            if ligne == b'restart':
                self.waterfall, self.waterfall_index = [], 0
                print("Remise à zéro des tableaux, parlez maintenant")
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
                print("\nlongeur: {}".format(len(morceau_fft)))
                if len(morceau_fft) == 62:
                    fft_array = np.array(morceau_fft)
                    if fft_array.max() > 1:  # SEUIL_DETECTION:
                        self.coefs_ffts.append(fft_array)
                        print("nouveau morceau dans coefs_ffts")
                        if len(self.waterfall) <= NOMBRE_FFT_RECONNAISSANCE:
                            self.waterfall.append(fft_array)
                        else:
                            if self.waterfall_index >= len(self.waterfall) - 1:
                                self.waterfall_index = 0
                            else:
                                self.waterfall_index += 1
                            self.waterfall[self.waterfall_index] = fft_array
                        self.graph_change = True
                    else:
                        print(fft_array.max())
                else:
                    print("erreur de taille" + str(len(morceau_fft)))
                    morceau_fft = None
                    continue
                if len(
                        self.coefs_ffts) > NOMBRE_FFT_RECONNAISSANCE:  # on attend d'avoir quelques échantillons pour éviter de valier un seul faux positif
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

    def lancer_enregistrementOLD(self, callback: Optional[Callable]):
        morceau_fft = None
        self.coefs_ffts = []
        if self.serial_port.isOpen():
            print("Début enregistrement")
            while len(self.coefs_ffts) <= NOMBRE_FFT_RECONNAISSANCE:
                ligne = self.serial_port.readline().replace(b'\r\n', b'')
                if ligne == b"begin":
                    morceau_fft = []  # une transformée de Fourier
                    # progessbar.step(len(coefs_ffts))
                    continue
                if ligne == b'end' and morceau_fft is not None:
                    if len(morceau_fft) == 64:
                        fft_array = np.array(morceau_fft)
                        if fft_array.max() > SEUIL_DETECTION:
                            self.coefs_ffts.append(fft_array)
                        else:
                            print(fft_array.max())
                    else:
                        morceau_fft = None
                        continue
                try:
                    nombre = float(ligne.decode('utf-8'))
                    if morceau_fft is not None:
                        morceau_fft.append(nombre)
                except (UnicodeDecodeError, ValueError):
                    pass
            print("Fin enregistrement")
            callback()

    def lancer_enregistrement(self, callback: Optional[Callable]):
        morceau_fft = None
        coefs_ffts = None

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
        self.config(menu=self.menu_bar)
        # Add commands to submenu
        self.file_menu.add_command(label="Analyser un ficher audio WAV", command=self.choisir_fichier_et_analyser)
        self.file_menu.add_command(label="Quitter", command=self.destroy)

        self.detection_menu.add_command(label="Détecter un loctuer avec Arduino",
                                        command=self.lancer_reconnaissance_vocale)
        self.detection_menu.add_command(label="Arrêter la détection", command=self.stop_reconnaissance_vocale)

        self.graph_menu.add_command(label="Afficher le graph tampon", command=self.plot_data)
        self.graph_menu.add_command(label="Effacer le graphique", command=self.reset_graph)

        self.serial_frame = tkinter.Frame(master=self)
        self.serial_frame.pack(fill=tkinter.BOTH)  # Conteneur pour les infos liées à la détéction des voix
        self.graph_frame = tkinter.Frame(master=self)
        self.graph_frame.pack()  # conteneur pour les graphiques

        self.nom = tkinter.Message(self.serial_frame, text="Lancez la reconnaisance vocale", font=('sans-serif', 30),
                                   width=400)  # family="sans-serif"
        self.nom.pack(side=tkinter.TOP, fill=tkinter.BOTH)
        tkinter.Button(master=self.serial_frame, text="Lancer la reconnaissance vocale",
                       command=self.lancer_reconnaissance_vocale).pack(side=tkinter.LEFT)

        self.setup_matplotlib_figure()

        self.affichage_probas = tkinter.Label(master=self, text="probas ici")
        self.affichage_probas.pack(fill=tkinter.BOTH)

        self.config(menu=self.menu_bar)
        self.setup_serial()
        self.after(300, self.lancer_reconnaissance_vocale)
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
        if self.serial_port is not None:
            if self.serial_port.isOpen():
                t = threading.Thread(target=self.reconnaitre_voix)
                self.reconnaissance_active = True
                t.start()
                self.after(3000, self.afficher_graphique)
                #self.after(5000, self.reset_graph_loop)
                # self.after(1000, self.afficher_fft_realtime)
            # self.after(2000, self.reset_graph_loop)
            else:
                print("Port série non disponible")
        else:
            print("port série non config")

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

    # s = P2I() #test terminal

    def reset_graph_loop(self):
        self.reset_graph()
        self.after(3000, self.reset_graph_loop)

GUI()

