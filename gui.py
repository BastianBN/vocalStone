import threading
import time
import tkinter
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from classificateur import wav_coefs_morceaux
import time
from serial import Serial
from classificateur import *
import platform
from pyfiglet import Figlet

def commande_test():
    print("test")


if platform.system() == 'Linux':  # Linux
    serial_port = Serial(port="/dev/ttyACM0", baudrate=115200, timeout=1, writeTimeout=1)
elif platform.system() == 'Darwin':  # macOS
    serial_port = Serial(port='/dev/cu.usbmodem1A151', baudrate=115200, timeout=1, writeTimeout=1)
else:  # Windows
    serial_port = Serial(port="COM3", baudrate=115200, timeout=1, writeTimeout=1)

class P2IGUI(tkinter.Tk):
    coefs_fft_mean: List = []
    fft_time_series = [[],[],[],[],[],[],[],[],[],[]]
    def __init__(self, *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)
        self.title("Reconnaissance vocale GUI")

        # root.geometry("150x50+0+0")
        # main bar
        self.menu_bar = tkinter.Menu(self)
        # Create the submenu (tearoff is if menu can pop out)
        self.file_menu = tkinter.Menu(self.menu_bar, tearoff=0)
        self.graph_menu = tkinter.Menu(self.menu_bar, tearoff=0)
        self.test_menu = tkinter.Menu(self.menu_bar, tearoff=0)
        # Add commands to submenu
        self.file_menu.add_command(label="Analyser un ficher audio WAV", command=self.choisir_fichier_et_analyser)
        self.file_menu.add_command(label="Détecter un loctuer avec Arduino", command=self.reconnaitre_voix)
        self.file_menu.add_command(label="Quitter", command=self.destroy)

        self.test_menu.add_command(label="Exécuter commande_test", command=commande_test)
        self.test_menu.add_command(label="Courbe 2", command=self.courbe2)
        self.test_menu.add_command(label="Courbe 3", command=self.courbe3)
        self.test_menu.add_command(label="Afficher Bastian", command=self.afficher_bastian)
        self.test_menu.add_command(label="Thread test", command=self.long_test)

        self.serial_frame = tkinter.Frame(master=self)
        self.serial_frame.pack()  # Conteneur pour les infos liées à la détéction des voix
        self.graph_frame = tkinter.Frame(master=self)
        self.graph_frame.pack()  # conteneur pour les graphiques

        self.nom = tkinter.Message(self.serial_frame, text="Jean Ribes", font=('times', 48))
        self.nom.pack()

        self.setup_matplotlib_figure()

        self.graph_menu.add_command(label="Enregistrer le graphique", command=self.toolbar.save_figure)
        self.graph_menu.add_command(label="Afficher le graph tampon", command=self.plot_data)
        self.graph_menu.add_command(label="Effacer le graphique", command=self.reset_graph)

        # Add the "File" drop down sub-menu in the main menu bar
        self.menu_bar.add_cascade(label="Fichier", menu=self.file_menu)
        self.menu_bar.add_cascade(label="Test", menu=self.test_menu)
        self.menu_bar.add_cascade(label="Graphique", menu=self.graph_menu)
        self.config(menu=self.menu_bar)

        self.mainloop()

    def courbe2(self):
        pass

    def courbe3(self):
        x = np.linspace(1, 5)
        y = np.power(x, 2)
        self.add_plot(x, y)
        # self.fig.add_subplot(111).plot(x, y)
        # self.canvas.draw()
        # self.graph_frame.update()

    def add_plot(self, X, Y, *args, **kwargs):
        #print("add plot")
        self.fig.add_subplot(111).plot(X, Y, linewidth=1, *args, **kwargs)
        self.canvas.draw()
        self.graph_frame.update()

    def plot(self, X, Y, *args, **kwargs):
        print("clear")
        self.fig.clear()
        self.add_plot(X, Y, *args, **kwargs)

    def setup_matplotlib_figure(self):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)  # A tk.DrawingArea.
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        # self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    def afficher_nom(self, nom: str):
        self.nom.configure(text=nom)

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
        X,Y,i=[],[],0
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
        for coefs in wav_coefs_morceaux(nom_fichier, 2*64, T=0.5):
            i=0
            self.plot_fft(coefs)
            time.sleep(0.5)
        self.data=(X,Y)
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
            morceau_fft=None
            coefs_ffts = []
            index_sw=0
            while True:
                ligne = serial_port.readline().replace(b'\r\n', b'')
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
                        # self.plot_fft(morceau_fft)
                    else:
                        morceau_fft = None
                        continue
                    if len(
                            coefs_ffts) > 20:  # on attend d'avoir quelques échantillons pour éviter de valier un seul faux positif
                        self.donnees = np.array(coefs_ffts)
                        classe_pred = ml.predire_classe_texte(self.donnees)
                        print(classe_pred)
                        self.afficher_nom(classe_pred)
                        # if classe_pred in classes_valides:
                        #    print("Personne autorisée à entrer !")
                        #    print(f.renderText(classe_pred))
                        self.coefs_fft_mean = [np.mean(x) for x in self.donnees.transpose()]
                        self.fft_time_series[index_sw]=self.coefs_fft_mean
                        if index_sw>8:
                            index_sw=0
                        else:
                            index_sw+=1
                        coefs_ffts = []  # on reset
                    morceau_fft = None  # pour bien faire sortir les erreurs

        ml = DetecteurDeVoix()
        if serial_port.isOpen():
            t = threading.Thread(target=callback)
            t.start()
            self.after(1000, self.afficher_fft_realtime)
            #self.after(10000, self.reset_graph_loop)

    #def OLDafficher_fft_realtime(self):
    #    if len(self.donnees)>0:
    #        self.plot_fft(self.coefs_fft_mean)
    #        self.after(500, self.OLDafficher_fft_realtime)

    def afficher_fft_realtime(self):
        if len(self.fft_time_series)>0:
            self.fig.clear()
            for ffts in self.fft_time_series:
                self.plot_fft(self.coefs_fft_mean)
            self.after(500, self.afficher_fft_realtime)


    def reset_graph_loop(self):
        self.fig.clear()
        self.after(10000, self.reset_graph_loop)

g = P2IGUI()
