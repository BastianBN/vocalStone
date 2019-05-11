import json
import threading
import time
import tkinter
from tkinter import filedialog, simpledialog, ttk
from tkinter.ttk import Progressbar

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
    serial_port = Serial(port='/dev/cu.usbmodem1A161', baudrate=115200, timeout=1, writeTimeout=1)
else:  # Windows
    serial_port = Serial(port="COM3", baudrate=115200, timeout=1, writeTimeout=1)

class P2IGUI(tkinter.Tk):
    coefs_fft_mean: List = []
    fft_time_series = [[],[],[],[],[],[],[],[],[],[]]
    morceau_fft=[]
    reconnaissance_active=True
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
        self.file_menu.add_command(label="Enregistrer un locuteur", command=self.enregistrer_echantillon)
        self.file_menu.add_command(label="Détecter un loctuer avec Arduino", command=self.reconnaitre_voix)
        self.file_menu.add_command(label="Arrêter la détection", command=self.stop_reconnaissance_vocale)
        self.file_menu.add_command(label="Quitter", command=self.destroy)

        self.test_menu.add_command(label="Exécuter commande_test", command=commande_test)
        self.test_menu.add_command(label="Courbe 2", command=self.courbe2)
        self.test_menu.add_command(label="Courbe 3", command=self.courbe3)
        self.test_menu.add_command(label="Afficher Bastian", command=self.afficher_bastian)
        self.test_menu.add_command(label="Thread test", command=self.long_test)

        self.graph_menu.add_command(label="Afficher le graph tampon", command=self.plot_data)
        self.graph_menu.add_command(label="Effacer le graphique", command=self.reset_graph)

        self.serial_frame = tkinter.Frame(master=self)
        self.serial_frame.pack()  # Conteneur pour les infos liées à la détéction des voix
        self.graph_frame = tkinter.Frame(master=self)
        self.graph_frame.pack()  # conteneur pour les graphiques

        self.nom = tkinter.Message(self.serial_frame, text="Lancez la reconnaisance vocale", font=('sans-serif', 30), width=400)
        #family="sans-serif"
        self.nom.pack()

        self.setup_matplotlib_figure()



        # Add the "File" drop down sub-menu in the main menu bar
        self.menu_bar.add_cascade(label="Fichier", menu=self.file_menu)
        self.menu_bar.add_cascade(label="Test", menu=self.test_menu)
        self.menu_bar.add_cascade(label="Graphique", menu=self.graph_menu)
        self.config(menu=self.menu_bar)

        self.mainloop()

    def courbe2(self):
        x=np.linspace(1,10)
        y=np.exp(x)
        self.plot(x,y)
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
        #self.toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame)
        # self.toolbar.update()
        #self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

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
            while self.reconnaissance_active:
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
                        self.morceau_fft = np.array(morceau_fft)
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

                        #self.coefs_fft_mean = [np.mean(x) for x in self.donnees.transpose()]
                        coefs_ffts = []  # on reset
                    morceau_fft = None  # pour bien faire sortir les erreurs

        #ml = DetecteurDeVoix()
        ml=None
        if serial_port.isOpen():
            t = threading.Thread(target=callback)
            self.reconnaissance_active=True
            t.start()
            self.after(1000, self.afficher_fft_realtime)
            self.after(2000, self.reset_graph_loop)
    def stop_reconnaissance_vocale(self):
        self.reconnaissance_active=False
    #def OLDafficher_fft_realtime(self):
    #    if len(self.donnees)>0:
    #        self.plot_fft(self.coefs_fft_mean)
    #        self.after(500, self.OLDafficher_fft_realtime)

    def afficher_fft_realtime(self):
        if len(self.fft_time_series)>0:
            #self.fig.clear()

            #self.coefs_fft_mean[63]=100
            #self.plot_fft(self.coefs_fft_mean)
            #to_display=[]
            #for ffts in np.array(self.fft_time_series).transpose():
            #    to_display.append(np.mean(ffts))
            #to_display = [np.mean(ffts) for ffts in np.array(self.fft_time_series).transpose()]
            #to_display[63]=100
            #self.plot_fft(to_display)

            self.morceau_fft[63]=100
            self.plot_fft(self.morceau_fft)

            self.after(100, self.afficher_fft_realtime)


    def reset_graph_loop(self):
        self.fig.clear()
        self.after(1500, self.reset_graph_loop)

    def enregistrer_echantillon(self):
        coefs_ffts = []
        fenetre_rec = tkinter.Toplevel()
        fenetre_rec.title("Enregistrement d'un nouvel échantillon")
        fenetre_rec.pack_propagate()

        progessbar = Progressbar(master=fenetre_rec, mode='determinate')
        #progessbar.pack()
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
            fenetre_rec.destroy()#fini !

        bouton_save = tkinter.Button(master=fenetre_rec, text="Ajouter à la BDD", command=handle_save, state='disabled')
        bouton_save.pack()
        def handle_fin_rec():
            progessbar.stop()
            bouton_save.configure(state='normal')
        def handle_rec():
            bouton_rec.configure(state='disabled')
            #progessbar.start()
            morceau_fft = None
            if serial_port.isOpen():
                print("Début enregistrement")
                while len(coefs_ffts) <= 40:
                    ligne = serial_port.readline().replace(b'\r\n', b'')
                    if ligne == b"begin":
                        morceau_fft = []  # une transformée de Fourier
                        #progessbar.step(len(coefs_ffts))
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
            #nom_input.insert(0, Personne.get(Personne.id==echantillons_view.curselection()[0]).nom) #c'est moche
            nom_input.insert(0, echantillons_view.get(echantillons_view.curselection()[0]))
            nom_input.setvar('text', echantillons_view.curselection())
        echantillons_view.bind("<Double-Button-1>", handle_select)
        #echantillons_view.bind("<Button-1>", handle_select)
        echantillons_view.pack()
        for p in Personne.select():
            echantillons_view.insert(tkinter.END, p.nom)
        #fenetre_rec.pack_slaves()
        fenetre_rec.focus()
        #nom = simpledialog.askstring(title="Enregistrement d'un nouvel échantillon", prompt="Nom du locuteur", parent=self)
        #time.sleep(3)
        fenetre_rec.pack_slaves()
        progessbar.stop()

g = P2IGUI()
