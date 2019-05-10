import threading
import time
import tkinter
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from classificateur import wav_coefs_morceaux


def commande_test():
    print("test")


class P2IGUI(tkinter.Tk):
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
        self.file_menu.add_command(label="Quit!", command=self.destroy)
        self.file_menu.add_command(label="Exit!", command=self.destroy)
        self.file_menu.add_command(label="End!", command=self.destroy)

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
        print("add plot")
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

    def choisir_fichier_et_analyser(self):
        nom_fichier = filedialog.askopenfilename(
            initialdir="./echantillons-test/bastian", title="Choisir un fichier",
            filetypes=(("WAV files", "*.wav"),)
        )
        print(nom_fichier)
        X, Y = [], []
        for coefs in wav_coefs_morceaux(nom_fichier, 2*64, T=0.5):
            i=0
            for coef in coefs:
                X.append(i)
                X.append(i)
                X.append(i)
                Y.append(0)
                Y.append(coef)
                Y.append(0)
                i += 1
            self.add_plot(X,Y)
            time.sleep(0.5)
        self.data=(X,Y)
        print("plot fini")

    def plot_data(self):
        self.add_plot(self.data[0], self.data[1])

    def reset_graph(self):
        self.fig.clear()
        self.canvas.draw()
        self.graph_frame.update()
g = P2IGUI()
g.afficher_nom("Bastian")
