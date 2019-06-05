import json
import os
import pickle
import platform
import re
import threading
import time
import tkinter
from datetime import datetime
from io import BytesIO
from tkinter import filedialog, ttk
from tkinter.ttk import Progressbar
from typing import List, Union
from typing import Optional, Callable

import librosa
import numpy
import numpy as np
import serial
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from peewee import *
from scipy.fftpack import fft
from scipy.io.wavfile import *
from scipy.signal.windows import hamming
from serial.tools import list_ports
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

SEUIL_DETECTION = 500
NOMBRE_FFT_ENREGISTREMENT = 30
NOMBRE_FFT_RECONNAISSANCE = 10  # matrice de (x,64) coefficients de fourier
VERBOSE = False  # pour afficher dans la console les données reçues


freq_ech = 10000
N = 62 * 2  # nombre de coefficients par échantillon, il faut multiplier par 2 à cause de la moitié négative
POURCENTAGE_AUTORISATION = 70



maBDD = MySQLDatabase('G223_B_BD2', user='G223_B', password='G223_B', host='pc-tp-mysql.insa-lyon.fr', port=3306)
#maBDD = PostgresqlDatabase('p2i', user='p2i', password='wheatstone', host='vps.ribes.ovh', port=5432)
maBDD.connect()

class Personne(Model):
    class Meta:
        database = maBDD
    nom = CharField(max_length=255, unique=True)
    autorisee = BooleanField(column_name='autorise', default=False)

class Echantillon(Model):
    """
    Cette classe contient la liste des coefficients de fourier dans une chaîne de caractère et avec des foerign keys
    utiliser ``_liste_coefs`` pour faire les requêtes et ``liste_coefs``pour les opéeations Python
    """
    class Meta:
        database = maBDD
    personne = ForeignKeyField(Personne, backref='echantillons')
    nom_echantillon = CharField(max_length=255)

    @property
    def matrice(self):
        morceaux = []
        for morceau in self.morceaux:
            morceaux.append(morceau.coefs)
        return numpy.array(morceaux)

class Morceau(Model):
    class Meta:
        database = maBDD
    #def __init__(self, echantiloon, _coefs, **kwargs):
    #def __init__(self, **kwargs):
    #    self.coefs = kwargs.get('coefs', None)
    #    super().__init__(**kwargs)
    echantillon = ForeignKeyField(Echantillon, backref='morceaux')
    _coefs = BlobField()

    @property
    def coefs(self):
        return numpy.load(BytesIO(self._coefs))

    @coefs.setter
    def coefs(self, coefs):
        with BytesIO() as b:
            numpy.save(b, coefs)
            self._coefs = b.getvalue()
class Entree(Model):
    class Meta:
        database = maBDD
    personne = ForeignKeyField(Personne, backref='historique')
    horodatage = DateTimeField(default=datetime.now) #pas besoin de le remplir du coup
    pourcentage_confiance = IntegerField() #entre 0 et 100

def enregistrer_entree_historique(classe_pred, probas, autorise):
    pers = Personne.get(Personne.nom==classe_pred)
    if autorise:
        Entree.create(personne=pers, pourcentage_confiance=probas[classe_pred])
    return classe_pred, probas, autorise

def wav_coefs_morceaux(nom_fichier, N= N, T= 0.01):
    """
    Fonction pour faire la transformée de Fourier depuis un fichier
    :param nom_fichier: fichier .wav à lire
    :param N: nombre de coefficients de Fourier réels positifs voulus
    :param T: fenêtre temporelle pour la FFT (10ms par défaut)
    :return: T*(sample rate) listes de N coefficients (matrice 2D)
    """
    fe, audio = read(nom_fichier)  # on lit chaque fichier audio
    morceaux = np.array_split(audio, len(audio) // (fe * T))  # on coupe en 100 morceaux de taille a peu prs egale
    coefs = []
    for morceau in morceaux:
        window = hamming(len(morceau))
        coefs.append(np.abs(fft(morceau * window, N)[0:N // 2]))  # partie réelle positive
    return coefs


def transformation_coefs(coefs): #un vecteur de coefficients
    melspectr = librosa.feature.melspectrogram(S=coefs, n_fft=N, y=None, sr=freq_ech)
    mfcc = librosa.feature.mfcc(S=melspectr, y=None, n_mfcc=13)
    return np.array(mfcc)
    #return mfcc(coefs, freq_ech)[0]

def utilisation_coefs(X, Y,  coefs, label=None,):
    X.append(transformation_coefs(coefs))
    if label is not None: Y.append(label)
    #for liste in transformation_coefs(coefs):
    #    X.append(liste) #au cas où on utilise les MFCC, il faut pouvoir itérer
    #    if label is not None: Y.append(label)

modele_qui_predit = KNeighborsClassifier
#modele_qui_predit=DecisionTreeClassifier


class BaseDetecteur():
    """
    Classe qui inclut tout le nécessaire pour analyser des coefficients de Fourier en machine learning
    """
    N = None
    T = None
    wav_file = re.compile(
        '^.+wav$')  # regexp qui sert à déterminer quels fichiers seront utilisés pour l'apprentissage et le test
    labels = []
    Xlearn, Ylearn = [], []  # listes d'entrainement pour le machine learning

    labels_dict = {0: 'silence'}  # {1:"random", 2: "ljklkj" ...}
    labels_reverse = {'silence': 0}  # {'random':1, 'lkjlkj':2 ... }

    classes_autorisees = ['jean']  # les gens dedans vont être autorisées pas le système

    def __init__(self,
                 fichier_modele=None,
                 modele=None,
                 dossier_apprentissage="echantillons-learn",
                 dossier_test="echantillons-test",
                 N=N,
                 T=0.1):  # "constructeur"

        self.modele = modele
        self.dossier_apprentissage = dossier_apprentissage
        self.dossier_test = dossier_test
        if fichier_modele is not None:
            self.charger_fichier(fichier_modele)
        if self.modele is None:
            self.charger_fichier()
            if self.modele is None:
                self.modele = modele_qui_predit()
                self.entrainer_modele()
        self.N = N
        self.T = T
        self.t1 = time.time()
        print(self.modele)

    def charger_fichier(self, nom_fichier="decisiontree.pickle"):
        try:
            f = open(nom_fichier, 'rb')
            self.modele = pickle.load(f)
            f.close()
        except:
            self.modele = modele_qui_predit()
            self.entrainer_modele()

    def enregistrer_modele(self, nom_fichier="decisiontree.pickle"):
        f = open(nom_fichier, 'wb+')
        pickle.dump(self.modele, f)
        f.close()

    def entrainer_modele(self):
        print("Entraînement de l'arbre de désicion")
        dirN = 1
        for dos in os.listdir(self.dossier_apprentissage):
            try:
                for fichier in os.listdir(self.dossier_apprentissage + "/" + dos):
                    if self.wav_file.match(fichier):
                        print(dos + "/" + fichier)
                        coefs_fft = wav_coefs_morceaux("{}/{}/{}".format(self.dossier_apprentissage, dos, fichier), N)
                        for coefs in np.abs(coefs_fft):
                            utilisation_coefs(self.Xlearn, self.Ylearn, coefs, label=dirN)
                self.labels.append(dos)
                self.labels_reverse[dirN] = dos
                self.labels_dict[dos] = dirN
                dirN += 1
            except NotADirectoryError:
                pass
        self.modele.fit(self.Xlearn, self.Ylearn)
        self.modele.fit(self.Xlearn, self.Ylearn)

    def predire_classe_probas(self, coefs_fft, dirN=None, verbose=False):
        # type: (np.array, Union[str, None], bool) -> Tuple[int, dict]
        Xtest, Ytest = [], []

        for coefs in coefs_fft: # c'est une matrice
            utilisation_coefs(Xtest, Ytest, coefs, dirN)  # Xtest est un vecteur ici
            if dirN is not None: Ytest.append(dirN)

        Ypred = self.modele.predict(Xtest)
        if verbose and dirN is not None:
            for i in range(1, len(Ypred)):
                print("prévu: {}".format(Ypred[i]))
                print("voulu: {}".format(Ytest[i]))
                self.gCYtest.append(Ytest[i])
                self.gCYpred.append(Ypred[i])
                if (Ytest[i] != Ypred[i]):
                    print(i)
                print('------------------')
            print("prédictions: {}".format(self.modele.predict(Xtest).tolist()))
            print(" attendues : {}".format(Ytest))
            plt.matshow(metrics.confusion_matrix(Ytest, Ypred))
            plt.show()
        comptage = np.bincount(Ypred)
        probas = {}
        nbre_ech = np.sum(comptage)
        for i in self.labels_dict.keys():
            try:
                probas[self.labels_dict[i]] = 100 * comptage[i] / nbre_ech  # pourcentage
            except IndexError:
                if i < len(comptage):  # il y a eu une vraie erreur
                    raise IndexError
                else:
                    pass  # le programme a juste pas fait beaucoup de choix différents
        classe_elue_n = comptage.argmax()
        classe_elue = self.labels_dict[classe_elue_n]
        if probas[classe_elue] > POURCENTAGE_AUTORISATION:
            return classe_elue, probas
        else:
            return self.labels_dict[0], probas

    def predire_classe_texte(self, coefs_fft, dirN=None, verbose=False):
        classe, probas = self.predire_classe_probas(coefs_fft, dirN, verbose)
        return classe

    def autoriser_personne_probas(self, coefs_fft):
        # type: (dict) -> Tuple[str, dict, bool]
        classe, probas = self.predire_classe_probas(coefs_fft)
        return classe, probas, (classe in self.classes_autorisees)


class DetecteurDeVoix(BaseDetecteur):

    def __init__(self, **kwargs):
        try:
            self.classes_autorisees = []  # on annule l'attribut hérité
            for personne in Personne.select():
                self.labels_dict[personne.id] = personne.nom
                self.labels_reverse[personne.nom] = personne.id
                if personne.autorisee:
                    self.classes_autorisees.append(personne.nom)
            print(self.labels_dict)
        except:
            print("aucune classe déterminée")
        super().__init__(**kwargs)

    def entrainer_modele(self):
        """
        Surcharge la méthode idoine pour charger les échantillons depuis la base de données plutôt que depuis des fichiers
        """
        print("Entraînement de l'arbre de décision depuis la base de données")
        for personne in Personne.select():
            print(personne.nom + str(personne.id))
            for echantillon in personne.echantillons:
                print(echantillon.nom_echantillon)
                for morceau in echantillon.morceaux: #  un vecteur
                    #for coefs in mfcc(morceau.coefs, freq_ech):
                    #    self.Xlearn.append(coefs)
                    #    self.Ylearn.append(personne.id)
                    #self.Xlearn.append(transformation_coefs(morceau.coefs))
                    #self.Ylearn.append(personne.id)
                    utilisation_coefs(self.Xlearn, self.Ylearn, morceau.coefs, label=personne.id)
            self.labels.append(personne.nom)
        to_learn = np.array(self.Xlearn)
        print(to_learn.shape)
        self.modele.fit(to_learn, self.Ylearn)

    def ajouter_echantillon_bdd(self, coefs_fft, personne, nom_echantillon):
        # try:
        #    personne = Personne.select().where(Personne.nom == nom_classe)
        # except bdd.PersonneDoesNotExist:
        #    personne = Personne.create(nom=nom_classe)
        # try:
        #    echantillon = Echantillon.get((Echantillon.nom_echantillon == nom_echantillon) & (Echantillon.personne == personne))
        # except bdd.EchantillonDoesNotExist:
        #    echantillon = Echantillon.create(nom_echantillon=nom_echantillon, personne=personne)

        echantillon, estilnouveau = Echantillon.get_or_create(nom_echantillon=nom_echantillon, personne=personne)
        for coefs in coefs_fft:
            m = Morceau(echantillon=echantillon)
            m.coefs = coefs  # pour que la conversion interne du tableau en string soit bien faite
            m.save()

    def remplir_bdd(self):
        print("Remplissage de la BDD avec des échantillons depuis le dossier " + self.dossier_apprentissage)
        Xlearn, Ylearn = [], []  # listes d'entrainement pour le machine learning
        for dos in os.listdir(self.dossier_apprentissage):
            personne, nouveau = Personne.get_or_create(nom=dos)
            self.labels_dict[personne.id] = personne.nom
            self.labels_reverse[personne.nom] = personne.id
            try:
                for fichier in os.listdir(self.dossier_apprentissage + "/" + dos):
                    if self.wav_file.match(fichier):
                        print(dos + "/" + fichier)
                        coefs_fft = wav_coefs_morceaux("{}/{}/{}".format(self.dossier_apprentissage, dos, fichier), N)
                        self.ajouter_echantillon_bdd(coefs_fft, personne=personne, nom_echantillon=fichier)
            except NotADirectoryError:
                pass


class TestP2I(BaseDetecteur):  # classe héritée pour les tests
    gCYtest, gCYpred = [], []  # matrice de confusion sur toutes les transformées de Fourier

    def tester_modele(self):
        coefs_fft = None
        gYtest, gYpred = [], []
        for dos in os.listdir(self.dossier_test):
            try:
                for fichier in os.listdir(self.dossier_test + "/" + dos):
                    if self.wav_file.match(fichier):  # pour tous les fichiers audio
                        dirN = self.labels_reverse[dos]
                        print(dos + "/" + fichier + " : " + str(dirN))
                        coefs_fft = wav_coefs_morceaux("{}/{}/{}".format(self.dossier_test, dos, fichier), N)
                        classe = self.predire_classe(coefs_fft, dirN, verbose=True)
                        gYpred.append(classe)
                        gYtest.append(dirN)
                self.labels_reverse[dirN] = dos
                self.labels_dict[dos] = dirN
                dirN += 1
            except NotADirectoryError:
                pass
        self.matrice_confusion = metrics.confusion_matrix(gYtest, gYpred)
        coefs_fft = None
        return gYtest, gYpred

    def afficher_matrice_confusion(self):
        if self.matrice_confusion is not None:
            plt.matshow(self.matrice_confusion)
            plt.show()
            print(self.matrice_confusion)
        else:
            print("la matrice de confusion n'est pas calculée")

    def test_confusion(self):
        self.tester_modele()
        self.afficher_matrice_confusion()

    def confusion_globale(self):
        mc = metrics.confusion_matrix(self.gCYtest, self.gCYpred)
        plt.matshow(mc)
        print(mc)
        plt.show()


def print_debug(s, *args, **kwargs):
    if VERBOSE:
        print(s) #print(s, *args, **kwargs)


class P2I(object):
    reconnaissance_active = True

    waterfall = [np.linspace(0, 100, 64)]
    waterfall_index = 0

    graph_change = False

    def __init__(self):
        self.setup_serial()
        self.lancer_reconnaissance_vocale()

    def plot(self, X, Y, *args, **kwargs):
        plt.plot(X, Y, *args, **kwargs)

    def afficher_nom(self, nom, autorise=None):
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

    def read_serial(self, analyse, repeter=True):
        if repeter:
            NOMBRE_FFT_REQUIS=NOMBRE_FFT_RECONNAISSANCE
        else:
            NOMBRE_FFT_REQUIS=NOMBRE_FFT_ENREGISTREMENT
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

    def afficher_probas(self, probas):
        print("  ".join(["{}: {}".format(k, round(v)) for k, v in probas.items()]))

    def voir_matrice_ffts(self, coefs_fft, nom):
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

    def lancer_enregistrement(self, callback):
        def analyse(donnees):  # en fait on l'utilise pas
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
        self.file_menu.add_command(label="Quitter", command=self.destroy)

        self.detection_menu.add_command(label="Détecter un loctuer avec Arduino",
                                        command=self.lancer_reconnaissance_vocale)
        self.detection_menu.add_command(label="Arrêter la détection", command=self.stop_reconnaissance_vocale)

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

    def afficher_nom(self, nom, autorise):
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
        self.reconnaissance_active=False #pour arrêter la reconnaisance avant l'enregistrement
        fenetre_rec = tkinter.Toplevel()
        fenetre_rec.title("Enregistrement d'un nouvel échantillon")
        fenetre_rec.pack_propagate()

        progessbar = Progressbar(master=fenetre_rec, mode='determinate')
        # progessbar.pack()
        nom_input = tkinter.Entry(master=fenetre_rec)
        nom_input.pack()
        personne = None

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

        def handle_fin_rec(coefs_ffts):
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
            e= Echantillon.get(Echantillon.id == var_id_echantillon.get())
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
            echantilon = Echantillon.get(Echantillon.id == var_id_echantillon.get())
            coefs_fft = []
            for morceau in echantilon.morceaux:
                coefs_fft.append(morceau.coefs)
            self.voir_matrice_ffts(np.array(coefs_fft), echantilon.personne.nom)

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

    def afficher_probas(self, probas):
        s = "  ".join(["{}: {}".format(k, round(v)) for k, v in probas.items()])
        self.affichage_probas.configure(text=s)

    def voir_matrice_ffts(self, coefs_fft, nom):
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

    def reset_graph_loop(self):
        """Pas nécessaire pour le waterfall"""
        self.reset_graph()
        self.after(3000, self.reset_graph_loop)

    def reset_ecoute(self):
        self.coefs_ffts=[]
        self.donnees=[]
        self.waterfall=[]
        self.waterfall_index=0

    def voir_matrice_mfcc(self, coefs_fft, nom):
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

class TestMFCC(P2I):
    def __init__(self):
        self.setup_serial()
        self.read_serial()

    def read_serial(self):
        morceau_fft = None
        self.coefs_ffts = []
        while self.reconnaissance_active:
            ligne = self.serial_port.readline().replace(b'\r\n', b'')
            print(ligne, end=" ; ")
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
                print_debug("\nlongeur: {}".format(len(morceau_fft)))
                if len(morceau_fft) == 62:
                    fft_array = np.array(morceau_fft)
                    self.coefs_ffts.append(fft_array)
                else:
                    print("erreur de taille" + str(len(morceau_fft)))
                    morceau_fft = None
                    continue
                if len(
                        self.coefs_ffts) > NOMBRE_FFT_RECONNAISSANCE:  # on attend d'avoir quelques échantillons pour éviter de valier un seul faux positif
                    self.donnees = np.array(self.coefs_ffts)
                    for y in mfcc(self.donnees, freq_ech):
                        plt.plot(np.linspace(1, 13, 13), y)
                        plt.show()
                    self.coefs_ffts = []  # on reset
                morceau_fft = None  # pour bien faire sortir les erreurs
