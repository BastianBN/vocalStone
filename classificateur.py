import os
import pickle
import time
from typing import List, Tuple, Dict

from scipy.io.wavfile import *
from scipy.fftpack import fft
import numpy as np
from matplotlib import pyplot as plt
import re
from sklearn import metrics
from scipy.signal.windows import hamming
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from bdd import *
from python_speech_features import mfcc

freq_ech = 20000
N = 64 * 2  # nombre de coefficients par échantillon, il faut multiplier par 2 à cause de la moitié négative


# wav_file = re.compile('^.+wav$')
# labels = []  #liste des répertoires
#
#
def wav_coefs_morceaux(nom_fichier: str, N: int = N, T: float = 0.01) -> List[List]:
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


def transformation_coefs(coefs: list)->list:
    #return coefs
    return mfcc(coefs, 9000)[0]


#modele_qui_predit = KNeighborsClassifier
modele_qui_predit=DecisionTreeClassifier


class BaseDetecteur():
    """
    Classe qui inclut tout le nécessaire pour analyser des coefficients de Fourier en machine learning
    """
    N = None
    T = None
    wav_file = re.compile(
        '^.+wav$')  # regexp qui sert à déterminer quels fichiers seront utilisés pour l'apprentissage et le test
    labels = []
    matrice_confusion: np.ndarray
    t1: float
    modele: modele_qui_predit
    Xlearn, Ylearn = [], []  # listes d'entrainement pour le machine learning

    labels_dict = {0: 'silence'}  # {1:"random", 2: "ljklkj" ...}
    labels_reverse = {'silence': 0}  # {'random':1, 'lkjlkj':2 ... }

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
                            self.Xlearn.append(transformation_coefs(coefs_fft))
                            self.Ylearn.append(dirN)
                self.labels.append(dos)
                self.labels_reverse[dirN] = dos
                self.labels_dict[dos] = dirN
                dirN += 1
            except NotADirectoryError:
                pass
        self.modele.fit(self.Xlearn, self.Ylearn)

    def predire_classe_probas(self, coefs_fft, dirN=None, verbose=False):  # ->Tuple[int, dict]:
        Xtest, Ytest = [], []
        for coefs in np.abs(coefs_fft):
            Xtest.append(transformation_coefs(coefs))
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
        print(probas)
        classe_elue_n = comptage.argmax()
        classe_elue = self.labels_dict[classe_elue_n]
        if probas[classe_elue] > 60:
            return classe_elue, probas
        else:
            return self.labels_dict[0], probas

    def predire_classe_texte(self, coefs_fft, dirN=None, verbose=False) -> str:
        classe = self.predire_classe(coefs_fft, dirN, verbose)
        print(classe)
        return self.labels[classe]


class DetecteurDeVoix(BaseDetecteur):
    # bdd:Database = None
    # def __init__(self, fichier_modele=None, modele=None,
    #             dossier_apprentissage="echantillons-learn", dossier_test="echantillons-test", N=128, T=0.1,
    #             bdd=MySQLDatabase('G223_B_BD2', user='G223_B', password='G223_B', host='pc-tp-mysql.insa-lyon.fr', port=3306)
    #             ):
    #    self.bdd = bdd
    #    super(BaseDetecteur).__init__(fichier_modele, modele, dossier_apprentissage, dossier_test, N, T)

    def __init__(self, **kwargs):
        try:
            for personne in Personne.select():
                self.labels_dict[personne.id] = personne.nom
                self.labels_reverse[personne.nom] = personne.id
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
                for morceau in echantillon.morceaux:
                    self.Xlearn.append(transformation_coefs(morceau.coefs))
                    self.Ylearn.append(personne.id)
            self.labels.append(personne.nom)
        self.modele.fit(self.Xlearn, self.Ylearn)

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

    def predire_classe_texte(self, coefs_fft, dirN=None, verbose=False) -> str:
        classe = self.predire_classe(coefs_fft, dirN, verbose)
        # print(classe)
        return self.labels_dict[classe]


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

# p2i = TestP2I()
# p2i.test_confusion()
# print(p2i.labels_dict)
# p2i.confusion_globale()
# print(p2i.predire_classe_texte(wav_coefs_morceaux("bonjour p2i/jean/1.wav", 128*2)))
