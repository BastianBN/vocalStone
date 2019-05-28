import platform
import numpy as np
import serial
from python_speech_features import mfcc
from serial.tools import list_ports
from typing import *

from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from bdd import *

freq_ech = 20000
modele = KNeighborsClassifier()
#modele = DecisionTreeClassifier()
NOMBRE_FFT_RECONNAISSANCE = 10


def setup_serial() -> Union[serial.Serial, None]:
    ports = list_ports.comports()
    for port in ports:
        if "Arduino" in port.description:
            print("Configuration de la carte {} branchée sur le port {}".format(port.description, port.device))
            serial_port = serial.Serial(port=port.device, baudrate=115200, timeout=1, writeTimeout=1)
            print(serial_port)
            return serial_port  # on sort de la boucle car on va pas configurer plusieurs ports série
    print("Configuration automatique du port série échouée, essai de configuration manuelle")
    try:
        if platform.system() == 'Linux':  # Linux
            serial_port = serial.Serial(port="/dev/ttyACM0", baudrate=115200, timeout=1, writeTimeout=1)
        elif platform.system() == 'Darwin':  # macOS
            serial_port = serial.Serial(port='/dev/cu.usbmodem1A161', baudrate=115200, timeout=1,
                                        writeTimeout=1)
        else:  # Windows
            serial_port = serial.Serial(port="COM4", baudrate=115200, timeout=1, writeTimeout=1)
    except serial.serialutil.SerialException:
        print("Port série non configuré")
        return None


labels = {}


def entrainement(modele):
    # type: (modele.__class__) -> Tuple[modele.__class__, dict]
    Xlearn, Ylearn = [], []
    for personne in Personne.select():
        print(personne.nom)
        for echantillon in personne.echantillons:
            print(echantillon.nom_echantillon)
            for morceau in echantillon.morceaux:
                coefs: np.array = morceau.coefs
                if coefs.max() > 500:
                    Xlearn.append(coefs if len(coefs)==62 else coefs[:-2])
                    Ylearn.append(personne.id)
        labels[personne.id] = personne.nom
    modele.fit(Xlearn, Ylearn)


def analyse(donnees: np.array):
    Ytest = []
    for coefs in donnees:
        if coefs.max() > 10:  # seuil pour éviter de reconnaitre du silence
            Ytest.append(coefs)
        else: print(coefs.max())
    if len(Ytest) <2:
        return
    Ypred=modele.predict(Ytest)
    comptage = np.bincount(Ypred)
    nbre_ech = np.sum(comptage)
    probas = {}
    for i in range(0, len(comptage)):
        if i in labels.keys():
            x= comptage[i] * 100 / nbre_ech
            probas[labels[i]] =x
    print("prédictions: "+"  ".join(["{}: {}".format(k, round(v)) for k, v in probas.items()]))


def read_serial(serial_port, analyse: Callable, repeter=True):
    morceau_fft = None
    coefs_ffts = []
    loop = True
    while loop:
        ligne = serial_port.readline().replace(b'\r\n', b'')
 #       print(ligne, end=" ; ")
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
                morceau_fft.append(nombre / 100)
        except (UnicodeDecodeError, ValueError):
            pass
        if ligne == b'end' and morceau_fft is not None:
  #          print("\nlongeur: {}".format(len(morceau_fft)))
            if len(morceau_fft) == 62:
                fft_array = np.array(morceau_fft)
                if fft_array.max() > 1:  # SEUIL_DETECTION:
                    coefs_ffts.append(fft_array)
  #                  print("nouveau morceau dans coefs_ffts")
                else:
                    print(fft_array.max())
            else:
                print("erreur de taille" + str(len(morceau_fft)))
                morceau_fft = None
                continue
            if len(
                    coefs_ffts) > NOMBRE_FFT_RECONNAISSANCE:  # on attend d'avoir quelques échantillons pour éviter de valier un seul faux positif
                donnees = np.array(coefs_ffts)
                analyse(donnees)
                coefs_ffts = []
            morceau_fft = None  # pour bien faire sortir les erreurs
        loop = repeter  # pour finir la boucle si pas besoin de repeter


entrainement(modele)
read_serial(setup_serial(), analyse)
