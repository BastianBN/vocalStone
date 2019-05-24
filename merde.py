import platform
import numpy as np
import serial
from python_speech_features import mfcc
from serial.tools import list_ports
from typing import *

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from bdd import *

freq_ech = 20000
modele: ClassifierMixin = DecisionTreeClassifier


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


def entrainement(modele):
    # type: (ClassifierMixin) -> Tuple[ClassifierMixin, dict]
    labels, Xlearn, Ylearn = {}, [], []
    for personne in Personne.select():
        print(personne.nom)
        for echantillon in personne.echantillons:
            print(echantillon.nom_echantillon)
            for morceau in echantillon.morceaux:
                for coefs in mfcc(morceau.coefs, freq_ech):
                    Xlearn.append(coefs)
                    Ylearn.append(personne.id)
        labels.append(personne.nom)
    modele.fit(Xlearn, Ylearn)
    return modele, labels

ml=entrainement(modele)


def read_serial(self, serial_port,analyse: Callable, repeter=True):
    morceau_fft = None
    self.coefs_ffts = []
    loop = True
    while self.reconnaissance_active and loop:
        ligne = serial_port.readline().replace(b'\r\n', b'')
        print(ligne, end=" ; ")
        if ligne == b'restart':
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
            morceau_fft = None  # pour bien faire sortir les erreurs
        loop = repeter  # pour finir la boucle si pas besoin de repeter