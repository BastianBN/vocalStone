# Module de lecture/ecriture du port série
# Port série ttyACM0
# Vitesse de baud : 9600
# Timeout en lecture : 1 sec
# Timeout en écriture : 1 sec
import json
import time

import numpy
from serial import Serial
import platform

# f=open("decisiontree.pickle")
# modele = pickle.load(f)
# f.close()
from bdd import Personne, Echantillon, Morceau

classes_valides = ['bastian',
                   'jean']  # les numéros des dossiers avec le nom des personnes à reconnaître 0=bastian, 1=jean
if platform.system() == 'Linux':  # Linux
    serial_port = Serial(port="/dev/ttyACM0", baudrate=115200, timeout=1, writeTimeout=1)
elif platform.system() == 'Darwin':  # macOS
    serial_port = Serial(port='/dev/cu.usbmodem1A151', baudrate=115200, timeout=1, writeTimeout=1)
else:  # Windows
    serial_port = Serial(port="COM3", baudrate=115200, timeout=1, writeTimeout=1)

with serial_port as port_serie:
    morceau_fft = None
    if port_serie.isOpen():
        coefs_ffts = []  # plusieurs transformées de Fourier
        print("Début enregistrement")
        while len(coefs_ffts) <= 40:
            ligne = port_serie.readline().replace(b'\r\n', b'')
            if ligne == b"begin":
                morceau_fft = []  # une transformée de Fourier
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
        # enregistrement
        print("Enregistrement terminé, écrivez votre nom > ", end='')
        nom = input()
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
