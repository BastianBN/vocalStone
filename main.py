# Module de lecture/ecriture du port série
# Port série ttyACM0
# Vitesse de baud : 9600
# Timeout en lecture : 1 sec
# Timeout en écriture : 1 sec
import time

from serial import Serial
from demo_pc import *
import os
f=open("decisiontree.pickle")
modele = pickle.load(f)
f.close()
classes_valides = [0, 1] #les numéros des dossiers avec le nom des personnes à reconnaître 0=bastian, 1=jean
if os.name == 'posix':
    serial_port = Serial(port="/dev/ttyACM0", baudrate=115200, timeout=1, writeTimeout=1)
else:
    serial_port = Serial(port="COM3", baudrate=115200, timeout=1, writeTimeout=1)
t1=time.time()
with serial_port as port_serie:
    if port_serie.isOpen():
        coefs_ffts = [] #plusieurs transformées de Fourier
        while True:
            ligne = port_serie.readline().replace(b'\r\n', b'')
            morceau_fft = [] #une transformée de Fourier
            if ligne == b"begin":
                while ligne != b'end':
                    ligne = port_serie.readline().replace(b'\r\n', b'')
                    if ligne != b'end': morceau_fft.append(ligne)
                coefs_ffts.append(morceau_fft)
                if len(coefs_ffts)>10: #on attend d'avoir quelques échantillons pour éviter de valier un seul faux positif
                    classe_pred = predire_classe(modele, coefs_ffts)
                    if classe_pred in classes_valides:
                        print("Personne autorisée à entrer !")
                        coefs_ffts = [] #on reset
            if time.time()-t1 > 60:
                t1=time.time()
                coefs_ffts=[] #on reset aussi toutes les minutes