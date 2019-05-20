# Module de lecture/ecriture du port série
# Port série ttyACM0
# Vitesse de baud : 9600
# Timeout en lecture : 1 sec
# Timeout en écriture : 1 sec
import platform

from pyfiglet import Figlet
from serial import Serial

from classificateur import *

#f=open("decisiontree.pickle")
#modele = pickle.load(f)
#f.close()
classes_valides = ['bastian', 'jean'] #les numéros des dossiers avec le nom des personnes à reconnaître 0=bastian, 1=jean
if platform.system() == 'Linux': #Linux
    serial_port = Serial(port="/dev/ttyACM0", baudrate=115200, timeout=1, writeTimeout=1)
elif platform.system() == 'Darwin': #macOS
    serial_port = Serial(port='/dev/cu.usbmodem1A151', baudrate=115200, timeout=1, writeTimeout=1)
else: #Windows
    serial_port = Serial(port="COM3", baudrate=115200, timeout=1, writeTimeout=1)
t1=time.time()
ml = DetecteurDeVoix()
f = Figlet(font='slant')
with serial_port as port_serie:
    morceau_fft=None
    if port_serie.isOpen():
        coefs_ffts = [] #plusieurs transformées de Fourier
        while True:
            ligne = port_serie.readline().replace(b'\r\n', b'')
            if ligne == b'restart':
                print("Remise à zéro des tableaux, parlez maintenant")
                coefs_ffts=[]
                morceau_fft=[]
                continue
            if ligne == b"begin":
                morceau_fft = []  # une transformée de Fourier
                continue
            #if ligne != b'end' and ligne != b'begin' and ligne !=b'\n' and ligne != b'' and ligne != b'restart' and morceau_fft is not None:
                #print(ligne)
            try:
                nombre = float(ligne.decode('utf-8'))
                if ligne != 'end' and morceau_fft is not None:
                    morceau_fft.append(nombre)
            except (UnicodeDecodeError, ValueError):
                pass
            if ligne == b'end' and morceau_fft is not None:
                if len(morceau_fft) == 64:
                    coefs_ffts.append(np.array(morceau_fft))
                else:
                    morceau_fft=None
                    continue
                if len(coefs_ffts)>20: #on attend d'avoir quelques échantillons pour éviter de valier un seul faux positif
                    donnees = np.array(coefs_ffts)
                    classe_pred = ml.predire_classe_texte(donnees)
                    print(classe_pred)
                    if classe_pred in ml.classes_autorisees:
                        print("{} est autorisé(e) à entrer !".format(classe_pred))
                    coefs_ffts = [] #on reset
                morceau_fft = None  # pour bien faire sortir les erreurs
            #if time.time()-t1 > 20:
            #    print("temps écoulé, on reset l'écoute")
            #    t1=time.time()
            #    coefs_ffts=[] #on reset aussi toutes les minutes