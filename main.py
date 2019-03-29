# Module de lecture/ecriture du port série
# Port série ttyACM0
# Vitesse de baud : 9600
# Timeout en lecture : 1 sec
# Timeout en écriture : 1 sec

from serial import Serial
from matplotlib import pyplot as plt
import os
if os.name == 'posix':
    serial_port = Serial(port="/dev/ttyACM0", baudrate=115200, timeout=1, writeTimeout=1)
else:
    serial_port = Serial(port="COM3", baudrate=115200, timeout=1, writeTimeout=1)
with serial_port as port_serie:
    if port_serie.isOpen():
        while True:
            ligne = port_serie.readline().replace(b'\r\n', b'')
            ffts = []
            if ligne == b"begin":
                while ligne != b'end':
                    ligne = port_serie.readline().replace(b'\r\n', b'')
                    if ligne != b'end': ffts.append(ligne)
                    print(ligne)
                print(b' '.join(ffts))
                x, y, i = [], [], 0 # oui Python c'est stylé
                for v in ffts: # on force la création de barres verticales
                    x.append(i)
                    x.append(i) # même abscisse
                    x.append(i)
                    y.append(0) # on fait les traits verticaux du graphique de Fourier
                    y.append(v) # on fait les traits verticaux du graphique de Fourier
                    y.append(0) # on fait les traits verticaux du graphique de Fourier
                    i += 1
                plt.plot(x, y)
                plt.ylabel('amplitude')
                plt.xlabel('fréquence')
                plt.show()