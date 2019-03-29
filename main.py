# Module de lecture/ecriture du port série
# Port série ttyACM0
# Vitesse de baud : 9600
# Timeout en lecture : 1 sec
# Timeout en écriture : 1 sec
from serial import Serial, unicode
from matplotlib import pyplot as plt

with Serial(port="COM3", baudrate=115200, timeout=1, writeTimeout=1) as port_serie:
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
                plt.plot(ffts)
                plt.ylabel('fft')
                plt.show()