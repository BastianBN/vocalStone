from scipy.io.wavfile import *
from scipy.fftpack import fft
import numpy as np
from matplotlib import pyplot as plt
N=128
T=1.0/10000.0
for i in range(1, 6): #boucle pour traiter 6 fichiers
    print(i)
    fe, audio = read('bonjour p2i/{}.wav'.format(i))#on lit chaque fichier audio
    coefs = np.abs(fft(audio, N)[N//2:N]) #partie réelle positive
    x, y = [], []
    j = 0
    for v in coefs: #présentation jolie du graph fourier
        x.append(j)
        x.append(j)
        x.append(j)
        y.append(0)
        y.append(v)
        y.append(0)
        j += 1
    plt.plot(x, y)
    plt.ylabel('amplitude')
    plt.xlabel('fréquence')
    plt.show()
    with open('coefs test jean.txt', 'a+') as f:
        f.write('audio: bonjour p2i jean {}'.format(i))
        f.write('\n[')
        for c in coefs:
            f.write(str(c))
            f.write(' , ')
        f.write(']\n----------------')
        f.write('\n')
    f.close()
