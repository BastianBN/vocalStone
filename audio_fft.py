from scipy.io.wavfile import *
from scipy.fftpack import fft
import numpy as np
from matplotlib import pyplot as plt
import json
N=128*2
T=1.0/10000.0
data: dict = {"N":N, "fichiers":[]}
for i in range(1, 9): #boucle pour traiter 6 fichiers
    print(i)
    fe, audio = read('bonjour p2i/{}.wav'.format(i))#on lit chaque fichier audio
    coefs = np.abs(fft(audio, N)[0:N//2]) #partie réelle positive #[N//2:N]
    data['fichiers'].append(
        {
            "nom": "{}.wav".format(i),
            "coefs":coefs.tolist()
        }
    )
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
    n: int = 0
    #with open('coefs test jean.txt', 'a+') as f:
    #    f.write('audio: bonjour p2i jean {}'.format(i))
    #    f.write('\n[')
    #    for c in coefs:
    #        f.write(str(c))
    #        f.write(' , ')
    #        n+=1
    #        print(n)
    #    f.write(']\n----------------')
    #    f.write('\n')
    #f.close()
with open('data2.json', 'w+') as f:
    json.dump(data, f, sort_keys=True, indent=4)
f.close()