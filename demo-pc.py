import os

from python_speech_features import mfcc
from scipy.io.wavfile import *
from scipy.fftpack import fft
import numpy as np
from matplotlib import pyplot as plt
import json
import re
import json

from python_speech_features import mfcc
from sklearn import tree, metrics

from scipy.signal import hanning
from scipy.signal.windows import hamming
modele = tree.DecisionTreeClassifier()
Xlearn,Ylearn = [],[] #listes d'entrainement pour le machine learning
wav_file = re.compile('^.+wav$')
labels=[]#liste des répertoires
for dos in os.listdir('echantillons-learn'):
    try:
        for fichier in os.listdir("echantillons-learn/"+dos):
            if wav_file.match(fichier):
                print(dos+"/"+fichier)
                fe, signal = read("echantillons-learn/{}/{}".format(dos, fichier))
                coefs_mel: np.array = mfcc(signal, fe, nfft=1103)
                for coefs in np.abs(coefs_mel):
                    Xlearn.append(coefs)
                    Ylearn.append(dos)
#               coefs = [np.mean(c) for c in np.abs(coefs_mel.transpose())]
                #donnees = wav_coefs(nom_fichier="bonjour p2i/{}/{}".format(dos, fichier), classe=dos)
               # x, y, j = [], [],0
               # for v in coefs:  # présentation jolie du graph fourier
               #     x.append(j)
               #     x.append(j)
               #     x.append(j)
               #     y.append(0)
               #     y.append(v)
               #     y.append(0)
               #     j += 1
               # plt.plot(x, y)
               # plt.show()
        labels.append(dos)
    except NotADirectoryError:
        pass

modele.fit(Xlearn, Ylearn)

Xtest, Ytest = [], []
for dos in os.listdir('echantillons-test'):
    try:
        for fichier in os.listdir("echantillons-test/"+dos):
            if wav_file.match(fichier):
                print(dos+"/"+fichier)
                fe, signal = read("echantillons-test/{}/{}".format(dos, fichier))
                coefs_mel: np.array = mfcc(signal, fe, nfft=1103)
                for coefs in np.abs(coefs_mel):
                    Xtest.append(coefs)
                    Ytest.append(dos)
#               coefs = [np.mean(c) for c in np.abs(coefs_mel.transpose())]
                #donnees = wav_coefs(nom_fichier="bonjour p2i/{}/{}".format(dos, fichier), classe=dos)
                #x, y, j = [], [],0
                #for v in coefs:  # présentation jolie du graph fourier
                #    x.append(j)
                #    x.append(j)
                #    x.append(j)
                #    y.append(0)
                #    y.append(v)
                #    y.append(0)
                #    j += 1
                #plt.plot(x, y)
                #plt.show()
    except NotADirectoryError:
        pass
Ypred = modele.predict(Xtest)
for i in range(1, len(Ypred)):
    print("prévu: {}".format(Ypred[i]))
    print("voulu: {}".format(Ytest[i]))
    if(Ytest[i]!=Ypred[i]):
        print(i)
    print('------------------')
print("prédictions: {}".format(modele.predict(Xtest).tolist()))
print(" attendues : {}".format(Ytest))
metrics.confusion_matrix(Ytest, Ypred, labels=labels)
