import os
from scipy.io.wavfile import *
from scipy.fftpack import fft
import numpy as np
from matplotlib import pyplot as plt
import re
from python_speech_features import mfcc
from sklearn import tree, metrics
from scipy.signal.windows import hamming
N=128*2
def wav_coefs_morceaux(nom_fichier: str):
    fe, audio = read(nom_fichier)#on lit chaque fichier audio
    morceaux = np.array_split(audio, 100) #on coupe en 100 morceaux de taille a peu prs egale
    coefs = []
    for morceau in morceaux:
        window = hamming(len(morceau))
        coefs.append(np.abs(fft(morceau*window, N)[0:N//2])) #partie réelle positive
    return coefs

modele = tree.DecisionTreeClassifier()
Xlearn,Ylearn = [],[] #listes d'entrainement pour le machine learning
wav_file = re.compile('^.+wav$')
labels,dirN=[],0#liste des répertoires
for dos in os.listdir('echantillons-learn'):
    try:
        for fichier in os.listdir("echantillons-learn/"+dos):
            if wav_file.match(fichier):
                print(dos+"/"+fichier)
                coefs_fft = wav_coefs_morceaux("echantillons-learn/{}/{}".format(dos, fichier))
                for coefs in np.abs(coefs_fft):
                    Xlearn.append(coefs)
                    Ylearn.append(dirN)
                labels.append(dos)
        dirN += 1
    except NotADirectoryError:
        pass

modele.fit(Xlearn, Ylearn)

gYtest, gYpred = [], []
dirN=0
for dos in os.listdir('echantillons-test'):
    try:
        for fichier in os.listdir("echantillons-test/"+dos):
            if wav_file.match(fichier):
                print(dos+"/"+fichier+" : "+str(dirN))
                Xtest, Ytest = [], []
                coefs_fft = wav_coefs_morceaux("echantillons-test/{}/{}".format(dos, fichier))
                for coefs in np.abs(coefs_fft):
                    Xtest.append(coefs)
                    Ytest.append(dirN)

                Ypred = modele.predict(Xtest)
                for i in range(1, len(Ypred)):
                    print("prévu: {}".format(Ypred[i]))
                    print("voulu: {}".format(Ytest[i]))
                    if (Ytest[i] != Ypred[i]):
                        print(i)
                    print('------------------')
                print("prédictions: {}".format(modele.predict(Xtest).tolist()))
                print(" attendues : {}".format(Ytest))
                #plt.matshow(metrics.confusion_matrix(Ytest, Ypred, labels=labels))
                #plt.show()
                gYtest.append(dirN)
                gYpred.append(int(np.bincount(Ypred).argmax()))
        dirN+=1
    except NotADirectoryError:
        pass

labeldict, n = {}, 0
for l in labels:
    labeldict[l]=n

mc = metrics.confusion_matrix(gYtest, gYpred)
print(mc)
plt.matshow(mc)
plt.show()

