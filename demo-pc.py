import itertools
import operator
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

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

modele = tree.DecisionTreeClassifier()
Xlearn,Ylearn = [],[] #listes d'entrainement pour le machine learning
wav_file = re.compile('^.+wav$')
labels,dirN=[],0#liste des répertoires
for dos in os.listdir('echantillons-learn'):
    try:
        for fichier in os.listdir("echantillons-learn/"+dos):
            if wav_file.match(fichier):
                print(dos+"/"+fichier)
                fe, signal = read("echantillons-learn/{}/{}".format(dos, fichier))
                coefs_mel: np.array = mfcc(signal, fe, nfft=1103)
                for coefs in np.abs(coefs_mel):
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
                fe, signal = read("echantillons-test/{}/{}".format(dos, fichier))
                coefs_mel: np.array = mfcc(signal, fe, nfft=1103)
                for coefs in np.abs(coefs_mel):
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
                gYpred.append(int(most_common(Ypred)))
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

