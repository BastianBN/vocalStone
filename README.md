# P2I Wheatstone - Reconnaissance vocale

## Test côté ordinateur
Le script `audio_fft.py` parcourt le dossier *bonjour p2i* qui contient les échantillons disponible sur le Google Drive. Il remplit le fichier `data.json`,
qu'on partitionne ensuite en deux fichiers, ``data_learn.json``
et `data_test.json`.
La classe de l'échantillon est définie par le nom du dossier.


Le script `classification.py` utilise ``data_learn.json`` pour entraîner un arbre de décision,
qui est testé sur  ``data_test.py``.
 