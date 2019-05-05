import json
from sklearn import tree

data_learn, data_test={},{}
with open('data_learn.json', 'r') as f:
    data_learn = json.load(f)
f.close()
with open('data_test.json', 'r') as f:
    data_test = json.load(f)
f.close()
#voix = [x['coefs'] for x in data['fichiers'] if x['type'] == "voix"]
#sinus = [x['coefs'] for x in data['fichiers'] if x['type'] == "sinus"]

X = [x['coefs'] for x in data_learn['fichiers']]
Y = [x['classe']  for x in data_learn['fichiers']]

modele = tree.DecisionTreeClassifier()
modele.fit(X, Y)

X_test = [x['coefs'] for x in data_test['fichiers']]
Y_test = [x['classe']  for x in data_test['fichiers']]
print("prédictions: {}".format(modele.predict(X_test).tolist()))
print(" attendues : {}".format(Y_test))