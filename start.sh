#!/bin/bash
set -e
if [ -e venv ] && [ -d venv ]
then
    echo "Pas besoin de créer un virtualenv"
    . venv/bin/activate
else
    echo "création du virtualenv"
    virtualenv -p /usr/bin/python3 venv
    . venv/bin/activate
    pip install -r requirements.txt
fi
if [[ -e decisiontree.pickle ]]
then
    echo "Modèle déjà présent, pas besoin de télécharger depuis la BDD"
else
    echo "Génération du modèle DecisionTree"
    python3 -c "from classificateur import DetecteurDeVoix;d=DetecteurDeVoix();d.enregistrer_modele()"
fi

python3 main.py