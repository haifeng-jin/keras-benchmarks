#!/bin/bash

venv_name=$1

python -m venv ~/.venv/$venv_name
source ~/.venv/$venv_name/bin/activate
pip install --upgrade pip
pip install -r requirements/$venv_name.txt
if [[ $venv_name == keras* ]]; then
    pip install keras==3.0.5
fi
pip install -e .
deactivate

echo "Installed libraries from $venv_name.txt"
