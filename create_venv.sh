#!/bin/bash

PYTHON_VERSION="3.11"
python$PYTHON_VERSION -m venv training_env

source training_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt