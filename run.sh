#!/bin/bash


sudo apt update
sudo apt install -y software-properties-common


sudo add-apt-repository ppa:deadsnakes/ppa -y

sudo apt update

sudo apt install -y python3.8 python3.8-venv python3.8-dev

python3.8 --version

python3.8 -m venv vorto_v_env


source ./vorto_v_env/bin/activate


python --version

echo "Virtual environment 'vrp_v_env' created and activated with Python 3.8."
