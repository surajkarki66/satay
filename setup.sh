#!/bin/bash
git submodule set-url fpgaconvnet-model https://github.com/AlexMontgomerie/fpgaconvnet-model.git
git submodule set-url fpgaconvnet-optimiser https://github.com/AlexMontgomerie/fpgaconvnet-optimiser.git
git submodule update --init --recursive
conda create -n satay python=3.10
conda activate satay
cd fpgaconvnet-model
python3 -m pip install .
cd ../fpgaconvnet-optimiser
python3 -m pip install .
pip install nvidia-pyindex
pip install onnx-graphsurgeon
