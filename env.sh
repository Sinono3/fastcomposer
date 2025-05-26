#!/usr/bin/env bash
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.0 -y
    conda activate $CONDA_ENV
    # This is optional if you prefer to use built-in nvcc
    conda install -c nvidia cuda-toolkit=11.8 -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# update pip to latest version for pyproject.toml setup.
pip install -U pip

# Required for CUDA 11.8
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# for fast attn
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118

# install fastcomposer
pip install -e .

# install torchprofile
# pip install git+https://github.com/zhijian-liu/torchprofile
