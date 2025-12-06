#!/bin/bash

# Create conda environment
conda create -n vicreg_hf python=3.10 -y

# Activate environment
source ~/.bashrc
conda activate vicreg_hf

# Install PyTorch (adjust cuda version as needed)
# Install PyTorch (using pip is often more robust for specific CUDA versions)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install datasets wandb scikit-learn pillow tqdm
