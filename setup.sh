#!/bin/bash -l

# Purge existing modules
module purge

# Load required modules
module load cuda/11.3.1
module load miniconda3/py39_4.10.3

# Create a new conda environment named covidtransformer
conda create --name covidtransformer python=3.9 -y

# Activate the environment
source activate covidtransformer

# Install required packages
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda install numpy scikit-learn scipy pandas -y
conda install -c conda-forge timm -y
conda install matplotlib tensorboard -y

echo "Environment covidtransformer setup completed."