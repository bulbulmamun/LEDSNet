#!/bin/bash 
#SBATCH --job-name=47_1DS5
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --output=LMSNet_Aug_01_47kp_GAN_16c_1k_1DS_05.out

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Now activate the conda environment
conda activate torch

# Check Python and pandas version
which python
python -c "import pandas as pd; print('Pandas version:', pd.__version__)"

# Run your Python script
python -u /home/uqabulbu/Codes_4_Study_4_Conf/1DS_47k/LMSNet_Aug_01_47kp_GAN_16c_1k_1DS_05.py