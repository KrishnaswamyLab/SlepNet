#!/bin/bash

#SBATCH --job-name=spectral_pems
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err
cd ~/project/SlepNet
module load miniconda
conda activate mfcn

python main_spec.py --dataset "pems03"
python main_spec.py --dataset "pems07"