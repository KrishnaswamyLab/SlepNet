#!/bin/bash

#SBATCH --job-name=slepnet_attn
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

python main.py --layer_type "attn_energy" --dataset "pvdm"
python main.py --layer_type "attn_distance" --dataset "pvdm"
python main.py --layer_type "attn_energy" --dataset "ra"
python main.py --layer_type "attn_distance" --dataset "ra"
python main.py --layer_type "attn_energy" --dataset "abide"
python main.py --layer_type "attn_distance" --dataset "abide"