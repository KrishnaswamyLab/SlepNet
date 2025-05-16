#!/bin/bash

#SBATCH --job-name=slep_distance_ablation
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

# python main.py --layer_type "batch_energy" --num_slepians=5 --dataset "pvdm"
# python main.py --layer_type "batch_energy" --num_slepians=15 --dataset "pvdm"
# python main.py --layer_type "batch_distance" --num_slepians=50 --dataset "pvdm"
python main.py --layer_type "batch_distance" --num_slepians=200 --dataset "pvdm"
# python main.py --layer_type "batch_energy" --num_slepians=5 --dataset "ra"
# python main.py --layer_type "batch_energy" --num_slepians=15 --dataset "ra"
# python main.py --layer_type "batch_distance" --num_slepians=50 --dataset "ra"
python main.py --layer_type "batch_distance" --num_slepians=200 --dataset "ra"
# python main.py --layer_type "batch_energy" --num_slepians=5 --dataset "abide"
# python main.py --layer_type "batch_energy" --num_slepians=15 --dataset "abide"
# python main.py --layer_type "batch_distance" --num_slepians=50 --dataset "abide"
# python main.py --layer_type "batch_distance" --num_slepians=200 --dataset "abide"
