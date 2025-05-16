#!/bin/bash

#SBATCH --job-name=gnn_pems
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

python main_gnn.py --model "GCN" --dataset "pems03"
python main_gnn.py --model "GIN" --dataset "pems03"
python main_gnn.py --model "GAT" --dataset "pems03"
python main_gnn.py --model "GraphSAGE" --dataset "pems03"
python main_gnn.py --model "GCN" --dataset "pems07"
python main_gnn.py --model "GIN" --dataset "pems07"
python main_gnn.py --model "GAT" --dataset "pems07"
python main_gnn.py --model "GraphSAGE" --dataset "pems07"