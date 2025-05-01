#!/bin/bash

#SBATCH --job-name=denoising1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time 4-00:00:00
#SBATCH --partition batch
#SBATCH --exclude=pat-t1
#SBATCH --output slurm/%x_%j.out

eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate denoising

cd ~/workspace2/Document/Denoising_distillation_copy

python main.py --n 110 --seed 42 --dataset cifar10 --weight_dir \
	--ptq --wq 4 --aq 4
