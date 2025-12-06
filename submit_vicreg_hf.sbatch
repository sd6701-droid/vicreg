#!/bin/bash
#SBATCH --output=/gpfs/data/shenlab/aj4718/ijepa/logs/slurm-%j.out
#SBATCH --error=/gpfs/data/shenlab/aj4718/ijepa/logs/slurm-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=160G
#SBATCH --time=20-00:00:00
#SBATCH --partition=a100_long

# Load modules
module load python/3.10
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate vicreg_hf

# Set distributed variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WANDB_API_KEY='f06a145e49c4c141b217ea006d4d9a8a0c9f390b'

# Run training
# Note: Using srun for distributed launch on multiple nodes
srun python main_vicreg.py \
    --arch resnet50x2 \
    --batch-size 1024 \
    --hf-datasets tsbpp/fall2025_deeplearning sm12377/tr_imgs:split=train \
    --epochs 100 \
    --exp-dir ./exp_resnet50x2_hf \
    --base-lr 0.2 \
    --sim-coeff 25.0 \
    --std-coeff 25.0 \
    --cov-coeff 1.0 \
    --num-workers 10 \
    --enable-wandb \
    --wandb-project vicreg-hf \
    --wandb-name resnet50x2-bs1024-4nodes
