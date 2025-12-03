#!/bin/bash
#SBATCH --job-name=dataset-create
#SBATCH --output=/home/sd6701/fall2025_deeplearning/logs/dataset_create-%j.out
#SBATCH --error=/home/sd6701/fall2025_deeplearning/logs/dataset_create-%j.err

#SBATCH --partition=n2c48m24
#SBATCH --account=csci_ga_2572-2025fa

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1G              # must be <= ~23G on n2c48m24
#SBATCH --time=08:00:00
#SBATCH --gres=none            # force CPU-only

set -euo pipefail

mkdir -p /home/sd6701/fall2025_deeplearning/logs

source /home/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate vicreg

python -u dataset_create.py
