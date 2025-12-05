#!/bin/bash
#SBATCH --job-name=vicreg-r101-96
#SBATCH --output=/home/sd6701/fall2025_deeplearning/logs/vicreg_r101-%j.out
#SBATCH --error=/home/sd6701/fall2025_deeplearning/logs/vicreg_r101-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               # or gpu:a100:1 if your cluster uses that syntax
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --partition=c12m85-a100-1  # change if you use a different partition
#SBATCH --account=csci_ga_2572-2025fa



echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"s
echo "Starting at: $(date)"
echo "Model: VICReg ResNet-101 on cc3m_96px"
echo "========================================"

cd /home/sd6701/test/vicreg

# Load CUDA if your cluster uses modules (optional – adjust or remove if not needed)
module load cuda/11.8 2>/dev/null || echo "cuda/11.8 module not loaded (maybe not needed)"

# Enable conda and activate vicreg env
source /home/sd6701/miniconda3/etc/profile.d/conda.sh
conda activate vicreg

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "========================================"


python main_vicreg.py \
  --data-dir /home/sd6701/datasets/dl_vicreg_final_exp_dataset_v2/all_images/train/ \
  --exp-dir /home/sd6701/fall2025_deeplearning/final-opts/vicreg_cc3m_resnet101_v2048/ \
  --arch resnet101 \
  --epochs 100 \
  --batch-size 2048 \
  --base-lr 0.25 \
  --log-freq-time 5 \
  --device cuda \
  --enable-wandb \
  --wandb-project dl_final_vicreg \
  --wandb-entity sd6701-new-york-university \
  --wandb-name r50_augs_glo_loc_2047
  --mlp 2048-2048-1024 \
  --use-local-loss \
  --local-mlp 1024-512 \
  --local-loss-weight 0.2



echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
