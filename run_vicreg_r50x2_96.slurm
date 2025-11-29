#!/bin/bash
#SBATCH --job-name=vicreg-r50x2-96
#SBATCH --output=/home/sd6701/fall2025_deeplearning/logs/vicreg_r50x2-%j.out
#SBATCH --error=/home/sd6701/fall2025_deeplearning/logs/vicreg_r50x2-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               # or gpu:a100:1 if your cluster uses that syntax
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --partition=c12m85-a100-1  # change if you use a different partition
#SBATCH --account=csci_ga_2572-2025fa

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Starting at: $(date)"
echo "Model: VICReg ResNet-50x2 on cc3m_96px"
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
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")')"
echo "========================================"

python main_vicreg.py \
  --data-dir /home/sd6701/datasets/fall2025_deeplearning/cc3m_96px_vicreg_final_exp \
  --exp-dir /home/sd6701/fall2025_deeplearning/outputs/vicreg_cc3m_resnet101/100k_dataset \
  --arch resnet101 \
  --epochs 15 \
  --batch-size 64 \
  --base-lr 0.15 \
  --log-freq-time 5 \
  --device cuda

echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
