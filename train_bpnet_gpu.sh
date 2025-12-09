#!/bin/bash
#SBATCH --job-name=bpnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bpnet_%j.out
#SBATCH --error=logs/bpnet_%j.err

# Load conda / Python environment (adjust module if needed)
module load miniconda3/23.11.0s
source ~/.bashrc
conda activate bpnet        # or source /course/cs1470/cs1470_env/bin/activate

echo "Job started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo "GPU info:"
nvidia-smi
echo ""

cd ~/BPNet

python -m bpnet.train_lstm \
  --train_mat data/Subset_Files/VitalDB_Train_Subset.mat \
  --val_mat data/Subset_Files/VitalDB_CalBased_Test_Subset.mat \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-3 \
  --log_dir runs/oscar \
  --checkpoint_dir checkpoints/oscar \
  --max_grad_norm 1.0

exit_code=$?
echo "Finished at: $(date)"
echo "Exit code: $exit_code"
exit $exit_code
