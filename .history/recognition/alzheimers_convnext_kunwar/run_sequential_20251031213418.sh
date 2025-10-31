#!/bin/bash
#SBATCH --job-name=alz_seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=72:00:00              # 72 hours for sequential experiments
#SBATCH --output=sequential_%j.out
#SBATCH --error=sequential_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s4697810@student.uq.edu.au

# Print job info
echo "=========================================="
echo "SEQUENTIAL EXPERIMENT RUNNER"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Load modules
module load miniconda3

# Activate conda environment
conda activate torch

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo ""

# Create directories
mkdir -p results
mkdir -p checkpoints

# Check available disk space
echo "=========================================="
echo "DISK SPACE CHECK"
echo "=========================================="
df -h .
echo ""

# Run sequential experiments
# Each experiment: Train → Predict → Delete checkpoint
# Only keeps best model at the end
echo "=========================================="
echo "STARTING SEQUENTIAL EXPERIMENTS"
echo "=========================================="
echo "Strategy: Train → Predict → Cleanup"
echo "Keeps only BEST model to save space"
echo ""

python run_experiments_sequential.py \
    --experiments all \
    --results-dir ./results \
    --keep-best 1 \
    --skip-completed

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ ALL EXPERIMENTS COMPLETED"
    echo "=========================================="
    echo "End Time: $(date)"
    echo ""
    
    # Show final disk usage
    echo "Final disk usage:"
    du -sh ./results
    du -sh ./checkpoints
    
    echo ""
    echo "Results saved to ./results/"
    echo "Best model saved to ./checkpoints/"
else
    echo ""
    echo "❌ Experiments failed"
    exit 1
fi
