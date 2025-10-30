#!/bin/bash
#PBS -N alzheimers_predict
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o predict_output.log

# Load required modules (adjust based on your HPC setup)
module load anaconda3
module load cuda/11.8

# Navigate to working directory
cd $PBS_O_WORKDIR

# Activate your conda environment (adjust name as needed)
# conda activate your_env_name

# Run prediction on full test set
echo "Starting prediction at $(date)"
echo "Working directory: $(pwd)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo ""

python predict.py \
    --checkpoint checkpoints/best_model_job319943.pth \
    --data_dir ./data/ADNI/AD_NC \
    --batch_size 64 \
    --num_workers 8 \
    --output_dir ./predictions

echo ""
echo "Prediction completed at $(date)"
echo "Results saved to: predictions/"
