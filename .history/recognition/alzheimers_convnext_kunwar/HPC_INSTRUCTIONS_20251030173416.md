# Running Predictions on Rangpur HPC

## Files Updated
- ✅ **predict.py** - Simplified, removed checkpoint/timeout handling
- ✅ **run_predict_hpc.sh** - PBS submission script for HPC

## Changes Made to predict.py

### Removed:
- ❌ Progress checkpointing (--save_interval)
- ❌ Timeout error handling
- ❌ Sample limiting (--max_samples)
- ❌ Resume from checkpoint functionality

### Kept:
- ✅ Model loading with architecture auto-detection
- ✅ Full test set evaluation
- ✅ Metrics calculation (accuracy, F1, ROC-AUC)
- ✅ Visualization generation
- ✅ Results saving

### Default Settings:
- `--num_workers`: 4 (good for HPC)
- `--batch_size`: 32 (can increase to 64-128 on HPC)

## How to Run on Rangpur

### 1. Upload files to HPC
```bash
scp -r . uqusername@rangpur.hpc.uq.edu.au:/scratch/user/uqusername/alzheimers/
```

### 2. Submit job
```bash
ssh uqusername@rangpur.hpc.uq.edu.au
cd /scratch/user/uqusername/alzheimers/recognition/alzheimers_convnext_kunwar
qsub run_predict_hpc.sh
```

### 3. Monitor job
```bash
qstat -u uqusername
tail -f predict_output.log
```

### 4. Download results
```bash
scp -r uqusername@rangpur.hpc.uq.edu.au:/scratch/user/uqusername/alzheimers/recognition/alzheimers_convnext_kunwar/predictions ./
```

## Expected Performance

### Local (Mac):
- Speed: ~57 sec/batch
- Total: ~4.5 hours
- Issues: Timeouts, crashes

### HPC (Rangpur):
- Speed: ~2-5 sec/batch
- Total: **10-25 minutes** 🚀
- Issues: None

## Results Location

After completion, results will be in `predictions/`:
- `results.txt` - Detailed metrics
- `confusion_matrix.png` - Confusion matrix
- `roc_curve.png` - ROC curve
- `sample_predictions.png` - Sample predictions

## Notes

- The script uses **best_model_job319943.pth** (ConvNeXt-Base, 79.69% val acc)
- Adjust `--batch_size` based on GPU memory (64-128 for A100/V100)
- Adjust conda environment name in run_predict_hpc.sh if needed
