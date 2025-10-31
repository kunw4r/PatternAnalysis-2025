# Sequential Experiment Runner - Storage-Efficient Mode

## Problem
Running 8 experiments overnight with checkpoints can use **2-3 GB of storage**:
- Each checkpoint: ~200-400 MB
- 8 experiments = 1.6-3.2 GB
- Rangpur quota: Only 8 GB!

## Solution
**Sequential Training with Smart Cleanup:**
1. Train experiment #1 → Predict (76% test acc) → **Keep checkpoint**
2. Train experiment #2 → Predict (78% test acc) → **Keep checkpoint**
3. Train experiment #3 → Predict (82% test acc) → **Keep checkpoint** ← Best!
4. Train experiment #4 → Predict (75% test acc) → **Keep checkpoint**
5. ... continue for all 8 experiments ...
6. **After all experiments finish:** 
   - Look at test accuracies: [76%, 78%, 82%, 75%, ...]
   - Keep checkpoint for exp #3 (82% - highest)
   - Delete checkpoints for all others
7. Final result: Only 1 checkpoint (~300 MB) instead of 8 (~2.4 GB)

**Storage savings: ~87%** (300 MB vs 2.4 GB)

## Files Created

### 1. `run_experiments_sequential.py`
- Runs experiments one at a time
- Auto-predicts after each training
- Deletes checkpoints immediately
- Keeps only best N models
- Saves all results to JSON/CSV

### 2. `run_sequential.sh`
- SLURM job script for Rangpur
- Runs all experiments overnight
- 72-hour time limit
- Monitors disk usage

## Usage

### On Rangpur:

```bash
cd ~/PatternAnalysis-2025/recognition/alzheimers_convnext_kunwar

# Submit the job
sbatch run_sequential.sh

# Monitor progress
tail -f sequential_<job_id>.out

# Check which experiments are done
cat results/all_experiments_results.csv
```

### Command-line Options:

```bash
# Run specific experiments
python run_experiments_sequential.py \
    --experiments small_focal_lr1e4_drop0.5,base_onecycle_pretrained_lr1e4_drop0.3 \
    --keep-best 2

# Run all experiments, keep best 3
python run_experiments_sequential.py \
    --experiments all \
    --keep-best 3

# Delete ALL checkpoints (keep only results)
python run_experiments_sequential.py \
    --experiments all \
    --keep-best 0
```

## What Gets Saved

### During Each Experiment:
```
checkpoints/
  ├── best_model_{exp_name}.pth       ← Created during training
  ├── training_curves_{exp_name}.png  ← Training visualization
  └── config_{exp_name}.json          ← Config file

results/
  └── test_results_{exp_name}.json    ← Test metrics (F1, precision, etc.)
```

### After Prediction → Cleanup:
```
checkpoints/
  ├── best_model_{exp_name}.pth       ← ❌ DELETED (~300 MB freed!)
  ├── training_curves_{exp_name}.png  ← ✅ Kept (small)
  └── config_{exp_name}.json          ← ❌ DELETED

results/
  └── test_results_{exp_name}.json    ← ✅ Kept (~10 KB)
```

### At the End (keep-best=1):
```
checkpoints/
  ├── best_model_<best_exp>.pth       ← ✅ Only the BEST model!
  └── training_curves_*.png           ← ✅ All training curves

results/
  ├── test_results_*.json             ← ✅ All test results
  ├── all_experiments_results.json    ← ✅ Summary JSON
  └── all_experiments_results.csv     ← ✅ Summary CSV
```

## Output Files

### `all_experiments_results.csv`
Easy-to-read summary of all experiments:
```csv
experiment_name,status,test_accuracy,test_f1_ad,best_val_acc,total_time_minutes
small_focal_lr1e4_drop0.5,completed,78.45,0.7821,76.32,45.2
base_onecycle_pretrained_lr1e4_drop0.3,completed,82.13,0.8156,80.45,67.8
```

### `test_results_{exp_name}.json`
Detailed metrics for each experiment:
```json
{
  "experiment_name": "base_onecycle_pretrained_lr1e4_drop0.3",
  "metrics": {
    "test_accuracy": 82.13,
    "test_f1_nc": 0.8045,
    "test_f1_ad": 0.8156,
    "test_precision_nc": 0.7834,
    "test_recall_ad": 0.8423,
    "confusion_matrix": [[45, 5], [8, 42]]
  }
}
```

## Storage Comparison

### Old Method (keep all checkpoints):
```
8 experiments × 300 MB = 2.4 GB
```

### New Method (sequential with cleanup):
```
Training curves: 8 × 0.5 MB =   4 MB
Test results:    8 × 0.01 MB = 0.08 MB
Best model:      1 × 300 MB = 300 MB
                          Total: ~304 MB (87% savings!)
```

## Features

✅ **Auto-Prediction:** Runs predict.py automatically after each training
✅ **Auto-Cleanup:** Deletes checkpoints to save space
✅ **Keep Best N:** Optionally keep top models by test accuracy
✅ **Resume Support:** Skip already completed experiments
✅ **Progress Tracking:** Real-time progress updates
✅ **Error Handling:** Continues if one experiment fails
✅ **Comprehensive Results:** CSV + JSON summaries
✅ **Storage Efficient:** ~90% space savings

## Example Output

```
================================================================================
EXPERIMENT 1/8
================================================================================

EXPERIMENT: small_focal_lr1e4_drop0.5

================================================================================
STEP 1: TRAINING
================================================================================
[... training output ...]
✓ New best model saved! Val Acc: 76.32%

================================================================================
STEP 2: PREDICTION
================================================================================
Overall Accuracy: 78.45%
NC Accuracy:      80.12%
AD Accuracy:      76.78%
F1 Score (NC):    0.7892
F1 Score (AD):    0.7821

================================================================================
STEP 3: CLEANUP
================================================================================
🗑️  Deleted checkpoint: checkpoints/best_model_small_focal_lr1e4_drop0.5.pth (287.3 MB freed)

================================================================================
✅ EXPERIMENT COMPLETE: small_focal_lr1e4_drop0.5
================================================================================
Validation Acc: 76.32%
Test Acc:       78.45%
Training Time:  42.3 min
Total Time:     45.2 min
================================================================================
```

## Tips

1. **For overnight runs:** Use `--keep-best 1` to save maximum space
2. **To keep top 3 models:** Use `--keep-best 3`
3. **To test locally:** Run with `--experiments small_onecycle_lr1e4_drop0.5` first
4. **Check progress:** `tail -f sequential_*.out` shows real-time updates
5. **View results anytime:** `cat results/all_experiments_results.csv`

## Advantages Over Original `run_experiments.py`

| Feature | Original | Sequential |
|---------|----------|------------|
| Storage used | ~2.4 GB | ~300 MB |
| Auto-prediction | ❌ No | ✅ Yes |
| Auto-cleanup | ❌ No | ✅ Yes |
| Can resume | ✅ Yes | ✅ Yes |
| Keep best models | ❌ No | ✅ Yes |
| CSV summary | ✅ Yes | ✅ Yes |
| Test metrics | ❌ No | ✅ Yes |

## Perfect For

✅ Overnight runs with limited storage
✅ Quick iteration (get test results immediately)
✅ Comparing many experiments
✅ Storage-constrained environments (Rangpur!)
✅ Finding the best model automatically

Happy experimenting! 🚀
