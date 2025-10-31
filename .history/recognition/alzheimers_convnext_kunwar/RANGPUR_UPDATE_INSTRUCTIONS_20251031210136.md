# Instructions for Updating Rangpur Files

## Changes Made to Prevent File Overwriting

When running multiple experiments in sequence with the same SLURM job ID, files were getting overwritten. Now each experiment gets unique filenames based on the experiment name.

## Files to Update on Rangpur

### 1. Update `run_experiments.py`

Find the section where `train(**config)` is called (around line 60-70) and add this ONE line:

```python
# BEFORE:
model, history = train(**config)

# AFTER (add the experiment_name to config):
config['experiment_name'] = exp_name  # ADD THIS LINE
model, history = train(**config)
```

That's the ONLY change needed in `run_experiments.py`!

### 2. Train.py Changes

The train.py changes have already been made to your local copy. You need to:

**Option A: Copy the entire updated train.py to Rangpur**
```bash
scp train.py <username>@rangpur.hpc.dc.uq.edu.au:~/PatternAnalysis-2025/recognition/alzheimers_convnext_kunwar/
```

**Option B: Or git pull the latest changes on Rangpur**
```bash
# On Rangpur:
cd ~/PatternAnalysis-2025/recognition/alzheimers_convnext_kunwar
git pull origin topic-recognition
```

### 3. Predict.py Changes

Same as train.py - either copy or git pull.

## What Files Get Saved Now

### Training (`train.py`):
- ✅ `best_model_{experiment_name}.pth` - Best model checkpoint
- ✅ `config_{experiment_name}.json` - Full training configuration
- ✅ `training_curves_{experiment_name}.png` - Train/val accuracy and loss plots

### Prediction (`predict.py`):
- ✅ `test_results_{experiment_name}.json` - F1, precision, recall metrics
- ✅ `confusion_matrix_slice_{experiment_name}.png` - Slice-level confusion matrix
- ✅ `confusion_matrix_patient_{experiment_name}.png` - Patient-level confusion matrix

## Example Experiment Names

For the experiments in your config:
- `small_focal_lr1e4_drop0.5`
- `base_onecycle_pretrained_lr1e4_drop0.3`
- `base_cosine_lr5e4_drop0.4`
- etc.

So you'll get files like:
- `best_model_small_focal_lr1e4_drop0.5.pth`
- `training_curves_base_onecycle_pretrained_lr1e4_drop0.3.png`
- `test_results_base_cosine_lr5e4_drop0.4.json`

**No more overwriting! Each experiment has unique files! 🎉**

## Quick Update Script for Rangpur

```bash
cd ~/PatternAnalysis-2025/recognition/alzheimers_convnext_kunwar

# Pull latest changes (if you pushed to git)
git pull origin topic-recognition

# OR manually edit run_experiments.py
# Add this line before train(**config):
# config['experiment_name'] = exp_name

# Verify changes
grep -n "experiment_name" run_experiments.py train.py predict.py
```

## Testing

After updates, test with one experiment:
```bash
python run_experiments.py --experiments small_onecycle_lr1e4_drop0.5
```

Check the checkpoints directory for unique filenames!
