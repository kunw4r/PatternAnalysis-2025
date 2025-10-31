# How "Keep Best Model" Works - Visual Guide

## The Process

```
┌─────────────────────────────────────────────────────────────────────┐
│  EXPERIMENT 1: 1_tiny_onecycle                                      │
├─────────────────────────────────────────────────────────────────────┤
│  [Train] → Val Acc: 76.32%                                          │
│  [Predict] → Test Acc: 78.45% ✓                                     │
│  [Save] → checkpoints/best_model_1_tiny_onecycle.pth (287 MB)       │
│  Status: Checkpoint KEPT temporarily                                │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│  EXPERIMENT 2: 2_small_onecycle                                     │
├─────────────────────────────────────────────────────────────────────┤
│  [Train] → Val Acc: 78.12%                                          │
│  [Predict] → Test Acc: 79.12% ✓                                     │
│  [Save] → checkpoints/best_model_2_small_onecycle.pth (305 MB)      │
│  Status: Checkpoint KEPT temporarily                                │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│  EXPERIMENT 3: 3_base_onecycle                                      │
├─────────────────────────────────────────────────────────────────────┤
│  [Train] → Val Acc: 79.45%                                          │
│  [Predict] → Test Acc: 80.21% ✓                                     │
│  [Save] → checkpoints/best_model_3_base_onecycle.pth (348 MB)       │
│  Status: Checkpoint KEPT temporarily                                │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│  EXPERIMENT 4: 4_base_pretrained  ⭐ BEST!                          │
├─────────────────────────────────────────────────────────────────────┤
│  [Train] → Val Acc: 80.45%                                          │
│  [Predict] → Test Acc: 82.13% ✓ ← HIGHEST!                         │
│  [Save] → checkpoints/best_model_4_base_pretrained.pth (348 MB)     │
│  Status: Checkpoint KEPT temporarily                                │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
    ... Experiments 5, 6, 7, 8 continue ...
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│  ALL 8 EXPERIMENTS COMPLETE                                         │
├─────────────────────────────────────────────────────────────────────┤
│  Test Accuracies:                                                   │
│    1_tiny_onecycle:          78.45%                                 │
│    2_small_onecycle:         79.12%                                 │
│    3_base_onecycle:          80.21%                                 │
│    4_base_pretrained:        82.13% ← BEST! ⭐                      │
│    5_base_focal_aggressive:  81.05%                                 │
│    6_base_cosine_highLR:     79.88%                                 │
│    7_small_mixup:            77.92%                                 │
│    8_base_crossentropy:      80.56%                                 │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│  CLEANUP PHASE: Keep best 1 model                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Comparing test accuracies...                                       │
│  Winner: 4_base_pretrained (82.13%)                                 │
│                                                                     │
│  ✅ KEEP:                                                           │
│    • best_model_4_base_pretrained.pth (348 MB)                      │
│                                                                     │
│  🗑️ DELETE:                                                         │
│    • best_model_1_tiny_onecycle.pth (287 MB freed!)                 │
│    • best_model_2_small_onecycle.pth (305 MB freed!)                │
│    • best_model_3_base_onecycle.pth (348 MB freed!)                 │
│    • best_model_5_base_focal_aggressive.pth (348 MB freed!)         │
│    • best_model_6_base_cosine_highLR.pth (348 MB freed!)            │
│    • best_model_7_small_mixup.pth (305 MB freed!)                   │
│    • best_model_8_base_crossentropy.pth (348 MB freed!)             │
│                                                                     │
│  Total freed: 2,289 MB (~2.2 GB)                                    │
│  Total kept:    348 MB                                              │
│  Savings:      87%                                                  │
└─────────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│  FINAL RESULT                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  checkpoints/                                                       │
│    └── best_model_4_base_pretrained.pth (348 MB) ⭐                 │
│                                                                     │
│  results/                                                           │
│    ├── test_results_1_tiny_onecycle.json (10 KB)                    │
│    ├── test_results_2_small_onecycle.json (10 KB)                   │
│    ├── test_results_3_base_onecycle.json (10 KB)                    │
│    ├── test_results_4_base_pretrained.json (10 KB) ⭐               │
│    ├── test_results_5_base_focal_aggressive.json (10 KB)            │
│    ├── test_results_6_base_cosine_highLR.json (10 KB)               │
│    ├── test_results_7_small_mixup.json (10 KB)                      │
│    ├── test_results_8_base_crossentropy.json (10 KB)                │
│    ├── all_experiments_results.json (20 KB)                         │
│    └── all_experiments_results.csv (5 KB)                           │
│                                                                     │
│  You know immediately:                                              │
│    ✓ 4_base_pretrained achieved 82.13% test accuracy               │
│    ✓ You have the checkpoint to use for predictions                │
│    ✓ You saved 2.2 GB of storage space                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Options for `--keep-best`

### `--keep-best 1` (Default - Maximum storage savings)
```
Keep: Only the #1 best model (82.13%)
Delete: All other 7 models
Storage: ~348 MB
```

### `--keep-best 2` (Keep top 2)
```
Keep: #1 (82.13%) and #2 (81.05%)
Delete: 6 other models
Storage: ~696 MB
```

### `--keep-best 3` (Keep top 3)
```
Keep: #1 (82.13%), #2 (81.05%), #3 (80.56%)
Delete: 5 other models
Storage: ~1044 MB (1 GB)
```

### `--keep-best 0` (Keep nothing - Results only!)
```
Keep: NO checkpoints
Delete: ALL 8 models
Storage: Only results JSONs (~100 KB total)
```
**Use this if you only care about finding which config works best, not the actual model!**

## Why This is Better Than Deleting Immediately

### ❌ If we deleted immediately:
```
Train Exp 1 → Predict → Delete ✓
Train Exp 2 → Predict → Delete ✓
Train Exp 3 → Predict → Delete ✓
...
Result: No models left, just test results
Can't use any model for actual predictions later!
```

### ✅ Our approach (keep best):
```
Train Exp 1 → Predict → Keep temporarily
Train Exp 2 → Predict → Keep temporarily
Train Exp 3 → Predict → Keep temporarily
...
All done → Compare → Keep BEST, delete rest
Result: You have the best model ready to use!
```

## Use Cases

| Scenario | --keep-best | Why |
|----------|-------------|-----|
| Find best config | 0 | Just need results, save max space |
| Production use | 1 | Need the best model to deploy |
| Compare top models | 2-3 | Want to ensemble or compare |
| Keep all (testing) | 8 | Debugging, want all checkpoints |

## Summary

**"Keep best model by test accuracy" means:**
1. Run ALL experiments first
2. Look at test accuracy of each
3. Keep checkpoint(s) with highest test accuracy
4. Delete all others
5. You end up with the best model(s) + all test results + huge storage savings!
