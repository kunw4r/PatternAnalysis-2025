#!/bin/bash
# Monitor prediction progress

LOG_FILE="prediction_full.log"
CHECKPOINT_FILE="predictions/prediction_progress.pkl"

echo "=========================================="
echo "PREDICTION PROGRESS MONITOR"
echo "=========================================="
echo ""

# Check if process is running
if ps aux | grep -q "[p]redict.py"; then
    echo "✓ Prediction process is RUNNING"
    ps aux | grep "[p]redict.py" | head -1
else
    echo "⚠ Prediction process is NOT running"
fi

echo ""
echo "------------------------------------------"
echo "Latest log output:"
echo "------------------------------------------"
if [ -f "$LOG_FILE" ]; then
    tail -30 "$LOG_FILE"
else
    echo "Log file not found yet..."
fi

echo ""
echo "------------------------------------------"
echo "Checkpoint status:"
echo "------------------------------------------"
if [ -f "$CHECKPOINT_FILE" ]; then
    python3 - << 'PY'
import pickle, os
try:
    with open('predictions/prediction_progress.pkl', 'rb') as f:
        data = pickle.load(f)
        print(f"✓ Saved progress: {len(data['preds'])} samples processed")
        print(f"  Last batch: {data['batch_idx']}")
except Exception as e:
    print(f"Could not read checkpoint: {e}")
PY
else
    echo "No checkpoint file (either not started or already completed)"
fi

echo ""
echo "------------------------------------------"
echo "Results files:"
echo "------------------------------------------"
if [ -d "predictions" ]; then
    ls -lh predictions/ 2>/dev/null | grep -E "\.(png|txt)$" || echo "No result files yet"
else
    echo "Predictions directory not created yet"
fi

echo ""
echo "=========================================="
echo "To monitor in real-time, run:"
echo "  tail -f $LOG_FILE"
echo "=========================================="
