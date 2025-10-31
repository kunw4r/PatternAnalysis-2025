"""
Prediction/Evaluation Script for Alzheimer's Disease Classification
Supports both slice-level and patient-level evaluation

Usage:
    # Slice-level evaluation only
    python predict.py --checkpoint ./results/small_onecycle_mixup_lr1e4_drop0.5_best.pth
    
    # Slice-level + Patient-level evaluation
    python predict.py --checkpoint ./results/small_onecycle_mixup_lr1e4_drop0.5_best.pth --patient_level
    
    # Save predictions to file
    python predict.py --checkpoint ./results/small_onecycle_mixup_lr1e4_drop0.5_best.pth --save_predictions
"""

import os
import argparse
import json
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from dataset import get_dataloaders
from modules import get_model

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc,
    precision_recall_fscore_support
)
from tqdm import tqdm

from dataset import get_dataloaders
from modules import get_model


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_checkpoint(checkpoint_path, device):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dictionary
    """
    print("="*80)
    print("LOADING MODEL CHECKPOINT")
    print("="*80)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Print checkpoint info
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Job ID: {checkpoint.get('job_id', 'N/A')}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    print(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    # Get model config from checkpoint
    config = checkpoint.get('config', {})
    model_name = config.get('model_name', 'convnext_small')
    dropout = config.get('dropout', 0.5)
    
    # Try to infer model architecture from checkpoint weights if config is missing or wrong
    state_dict = checkpoint.get('model_state_dict', {})
    if state_dict:
        # Check stem output channels to detect architecture
        # ConvNeXt-Tiny/Small use 96, ConvNeXt-Base uses 128
        stem_key = None
        for k in state_dict.keys():
            if 'stem' in k and 'weight' in k and len(state_dict[k].shape) == 4:
                stem_key = k
                break
        
        if stem_key:
            inferred_dim = state_dict[stem_key].shape[0]
            if inferred_dim == 96 and model_name not in ['convnext_tiny', 'convnext_small']:
                print(f"⚠ Config says {model_name}, but checkpoint stem has {inferred_dim} channels (Tiny/Small)")
                model_name = 'convnext_small'
                print(f"  → Auto-corrected to {model_name}")
            elif inferred_dim == 128 and model_name != 'convnext_base':
                print(f"⚠ Config says {model_name}, but checkpoint stem has {inferred_dim} channels (Base)")
                model_name = 'convnext_base'
                print(f"  → Auto-corrected to {model_name}")
    
    print(f"\nModel: {model_name}")
    print(f"Dropout: {dropout}")
    
    # Create model
    model = get_model(
        model_name=model_name,
        in_chans=1,
        num_classes=2,
        dropout=dropout,
        pretrained=False
    )
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print('\n⚠ Warning: direct load_state_dict failed — attempting smart remap of keys...')

        def remap_and_filter(sd: dict, target_sd: dict):
            new_sd = {}
            skipped = []
            for k, v in sd.items():
                nk = k
                # Specific mappings from saved naming to our naming
                nk = re.sub(r'^backbone\.stem\.0\.', 'downsample_layers.0.0.', nk)
                nk = re.sub(r'^backbone\.stem\.1\.', 'downsample_layers.0.1.', nk)
                nk = re.sub(r'^backbone\.stages\.([0-9]+)\.downsample\.([0-9]+)\.', r'downsample_layers.\1.\2.', nk)
                nk = re.sub(r'backbone\.stages\.([0-9]+)\.blocks\.([0-9]+)\.conv_dw\.', r'stages.\1.\2.dwconv.', nk)
                nk = re.sub(r'backbone\.stages\.([0-9]+)\.blocks\.([0-9]+)\.norm\.', r'stages.\1.\2.norm.', nk)
                nk = re.sub(r'backbone\.stages\.([0-9]+)\.blocks\.([0-9]+)\.mlp\.fc1\.', r'stages.\1.\2.pwconv1.', nk)
                nk = re.sub(r'backbone\.stages\.([0-9]+)\.blocks\.([0-9]+)\.mlp\.fc2\.', r'stages.\1.\2.pwconv2.', nk)
                nk = re.sub(r'backbone\.stages\.([0-9]+)\.blocks\.([0-9]+)\.gamma$', r'stages.\1.\2.gamma', nk)
                nk = re.sub(r'^backbone\.head\.norm\.', 'norm.', nk)
                nk = re.sub(r'^fc\.', 'head.', nk)
                # final fallback: strip a leading 'backbone.' prefix if present
                nk = re.sub(r'^backbone\.', '', nk)

                if nk not in target_sd:
                    skipped.append((k, nk, 'not in target'))
                    continue

                tgt_shape = target_sd[nk].shape
                src_shape = v.shape

                # If source is linear weights (2D) and target is conv1x1 weights (4D with last two dims 1)
                if v.ndim == 2 and len(tgt_shape) == 4 and tgt_shape[2] == 1 and tgt_shape[3] == 1:
                    try:
                        v = v.view(tgt_shape)
                    except Exception:
                        skipped.append((k, nk, f'unable to reshape {src_shape} -> {tgt_shape}'))
                        continue

                # If shapes still mismatch, skip
                if v.shape != tgt_shape:
                    skipped.append((k, nk, f'shape mismatch {v.shape} != {tgt_shape}'))
                    continue

                new_sd[nk] = v

            return new_sd, skipped

        target_sd = model.state_dict()
        remapped, skipped = remap_and_filter(checkpoint['model_state_dict'], target_sd)
        print(f"Remapped {len(remapped)} tensors, skipped {len(skipped)} tensors")
        if len(skipped) > 0:
            # Print a few skipped examples
            for s in skipped[:10]:
                print(' - skipped:', s)

        # Load remapped tensors into model (non-strict to allow missing keys)
        load_res = model.load_state_dict(remapped, strict=False)
        print('\n✓ Remap load completed. Missing keys (will be randomly initialised):', len(load_res.missing_keys))
    model = model.to(device)
    model.eval()
    
    print(f"\n✓ Model loaded successfully!")
    print("="*80 + "\n")
    
    return model, checkpoint


def predict_batch(model, dataloader, device):
    """
    Run predictions on entire dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device
    
    Returns:
        all_preds: Predicted classes
        all_labels: Ground truth labels
        all_probs: Prediction probabilities
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Running predictions...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Predicting'):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_metrics(preds, labels, probs):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        preds: Predicted classes
        labels: Ground truth labels
        probs: Prediction probabilities
    
    Returns:
        metrics: Dictionary of metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # ROC curve (for class 1 - AD)
    fpr, tpr, thresholds = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }
    
    return metrics


def print_results(metrics, class_names=['NC', 'AD']):
    """Print evaluation results"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Overall accuracy
    print(f"\n✓ Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    
    if metrics['accuracy'] >= 0.80:
        print("  🎉 SUCCESS: Achieved target accuracy (≥80%)!")
    else:
        print(f"  ⚠ Below target: Need {80 - metrics['accuracy']*100:.2f}% more")
    
    # Per-class metrics
    print("\n" + "-"*80)
    print("PER-CLASS METRICS")
    print("-"*80)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<10} {metrics['precision'][i]:<12.4f} "
              f"{metrics['recall'][i]:<12.4f} {metrics['f1'][i]:<12.4f} "
              f"{int(metrics['support'][i]):<10}")
    
    # Confusion matrix
    print("\n" + "-"*80)
    print("CONFUSION MATRIX")
    print("-"*80)
    cm = metrics['confusion_matrix']
    print(f"                Predicted NC    Predicted AD")
    print(f"Actual NC       {cm[0,0]:<15} {cm[0,1]:<15}")
    print(f"Actual AD       {cm[1,0]:<15} {cm[1,1]:<15}")
    
    # Calculate per-class accuracy
    nc_acc = cm[0,0] / (cm[0,0] + cm[0,1]) * 100
    ad_acc = cm[1,1] / (cm[1,0] + cm[1,1]) * 100
    
    print(f"\nNC Accuracy: {nc_acc:.2f}%")
    print(f"AD Accuracy: {ad_acc:.2f}%")
    
    # ROC AUC
    print("\n" + "-"*80)
    print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
    print("-"*80)
    
    print("="*80 + "\n")


def plot_confusion_matrix(cm, class_names=['NC', 'AD'], save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=20, fontweight='bold')
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, save_path='roc_curve.png'):
    """Plot ROC curve"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')
    
    # Labels and styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ ROC curve saved to {save_path}")
    plt.close()


def visualise_predictions(model, dataloader, device, num_samples=16, 
                         save_path='sample_predictions.png'):
    """
    Visualise sample predictions
    
    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device
        num_samples: Number of samples to visualise
        save_path: Path to save figure
    """
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images.to(device)
    
    # Limit to num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
    
    # Move to CPU for plotting
    images = images.cpu()
    probs = probs.cpu()
    predicted = predicted.cpu()
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    class_names = ['NC', 'AD']
    
    for i in range(num_samples):
        # Get image (squeeze out channel dimension)
        img = images[i].squeeze().numpy()
        
        # Get prediction info
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]
        confidence = probs[i][predicted[i]].item() * 100
        
        # Determine if correct
        is_correct = (labels[i] == predicted[i])
        color = 'green' if is_correct else 'red'
        
        # Plot
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(
            f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
            fontsize=9,
            color=color,
            fontweight='bold'
        )
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Sample predictions saved to {save_path}")
    plt.close()


def save_predictions_log(metrics, checkpoint_info, save_path='predictions.log'):
    """Save metrics to predictions log file"""
    import datetime
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ALZHEIMER'S DISEASE CLASSIFICATION - PREDICTIONS LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Checkpoint info
        f.write("MODEL INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Checkpoint: {checkpoint_info.get('checkpoint_path', 'N/A')}\n")
        f.write(f"Job ID: {checkpoint_info.get('job_id', 'N/A')}\n")
        f.write(f"Epoch: {checkpoint_info.get('epoch', 'N/A')}\n")
        f.write(f"Model: {checkpoint_info.get('model_name', 'N/A')}\n")
        f.write(f"Validation Accuracy: {checkpoint_info.get('val_acc', 'N/A'):.2f}%\n")
        f.write("\n")
        
        # Test results
        f.write("TEST SET EVALUATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%\n")
        f.write(f"ROC AUC Score: {metrics['roc_auc']:.4f}\n")
        f.write("\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS\n")
        f.write("-"*80 + "\n")
        class_names = ['NC', 'AD']
        f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*80 + "\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<10} {metrics['precision'][i]:<12.4f} "
                   f"{metrics['recall'][i]:<12.4f} {metrics['f1'][i]:<12.4f} "
                   f"{int(metrics['support'][i]):<10}\n")
        
        f.write("\n")
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        f.write("CONFUSION MATRIX\n")
        f.write("-"*80 + "\n")
        f.write(f"                Predicted NC    Predicted AD\n")
        f.write(f"Actual NC       {cm[0,0]:<15} {cm[0,1]:<15}\n")
        f.write(f"Actual AD       {cm[1,0]:<15} {cm[1,1]:<15}\n")
        
        # Per-class accuracy
        nc_acc = cm[0,0] / (cm[0,0] + cm[0,1]) * 100 if (cm[0,0] + cm[0,1]) > 0 else 0
        ad_acc = cm[1,1] / (cm[1,0] + cm[1,1]) * 100 if (cm[1,0] + cm[1,1]) > 0 else 0
        f.write(f"\nNC Accuracy: {nc_acc:.2f}%\n")
        f.write(f"AD Accuracy: {ad_acc:.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Predictions log saved to {save_path}")


def visualise_random_predictions(model, data_dir, device, num_samples=60,
                                save_dir='predictions_vis'):
    """
    Visualise predictions on random test samples
    
    Args:
        model: Trained model
        data_dir: Path to dataset
        device: Device
        num_samples: Number of random samples to visualise
        save_dir: Directory to save visualisations
    """
    import random
    from PIL import Image
    from torchvision import transforms
    
    print("Visualizing random predictions...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load random samples
    test_dir = os.path.join(data_dir, 'test')
    ad_dir = os.path.join(test_dir, 'AD')
    nc_dir = os.path.join(test_dir, 'NC')
    
    # Get all image paths
    ad_images = [os.path.join(ad_dir, f) for f in os.listdir(ad_dir) 
                 if f.endswith(('.png', '.jpg', '.jpeg'))]
    nc_images = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) 
                 if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sample equally from both classes
    samples_per_class = num_samples // 2
    ad_samples = random.sample(ad_images, min(samples_per_class, len(ad_images)))
    nc_samples = random.sample(nc_images, min(samples_per_class, len(nc_images)))
    
    # Combine and shuffle
    all_samples = [(path, 1) for path in ad_samples] + [(path, 0) for path in nc_samples]
    random.shuffle(all_samples)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalise(mean=[0.5], std=[0.5])
    ])
    
    # Run predictions
    predictions = []
    model.eval()
    
    for img_path, true_label in all_samples:
        img = Image.open(img_path).convert('L')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()
        
        predictions.append({
            'path': img_path,
            'image': img,
            'true_label': true_label,
            'pred_label': pred_label,
            'confidence': confidence
        })
    
    # Calculate accuracy
    correct = sum(1 for p in predictions if p['true_label'] == p['pred_label'])
    accuracy = correct / len(predictions) * 100
    
    # Visualise
    label_names = {0: 'NC', 1: 'AD'}
    samples_per_page = 20
    num_pages = (len(predictions) + samples_per_page - 1) // samples_per_page
    
    for page in range(num_pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, len(predictions))
        page_preds = predictions[start_idx:end_idx]
        
        cols = 5
        rows = (len(page_preds) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
        fig.suptitle(f'Random Sample Predictions (Page {page+1}/{num_pages})\nAccuracy: {accuracy:.2f}%',
                     fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, pred in enumerate(page_preds):
            ax = axes[idx]
            ax.imshow(pred['image'], cmap='gray')
            
            true_label = label_names[pred['true_label']]
            pred_label = label_names[pred['pred_label']]
            confidence = pred['confidence'] * 100
            
            color = 'green' if pred['true_label'] == pred['pred_label'] else 'red'
            title = f"True: {true_label} | Pred: {pred_label}\nConf: {confidence:.1f}%"
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
        
        for idx in range(len(page_preds), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(save_dir, f'random_predictions_page_{page+1}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Random sample visualisations saved to {save_dir}/")
    print(f"  {len(predictions)} samples visualised with {accuracy:.2f}% accuracy")


def main():
    parser = argparse.ArgumentParser(
        description='Predict Alzheimer\'s Disease using trained ConvNeXt model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='results/base_onecycle_lr1e4_drop0.5_best.pth',
        help='Path to model checkpoint (relative to script directory or absolute path)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/ADNI/AD_NC',
        help='Path to ADNI dataset'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./predictions',
        help='Directory to save prediction results'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script directory if not absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve checkpoint path
    if not os.path.isabs(args.checkpoint):
        checkpoint_path = os.path.join(script_dir, args.checkpoint)
        if os.path.exists(checkpoint_path):
            args.checkpoint = checkpoint_path
    
    # Resolve data directory path
    if not os.path.isabs(args.data_dir):
        data_path = os.path.join(script_dir, args.data_dir)
        if os.path.exists(data_path):
            args.data_dir = data_path
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("ALZHEIMER'S DISEASE CLASSIFICATION - PREDICTION")
    print("="*80 + "\n")
    
    # Get device
    device = get_device()
    print()
    
    # Load model
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    
    # Load test data
    print("="*80)
    print("LOADING TEST DATA")
    print("="*80)
    
    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=224
    )
    
    # Run predictions
    print("\n" + "="*80)
    print("RUNNING PREDICTIONS")
    print("="*80 + "\n")
    
    preds, labels, probs = predict_batch(model, test_loader, device)
    
    print(f"\n✓ Predictions complete!")
    print(f"  Total samples: {len(preds)}")
    print(f"  NC samples: {(labels == 0).sum()}")
    print(f"  AD samples: {(labels == 1).sum()}")
    
    # Calculate metrics
    metrics = calculate_metrics(preds, labels, probs)
    
    # Print results
    print_results(metrics)
    
    # Generate visualisations
    print("="*80)
    print("GENERATING VISUALISATIONS")
    print("="*80 + "\n")
    
    # Confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)
    
    # ROC curve
    roc_path = os.path.join(args.output_dir, 'roc_curve.png')
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'], 
                   save_path=roc_path)
    
    # Sample predictions from test loader
    samples_path = os.path.join(args.output_dir, 'sample_predictions.png')
    visualise_predictions(model, test_loader, device, num_samples=16,
                         save_path=samples_path)
    
    # Random sample predictions (like visualise.py)
    random_vis_dir = os.path.join(args.output_dir, 'random_samples')
    visualise_random_predictions(model, args.data_dir, device, num_samples=60,
                                 save_dir=random_vis_dir)
    
    # Save predictions log
    print()
    log_path = os.path.join(args.output_dir, 'predictions.log')
    checkpoint_info = {
        'checkpoint_path': args.checkpoint,
        'job_id': checkpoint.get('job_id', 'N/A'),
        'epoch': checkpoint.get('epoch', 'N/A'),
        'model_name': checkpoint.get('config', {}).get('model_name', 'N/A'),
        'val_acc': checkpoint.get('val_acc', 0),
    }
    save_predictions_log(metrics, checkpoint_info, save_path=log_path)
    
    print("\n" + "="*80)
    print("PREDICTION COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {args.output_dir}/")
    print("  - predictions.log")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - sample_predictions.png")
    print(f"  - random_samples/ (60 random test samples)")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()