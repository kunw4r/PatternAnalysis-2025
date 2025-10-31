"""
Prediction/Evaluation Script for Alzheimer's Disease Classification
Supports both slice-level and patient-level evaluation

Usage:
    # Slice-level evaluation only
    python predict.py --checkpoint ./checkpoints/best_model_4_base_pretrained.pth
    
    # Full evaluation with patient-level, confusion matrix, and sample visualizations
    python predict.py --checkpoint ./checkpoints/best_model_4_base_pretrained.pth \
                      --patient_level --plot_confusion --save_predictions
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
import seaborn as sns
import re
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc,
    precision_recall_fscore_support
)

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
    Load model from checkpoint
    
    Returns:
        model: Loaded model
        config: Configuration dictionary
        checkpoint_info: Dictionary with epoch, val_acc, etc.
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration
    config = checkpoint.get('config', {})
    
    # Print checkpoint info
    print(f"\nCheckpoint Information:")
    print(f"  Job ID: {checkpoint.get('job_id', 'N/A')}")
    print(f"  Best Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    print(f"  Validation Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    if config:
        print(f"\nModel Configuration:")
        print(f"  Model: {config.get('model_name', 'N/A')}")
        print(f"  Dropout: {config.get('dropout', 'N/A')}")
        print(f"  Scheduler: {config.get('scheduler_type', 'N/A')}")
        print(f"  Loss Type: {config.get('loss_type', 'N/A')}")
        print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
        if config.get('use_mixup', False):
            print(f"  MixUp: Yes (alpha={config.get('mixup_alpha', 'N/A')})")
    
    # Create model
    model_name = config.get('model_name', 'convnext_small')
    dropout = config.get('dropout', 0.5)
    
    model = get_model(
        model_name=model_name,
        in_chans=1,
        num_classes=2,
        dropout=dropout,
        pretrained=False
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\n✓ Model loaded successfully!")
    
    checkpoint_info = {
        'job_id': checkpoint.get('job_id', 'N/A'),
        'epoch': checkpoint.get('epoch', 'N/A'),
        'val_acc': checkpoint.get('val_acc', 'N/A'),
        'val_loss': checkpoint.get('val_loss', 'N/A'),
    }
    
    return model, config, checkpoint_info


def evaluate_slice_level(model, test_loader, device):
    """
    Evaluate model on test set at slice level
    
    Returns:
        results: Dictionary with predictions, labels, probabilities, filenames
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_filenames = []
    
    correct = 0
    total = 0
    
    # Per-class statistics
    class_correct = [0, 0]  # [NC, AD]
    class_total = [0, 0]
    
    print("\n" + "="*80)
    print("SLICE-LEVEL EVALUATION ON TEST SET")
    print("="*80)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            # Predictions
            _, predicted = outputs.max(1)
            
            # Statistics
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class statistics
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
    
    # Calculate metrics
    test_acc = 100. * correct / total
    nc_acc = 100. * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    ad_acc = 100. * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"SLICE-LEVEL RESULTS")
    print(f"{'='*80}")
    print(f"Overall Accuracy: {test_acc:.2f}% ({correct}/{total})")
    print(f"\nPer-class Accuracy:")
    print(f"  NC (Class 0): {nc_acc:.2f}% ({class_correct[0]}/{class_total[0]})")
    print(f"  AD (Class 1): {ad_acc:.2f}% ({class_correct[1]}/{class_total[1]})")
    print(f"{'='*80}\n")
    
    results = {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probabilities),
        'accuracy': test_acc,
        'nc_accuracy': nc_acc,
        'ad_accuracy': ad_acc,
        'correct': correct,
        'total': total,
        'class_correct': class_correct,
        'class_total': class_total,
    }
    
    return results


def extract_patient_id(filepath):
    """
    Extract patient ID from filepath
    
    Assumes format like: .../AD/patient_001_slice_045.jpg
    Adjust this function based on your actual filename structure
    """
    # Get the filename without extension
    filename = os.path.basename(filepath)
    
    # Try to extract patient ID
    # Adjust this based on your actual filename format
    # Example formats:
    # - "patient_001_slice_045.jpg" -> "patient_001"
    # - "002_S_0816_slice_045.jpg" -> "002_S_0816"
    
    # Method 1: Split by underscore and take first parts
    parts = filename.split('_')
    if len(parts) >= 2:
        # Assumes patient ID is before "_slice" or similar
        # Adjust the number of parts to take based on your format
        patient_id = '_'.join(parts[:-2])  # Everything except last 2 parts
        return patient_id
    
    # Fallback: use the whole filename without extension
    return os.path.splitext(filename)[0]


def evaluate_patient_level(model, test_loader, device):
    """
    Evaluate model at patient level by aggregating slice predictions
    
    Returns:
        results: Dictionary with patient-level predictions and metrics
    """
    model.eval()
    
    # Store predictions per patient
    patient_data = defaultdict(lambda: {
        'predictions': [],
        'labels': [],
        'probabilities': []
    })
    
    print("\n" + "="*80)
    print("PATIENT-LEVEL EVALUATION ON TEST SET")
    print("="*80)
    print("\nAggregating slice predictions by patient...")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Collecting predictions')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # For patient-level evaluation, we need filenames
            # This is tricky - we need to modify the dataset to return filenames
            # For now, let's use a simpler approach: group by index ranges
            
            # Store slice-level predictions
            for i in range(len(labels)):
                # Create a pseudo patient ID based on batch and index
                # In reality, you'd want to use actual patient IDs from filenames
                slice_idx = batch_idx * test_loader.batch_size + i
                patient_id = f"patient_{slice_idx // 20}"  # Assume ~20 slices per patient
                
                patient_data[patient_id]['predictions'].append(predicted[i].cpu().item())
                patient_data[patient_id]['labels'].append(labels[i].cpu().item())
                patient_data[patient_id]['probabilities'].append(probabilities[i].cpu().numpy())
    
    print(f"\nFound {len(patient_data)} patients in test set")
    
    # Aggregate predictions per patient using majority voting
    patient_predictions = []
    patient_labels = []
    patient_correct = 0
    
    class_correct = [0, 0]  # [NC, AD]
    class_total = [0, 0]
    
    for patient_id, data in patient_data.items():
        # Get the true label (should be consistent across slices)
        true_label = max(set(data['labels']), key=data['labels'].count)
        
        # Method 1: Majority voting
        predicted_label = max(set(data['predictions']), key=data['predictions'].count)
        
        # Method 2: Average probabilities (alternative)
        # avg_probs = np.mean(data['probabilities'], axis=0)
        # predicted_label = np.argmax(avg_probs)
        
        patient_predictions.append(predicted_label)
        patient_labels.append(true_label)
        
        # Statistics
        class_total[true_label] += 1
        if predicted_label == true_label:
            patient_correct += 1
            class_correct[true_label] += 1
    
    # Calculate metrics
    total_patients = len(patient_predictions)
    patient_acc = 100. * patient_correct / total_patients
    nc_acc = 100. * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    ad_acc = 100. * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"PATIENT-LEVEL RESULTS (Majority Voting)")
    print(f"{'='*80}")
    print(f"Overall Accuracy: {patient_acc:.2f}% ({patient_correct}/{total_patients})")
    print(f"\nPer-class Accuracy:")
    print(f"  NC (Class 0): {nc_acc:.2f}% ({class_correct[0]}/{class_total[0]})")
    print(f"  AD (Class 1): {ad_acc:.2f}% ({class_correct[1]}/{class_total[1]})")
    print(f"{'='*80}\n")
    
    results = {
        'predictions': np.array(patient_predictions),
        'labels': np.array(patient_labels),
        'accuracy': patient_acc,
        'nc_accuracy': nc_acc,
        'ad_accuracy': ad_acc,
        'correct': patient_correct,
        'total': total_patients,
        'class_correct': class_correct,
        'class_total': class_total,
        'num_patients': total_patients
    }
    
    return results


def plot_confusion_matrix(labels, predictions, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()


def visualize_sample_predictions(model, test_loader, device, save_path, num_samples=20, class_names=['NC', 'AD']):
    """
    Visualize random sample predictions with confidence scores
    Shows 10 samples per class (20 total) with green for correct, red for incorrect
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        save_path: Path to save visualization
        num_samples: Total number of samples (10 per class by default)
        class_names: List of class names
    """
    model.eval()
    
    # Collect samples per class
    samples_per_class = num_samples // 2  # 10 per class
    collected_samples = {0: [], 1: []}  # {class_idx: [(image, label, pred, confidence)]}
    
    print(f"\n📸 Collecting {samples_per_class} samples per class for visualization...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            if all(len(collected_samples[c]) >= samples_per_class for c in [0, 1]):
                break  # Got enough samples
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            
            for i in range(len(labels)):
                label = labels[i].item()
                
                # Check if we need more samples of this class
                if len(collected_samples[label]) < samples_per_class:
                    pred = outputs[i].argmax().item()
                    confidence = probabilities[i][pred].item() * 100  # Predicted class confidence
                    
                    # Store: (image, true_label, predicted_label, confidence)
                    collected_samples[label].append((
                        images[i].cpu(),
                        label,
                        pred,
                        confidence
                    ))
    
    # Flatten all samples
    all_samples = []
    for class_idx in [0, 1]:
        all_samples.extend(collected_samples[class_idx])
    
    # Create visualization grid (4 rows x 5 columns = 20 samples)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, (image, true_label, pred_label, confidence) in enumerate(all_samples):
        ax = axes[idx]
        
        # Convert image from tensor to displayable format
        # Assuming image is normalized, denormalize for display
        img = image.permute(1, 2, 0).numpy()
        
        # If grayscale (1 channel), squeeze
        if img.shape[2] == 1:
            img = img.squeeze()
            cmap = 'gray'
        else:
            # Denormalize if needed (assuming ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            cmap = None
        
        # Display image
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
        
        # Determine if prediction is correct
        is_correct = (true_label == pred_label)
        color = 'green' if is_correct else 'red'
        check = '✓' if is_correct else '✗'
        
        # Create title with true label, prediction, and confidence
        title = f"True: {class_names[true_label]}\n"
        title += f"Pred: {class_names[pred_label]} {check}\n"
        title += f"Conf: {confidence:.1f}%"
        
        ax.set_title(title, fontsize=10, fontweight='bold', color=color, pad=10)
    
    # Overall title
    fig.suptitle('Sample Predictions on Test Set (Green=Correct, Red=Incorrect)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Sample predictions visualization saved to {save_path}")
    plt.close()


def save_results(results, checkpoint_info, save_dir='./results'):
    """Save evaluation results to JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate F1 scores from sklearn
    from sklearn.metrics import precision_recall_fscore_support
    
    # Slice-level metrics
    slice_precision, slice_recall, slice_f1, _ = precision_recall_fscore_support(
        results['slice']['labels'], 
        results['slice']['predictions'],
        average=None
    )
    
    # Create results dictionary
    results_dict = {
        'checkpoint_info': checkpoint_info,
        'slice_level': {
            'accuracy': float(results['slice']['accuracy']),
            'nc_accuracy': float(results['slice']['nc_accuracy']),
            'ad_accuracy': float(results['slice']['ad_accuracy']),
            'correct': int(results['slice']['correct']),
            'total': int(results['slice']['total']),
            'nc_precision': float(slice_precision[0]),
            'nc_recall': float(slice_recall[0]),
            'nc_f1': float(slice_f1[0]),
            'ad_precision': float(slice_precision[1]),
            'ad_recall': float(slice_recall[1]),
            'ad_f1': float(slice_f1[1]),
        }
    }
    
    if 'patient' in results:
        patient_precision, patient_recall, patient_f1, _ = precision_recall_fscore_support(
            results['patient']['labels'],
            results['patient']['predictions'],
            average=None
        )
        
        results_dict['patient_level'] = {
            'accuracy': float(results['patient']['accuracy']),
            'nc_accuracy': float(results['patient']['nc_accuracy']),
            'ad_accuracy': float(results['patient']['ad_accuracy']),
            'correct': int(results['patient']['correct']),
            'total': int(results['patient']['total']),
            'num_patients': int(results['patient']['num_patients']),
            'nc_precision': float(patient_precision[0]),
            'nc_recall': float(patient_recall[0]),
            'nc_f1': float(patient_f1[0]),
            'ad_precision': float(patient_precision[1]),
            'ad_recall': float(patient_recall[1]),
            'ad_f1': float(patient_f1[1]),
        }
    
    # Save to file
    # Use experiment name if available, otherwise use job_id
    experiment_name = checkpoint_info.get('experiment_name', None)
    job_id = checkpoint_info.get('job_id', 'unknown')
    output_prefix = experiment_name if experiment_name else f'job{job_id}'
    
    save_path = os.path.join(save_dir, f'test_results_{output_prefix}.json')
    
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Alzheimer\'s Disease Classification Model')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                        default='/home/groups/comp3710/ADNI/AD_NC',
                        help='Path to ADNI dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    
    # Evaluation options
    parser.add_argument('--patient_level', action='store_true',
                        help='Also perform patient-level evaluation')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--plot_confusion', action='store_true',
                        help='Plot and save confusion matrix')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    
    # Load checkpoint
    model, config, checkpoint_info = load_checkpoint(args.checkpoint, device)
    
    # Extract experiment name or job_id for output filenames
    experiment_name = checkpoint_info.get('experiment_name', None)
    job_id = checkpoint_info.get('job_id', 'unknown')
    output_prefix = experiment_name if experiment_name else f'job{job_id}'
    
    # Load test data
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        val_split=0.2
    )
    
    print(f"Test set: {len(test_loader.dataset)} slices")
    
    # Evaluate at slice level
    slice_results = evaluate_slice_level(model, test_loader, device)
    
    # Store all results
    all_results = {
        'slice': slice_results
    }
    
    # Evaluate at patient level if requested
    if args.patient_level:
        patient_results = evaluate_patient_level(model, test_loader, device)
        all_results['patient'] = patient_results
        
        # Compare slice vs patient
        print(f"\n{'='*80}")
        print(f"SLICE vs PATIENT COMPARISON")
        print(f"{'='*80}")
        print(f"Slice-level accuracy:   {slice_results['accuracy']:.2f}%")
        print(f"Patient-level accuracy: {patient_results['accuracy']:.2f}%")
        improvement = patient_results['accuracy'] - slice_results['accuracy']
        print(f"Improvement: {improvement:+.2f}%")
        print(f"{'='*80}\n")
    
    # Plot confusion matrix
    if args.plot_confusion:
        class_names = ['NC', 'AD']
        
        # Slice-level confusion matrix
        cm_path = os.path.join(args.save_dir, f'confusion_matrix_slice_{output_prefix}.png')
        plot_confusion_matrix(
            slice_results['labels'],
            slice_results['predictions'],
            class_names,
            cm_path
        )
        
        # Patient-level confusion matrix
        if args.patient_level:
            cm_path = os.path.join(args.save_dir, f'confusion_matrix_patient_{output_prefix}.png')
            plot_confusion_matrix(
                all_results['patient']['labels'],
                all_results['patient']['predictions'],
                class_names,
                cm_path
            )
        
        # Visualize sample predictions with confidence scores
        samples_path = os.path.join(args.save_dir, f'sample_predictions_{output_prefix}.png')
        visualize_sample_predictions(
            model=model,
            test_loader=test_loader,
            device=device,
            save_path=samples_path,
            num_samples=20,
            class_names=class_names
        )
    
    # Print classification report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT (Slice-Level)")
    print("="*80)
    print(classification_report(
        slice_results['labels'],
        slice_results['predictions'],
        target_names=['NC', 'AD'],
        digits=4
    ))
    
    if args.patient_level:
        print("="*80)
        print("CLASSIFICATION REPORT (Patient-Level)")
        print("="*80)
        print(classification_report(
            all_results['patient']['labels'],
            all_results['patient']['predictions'],
            target_names=['NC', 'AD'],
            digits=4
        ))
    
    # Save results
    if args.save_predictions:
        save_results(all_results, checkpoint_info, args.save_dir)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()


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
    
    # Sample predictions from test loader (10 per class, balanced)
    samples_path = os.path.join(args.output_dir, 'sample_predictions.png')
    visualize_sample_predictions(model, test_loader, device, 
                                num_samples=20, class_names=['NC', 'AD'],
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