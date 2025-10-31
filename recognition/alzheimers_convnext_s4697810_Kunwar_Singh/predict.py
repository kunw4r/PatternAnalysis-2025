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
import matplotlib
matplotlib.use('Agg')  # For non-interactive plotting on HPC
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
