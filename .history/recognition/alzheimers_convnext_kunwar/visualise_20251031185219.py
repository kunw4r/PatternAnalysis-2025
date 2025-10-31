"""
Visualize predictions from the trained ConvNeXt model on random test samples.
Shows actual vs predicted labels with confidence scores.
"""

import os
import sys
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import get_model


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config = checkpoint.get('config', {})
    model_name = config.get('model_name', 'convnext_base')
    dropout = config.get('dropout', 0.5)
    
    # Detect architecture from checkpoint
    stem_weight = None
    for key in checkpoint['model_state_dict'].keys():
        if 'downsample_layers.0.0.weight' in key or 'stem.0.weight' in key:
            stem_weight = checkpoint['model_state_dict'][key]
            break
    
    if stem_weight is not None:
        out_channels = stem_weight.shape[0]
        if out_channels == 128 and 'base' not in model_name.lower():
            print(f"⚠ Config says {model_name}, but checkpoint has {out_channels} channels (Base)")
            model_name = 'convnext_base'
        elif out_channels == 96 and 'small' not in model_name.lower() and 'tiny' not in model_name.lower():
            print(f"⚠ Config says {model_name}, but checkpoint has {out_channels} channels (Small/Tiny)")
            model_name = 'convnext_small'
    
    print(f"✓ Model loaded: {model_name} (dropout={dropout})")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Acc: {checkpoint.get('best_val_acc', 0)*100:.2f}%")
    
    # Create model
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
    model.eval())
    
    return model


def get_transform():
    """Get image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def load_random_samples(data_dir, num_samples=60):
    """Load random samples from AD and NC classes."""
    test_dir = os.path.join(data_dir, 'test')
    ad_dir = os.path.join(test_dir, 'AD')
    nc_dir = os.path.join(test_dir, 'NC')
    
    # Get all image paths
    ad_images = [os.path.join(ad_dir, f) for f in os.listdir(ad_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    nc_images = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sample equally from both classes
    samples_per_class = num_samples // 2
    ad_samples = random.sample(ad_images, min(samples_per_class, len(ad_images)))
    nc_samples = random.sample(nc_images, min(samples_per_class, len(nc_images)))
    
    # Combine and shuffle
    all_samples = [(path, 1) for path in ad_samples] + [(path, 0) for path in nc_samples]
    random.shuffle(all_samples)
    
    print(f"\nLoaded {len(ad_samples)} AD samples and {len(nc_samples)} NC samples")
    
    return all_samples


def predict_samples(model, samples, device, transform):
    """Run predictions on samples."""
    predictions = []
    
    print("\nRunning predictions...")
    for img_path, true_label in tqdm(samples):
        # Load and preprocess image
        img = Image.open(img_path).convert('L')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()
        
        predictions.append({
            'path': img_path,
            'true_label': true_label,
            'pred_label': pred_label,
            'confidence': confidence,
            'probs': probs[0].cpu().numpy()
        })
    
    return predictions


def visualize_predictions(predictions, output_dir='predictions_vis', samples_per_page=20):
    """Visualize predictions in a grid layout."""
    os.makedirs(output_dir, exist_ok=True)
    
    label_names = {0: 'NC', 1: 'AD'}
    
    # Calculate metrics
    correct = sum(1 for p in predictions if p['true_label'] == p['pred_label'])
    total = len(predictions)
    accuracy = correct / total * 100
    
    # Separate correct and incorrect predictions
    correct_preds = [p for p in predictions if p['true_label'] == p['pred_label']]
    incorrect_preds = [p for p in predictions if p['true_label'] != p['pred_label']]
    
    print(f"\n{'='*80}")
    print(f"VISUALIZATION RESULTS")
    print(f"{'='*80}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct} ({accuracy:.2f}%)")
    print(f"Incorrect: {len(incorrect_preds)} ({(total-correct)/total*100:.2f}%)")
    print(f"{'='*80}\n")
    
    # Create visualizations
    num_pages = (len(predictions) + samples_per_page - 1) // samples_per_page
    
    for page in range(num_pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, len(predictions))
        page_preds = predictions[start_idx:end_idx]
        
        # Calculate grid size
        cols = 5
        rows = (len(page_preds) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
        fig.suptitle(f'Alzheimer\'s Disease Predictions (Page {page+1}/{num_pages})\nOverall Accuracy: {accuracy:.2f}%', 
                     fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, pred in enumerate(page_preds):
            ax = axes[idx]
            
            # Load and display image
            img = Image.open(pred['path']).convert('L')
            ax.imshow(img, cmap='gray')
            
            # Create title with prediction info
            true_label = label_names[pred['true_label']]
            pred_label = label_names[pred['pred_label']]
            confidence = pred['confidence'] * 100
            
            # Color code: green for correct, red for incorrect
            color = 'green' if pred['true_label'] == pred['pred_label'] else 'red'
            
            title = f"True: {true_label} | Pred: {pred_label}\nConfidence: {confidence:.1f}%"
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(len(page_preds), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'predictions_page_{page+1}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
    
    # Create a summary visualization of incorrect predictions
    if incorrect_preds:
        num_show = min(20, len(incorrect_preds))
        cols = 5
        rows = (num_show + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
        fig.suptitle(f'Incorrect Predictions (Showing {num_show}/{len(incorrect_preds)})', 
                     fontsize=16, fontweight='bold', color='red')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx in range(num_show):
            pred = incorrect_preds[idx]
            ax = axes[idx]
            
            img = Image.open(pred['path']).convert('L')
            ax.imshow(img, cmap='gray')
            
            true_label = label_names[pred['true_label']]
            pred_label = label_names[pred['pred_label']]
            confidence = pred['confidence'] * 100
            
            title = f"True: {true_label} | Pred: {pred_label}\nConf: {confidence:.1f}%"
            ax.set_title(title, fontsize=10, color='red', fontweight='bold')
            ax.axis('off')
        
        for idx in range(num_show, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'incorrect_predictions.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_path}")
    
    print(f"\n✓ All visualizations saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize predictions on random test samples'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='results/base_onecycle_lr1e4_drop0.5_best.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/ADNI/AD_NC',
        help='Path to ADNI dataset'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=60,
        help='Number of random samples to visualize'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='predictions_vis',
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.checkpoint):
        checkpoint_path = os.path.join(script_dir, args.checkpoint)
        if os.path.exists(checkpoint_path):
            args.checkpoint = checkpoint_path
    
    if not os.path.isabs(args.data_dir):
        data_path = os.path.join(script_dir, args.data_dir)
        if os.path.exists(data_path):
            args.data_dir = data_path
    
    print("\n" + "="*80)
    print("ALZHEIMER'S DISEASE PREDICTION VISUALIZATION")
    print("="*80 + "\n")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_checkpoint(args.checkpoint, device)
    
    # Load random samples
    samples = load_random_samples(args.data_dir, args.num_samples)
    
    # Get transform
    transform = get_transform()
    
    # Run predictions
    predictions = predict_samples(model, samples, device, transform)
    
    # Visualize results
    visualize_predictions(predictions, args.output_dir)


if __name__ == '__main__':
    main()
