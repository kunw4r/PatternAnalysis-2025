"""
Training Script for Alzheimer's Disease Classification
Using custom-built ConvNeXt with advanced training techniques

Features:
- From-scratch or pretrained training
- Label Smoothing loss
- Optional MixUp augmentation
- OneCycleLR scheduler
- Early stopping
- Comprehensive logging
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from dataset import get_dataloaders
from modules import get_model, get_loss_function, MixUpAugmentation


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


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, 
                    epoch, use_mixup=False, mixup=None):
    """
    Train for one epoch
    
    Args:
        model: ConvNeXt model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        use_mixup: Whether to use MixUp augmentation
        mixup: MixUpAugmentation object
    
    Returns:
        epoch_loss: Average training loss
        epoch_acc: Training accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Apply MixUp if enabled
        if use_mixup and mixup is not None:
            images, labels_a, labels_b, lam = mixup.mixup_data(images, labels)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # MixUp loss
            loss = mixup.mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            # Standard forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # Step per batch for OneCycleLR
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{current_lr:.2e}'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """
    Validate the model
    
    Args:
        model: ConvNeXt model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
    
    Returns:
        epoch_loss: Average validation loss
        epoch_acc: Validation accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For computing per-class accuracy
    class_correct = [0, 0]  # [NC, AD]
    class_total = [0, 0]
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Valid]')
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class statistics
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    # Print per-class accuracy
    print(f"\n  Per-class accuracy:")
    print(f"    NC: {100.*class_correct[0]/class_total[0]:.2f}% ({class_correct[0]}/{class_total[0]})")
    print(f"    AD: {100.*class_correct[1]/class_total[1]:.2f}% ({class_correct[1]}/{class_total[1]})")
    
    return epoch_loss, epoch_acc


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path='training_curves.png'):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2, marker='s', markersize=4)
    ax2.axhline(y=80, color='g', linestyle='--', label='Target (80%)', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training curves saved to {save_path}")
    plt.close()


def train(
    # Data parameters
    data_dir='/home/groups/comp3710/ADNI/AD_NC',
    batch_size=32,
    num_workers=4,
    img_size=224,
    
    # Model parameters
    model_name='convnext_small',
    dropout=0.5,
    pretrained=False,
    pretrain_stages='early',  # 'all', 'early', or 'stem'
    
    # Training parameters
    num_epochs=30,
    learning_rate=1e-4,
    weight_decay=0.01,
    
    # Loss and augmentation
    loss_type='label_smoothing',  # 'label_smoothing' or 'cross_entropy'
    label_smoothing=0.1,
    use_mixup=False,
    mixup_alpha=0.4,
    
    # Other
    save_dir='./checkpoints',
    patience=10
):
    """
    Main training function
    
    Args:
        data_dir: Path to ADNI dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        img_size: Input image size
        model_name: 'convnext_tiny', 'convnext_small', or 'convnext_base'
        dropout: Dropout rate
        pretrained: Whether to use pretrained weights
        pretrain_stages: 'all', 'early', or 'stem'
        num_epochs: Maximum number of epochs
        learning_rate: Base learning rate
        weight_decay: Weight decay for AdamW
        loss_type: 'label_smoothing' or 'cross_entropy'
        label_smoothing: Smoothing parameter
        use_mixup: Whether to use MixUp augmentation
        mixup_alpha: MixUp alpha parameter
        save_dir: Directory to save checkpoints
        patience: Early stopping patience
    
    Returns:
        model: Trained model
        history: Training history
    """
    
    # Get SLURM JOB ID (if running on cluster)
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get device
    device = get_device()
    print(f"Job ID: {job_id}\n")
    
    # Load data
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    train_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size
    )
    
    # Create model
    print("="*80)
    print("CREATING MODEL")
    print("="*80)
    model = get_model(
        model_name=model_name,
        in_chans=1,  # Grayscale input
        num_classes=2,  # AD vs NC
        dropout=dropout,
        pretrained=pretrained,
        pretrain_stages=pretrain_stages if pretrained else 'all'
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Dropout: {dropout}")
    if pretrained:
        print(f"Pretrained: Yes (stages={pretrain_stages})")
    else:
        print(f"Pretrained: No (training from scratch)")
    
    # Loss function
    print("\n" + "="*80)
    print("TRAINING SETUP")
    print("="*80)
    criterion = get_loss_function(loss_type=loss_type, smoothing=label_smoothing)
    
    # MixUp
    mixup = None
    if use_mixup:
        mixup = MixUpAugmentation(alpha=mixup_alpha)
        print(f"Using MixUp augmentation (alpha={mixup_alpha})")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (OneCycleLR)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,  # Peak at 10x base LR
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.3,  # Warmup for first 30%
        anneal_strategy='cos',
        div_factor=10,  # Start at lr/10
        final_div_factor=1e4  # End very low
    )
    
    print(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"Scheduler: OneCycleLR (max_lr={learning_rate*10})")
    
    # Save configuration
    config = {
        'job_id': job_id,
        'model_name': model_name,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'dropout': dropout,
        'pretrained': pretrained,
        'pretrain_stages': pretrain_stages if pretrained else None,
        'loss_type': loss_type,
        'label_smoothing': label_smoothing if loss_type == 'label_smoothing' else None,
        'use_mixup': use_mixup,
        'mixup_alpha': mixup_alpha if use_mixup else None,
        'total_params': total_params,
        'trainable_params': trainable_params,
    }
    
    config_path = os.path.join(save_dir, f'config_job{job_id}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Config saved to {config_path}")
    
    # Training history
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print("\n" + "="*80)
    print(f"STARTING TRAINING ({num_epochs} epochs)")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, 
            epoch, use_mixup, mixup
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, device, epoch
        )
        
        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary")
        print(f"{'='*80}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.8f}")
        print(f"{'='*80}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            checkpoint_path = os.path.join(save_dir, f'best_model_job{job_id}.pth')
            torch.save({
                'job_id': job_id,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config,
            }, checkpoint_path)
            print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%\n")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience} patience)\n")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"{'='*80}")
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
            print(f"{'='*80}\n")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    if best_val_acc >= 80.0:
        print("\n🎉 SUCCESS: Reached target accuracy of 80%!")
    else:
        print(f"\n⚠ Best accuracy ({best_val_acc:.2f}%) is below target (80%)")
        print("Consider:")
        print("  - Training for more epochs")
        print("  - Using pretrained weights (pretrained=True)")
        print("  - Enabling MixUp (use_mixup=True)")
        print("  - Adjusting learning rate or dropout")
    
    # Plot training curves
    curves_path = os.path.join(save_dir, f'training_curves_job{job_id}.png')
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path=curves_path)
    
    # Update config with final results
    config['best_val_acc'] = best_val_acc
    config['best_epoch'] = best_epoch
    config['total_epochs'] = epoch + 1
    config['total_time_minutes'] = total_time / 60
    config['total_time_hours'] = total_time / 3600
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Best model: best_model_job{job_id}.pth")
    print(f"✓ Config: config_job{job_id}.json")
    print(f"✓ Curves: training_curves_job{job_id}.png")
    print("="*80 + "\n")
    
    return model, {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch
    }


if __name__ == '__main__':
    # Choose your training strategy:
    
    # OPTION 1: From Scratch
    model, history = train(
        data_dir='/home/groups/comp3710/ADNI/AD_NC',
        model_name='convnext_small',
        batch_size=32,
        num_epochs=40,
        learning_rate=1e-4,
        dropout=0.5,
        pretrained=False,  # ← Train from scratch
        loss_type='label_smoothing',
        label_smoothing=0.1,
        use_mixup=False,
        save_dir='./checkpoints'
    )
    
    # OPTION 2: Partial Pretrained (Conservative - if Option 1 doesn't reach 80%)
    # model, history = train(
    #     data_dir='/home/groups/comp3710/ADNI/AD_NC',
    #     model_name='convnext_small',
    #     batch_size=32,
    #     num_epochs=30,
    #     learning_rate=2e-4,
    #     dropout=0.5,
    #     pretrained=True,
    #     pretrain_stages='early',  # ← Only stem + stage 1-2
    #     loss_type='label_smoothing',
    #     label_smoothing=0.1,
    #     use_mixup=True,  # Add MixUp for more robustness
    #     mixup_alpha=0.4,
    #     save_dir='./checkpoints'
    # )
    
    # OPTION 3: Full Pretrained (Aggressive - last resort)
    # model, history = train(
    #     data_dir='/home/groups/comp3710/ADNI/AD_NC',
    #     model_name='convnext_small',
    #     batch_size=32,
    #     num_epochs=20,
    #     learning_rate=3e-4,
    #     dropout=0.5,
    #     pretrained=True,
    #     pretrain_stages='all',  # ← Full backbone
    #     loss_type='label_smoothing',
    #     label_smoothing=0.1,
    #     use_mixup=True,
    #     mixup_alpha=0.4,
    #     save_dir='./checkpoints'
    # )