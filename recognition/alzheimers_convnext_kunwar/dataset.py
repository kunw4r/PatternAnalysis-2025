"""
ADNI Dataset Loader for Alzheimer's Disease Classification
Loads brain MRI slices and prepares them for ConvNeXt training
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class ADNIDataset(Dataset):
    """
    ADNI Alzheimer's Disease Dataset
    
    Binary classification:
    - Class 0: NC (Normal Control)
    - Class 1: AD (Alzheimer's Disease)
    """
    
    def __init__(self, samples, transform=None):
        """
        Args:
            samples: List of (image_path, label) tuples
            transform: Torchvision transforms to apply
        """
        self.samples = samples
        self.transform = transform
        self.classes = ['NC', 'AD']
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_samples_from_folder(data_dir, split='train'):
    """
    Load all image paths and labels from a split folder.
    
    Args:
        data_dir: Root directory (e.g., '/home/groups/comp3710/ADNI/AD_NC')
        split: 'train' or 'test'
    
    Returns:
        samples: List of (image_path, label) tuples
        class_counts: Dictionary with counts per class
    """
    split_dir = os.path.join(data_dir, split)
    
    if not os.path.exists(split_dir):
        raise RuntimeError(f"Dataset directory not found: {split_dir}")
    
    samples = []
    class_to_idx = {'AD': 1, 'NC': 0}
    classes = ['AD', 'NC']
    class_counts = {cls: 0 for cls in classes}
    
    for class_name in classes:
        class_dir = os.path.join(split_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        class_idx = class_to_idx[class_name]
        
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                img_path = os.path.join(class_dir, img_name)
                samples.append((img_path, class_idx))
                class_counts[class_name] += 1
    
    return samples, class_counts


def get_transforms(split='train', img_size=224):
    """
    Get data transforms for different splits.
    
    Args:
        split: 'train', 'val', or 'test'
        img_size: Target image size (default: 224)
    
    Returns:
        torchvision.transforms.Compose object
    """
    
    if split == 'train':
        # Training augmentations
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalise(mean=[0.5], std=[0.5]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        # Val/test: no augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalise(mean=[0.5], std=[0.5]),
        ])
    
    return transform


def get_dataloaders(data_dir, batch_size=32, num_workers=4, img_size=224, val_split=0.2):
    """
    Create train, val, and test dataloaders with proper splitting.
    
    This function:
    1. Loads the 'train' folder and splits it into train (80%) + val (20%)
    2. Loads the 'test' folder separately (for final evaluation only)
    
    Args:
        data_dir: Root directory (e.g., '/home/groups/comp3710/ADNI/AD_NC')
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        img_size: Target image size (default: 224)
        val_split: Proportion of training data to use for validation (default: 0.2)
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation (split from train)
        test_loader: DataLoader for testing (separate folder)
    """
    
    print("\n" + "="*80)
    print("CREATING DATASETS WITH TRAIN/VAL/TEST SPLIT")
    print("="*80)
    
    # Load training samples
    print(f"\nLoading samples from: {data_dir}")
    train_samples, train_counts = load_samples_from_folder(data_dir, 'train')
    
    print(f"\nOriginal train folder:")
    print(f"  Total samples: {len(train_samples)}")
    for cls in ['AD', 'NC']:
        print(f"  {cls}: {train_counts[cls]}")
    
    # Calculate split sizes
    total_train = len(train_samples)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size
    
    print(f"\nSplitting into train/val:")
    print(f"  Train: {train_size} ({(1-val_split)*100:.0f}%)")
    print(f"  Val:   {val_size} ({val_split*100:.0f}%)")
    
    # Split indices randomly but reproducibly
    indices = list(range(total_train))
    torch.manual_seed(42)  # Reproducible split
    train_indices = torch.randperm(total_train)[:train_size].tolist()
    val_indices = torch.randperm(total_train)[train_size:].tolist()
    
    # Create train and val samples
    train_samples_split = [train_samples[i] for i in train_indices]
    val_samples_split = [train_samples[i] for i in val_indices]
    
    # Get transforms
    train_transform = get_transforms('train', img_size)
    val_transform = get_transforms('val', img_size)
    test_transform = get_transforms('test', img_size)
    
    # Create datasets
    train_dataset = ADNIDataset(train_samples_split, transform=train_transform)
    val_dataset = ADNIDataset(val_samples_split, transform=val_transform)
    
    # Load test dataset (separate folder)
    test_samples, test_counts = load_samples_from_folder(data_dir, 'test')
    test_dataset = ADNIDataset(test_samples, transform=test_transform)
    
    print(f"\nTest folder:")
    print(f"  Total samples: {len(test_samples)}")
    for cls in ['AD', 'NC']:
        print(f"  {cls}: {test_counts[cls]}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoader Summary:")
    print(f"  Train batches: {len(train_loader)} (batch_size={batch_size})")
    print(f"  Val batches:   {len(val_loader)} (batch_size={batch_size})")
    print(f"  Test batches:  {len(test_loader)} (batch_size={batch_size})")
    print("="*80 + "\n")
    
    return train_loader, val_loader, test_loader


# Test the dataset loader
if __name__ == '__main__':
    print("Testing ADNI Dataset Loader...")
    print("="*80)
    
    # Test with actual data path
    data_dir = '/home/groups/comp3710/ADNI/AD_NC'
    
    if os.path.exists(data_dir):
        # Create dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=16,
            num_workers=2,
            img_size=224,
            val_split=0.2
        )
        
        # Get a batch from each loader
        print("Testing train loader:")
        images, labels = next(iter(train_loader))
        print(f"  Images shape: {images.shape}")
        print(f"  Labels: {labels.tolist()}")
        
        print("\nTesting val loader:")
        images, labels = next(iter(val_loader))
        print(f"  Images shape: {images.shape}")
        print(f"  Labels: {labels.tolist()}")
        
        print("\nTesting test loader:")
        images, labels = next(iter(test_loader))
        print(f"  Images shape: {images.shape}")
        print(f"  Labels: {labels.tolist()}")
        
        print("\n✓ All tests passed!")
    else:
        print(f"⚠ Warning: Data directory not found: {data_dir}")
        print("This is expected if not running on Rangpur cluster")
