"""
ADNI Dataset Loader for Alzheimer's Disease Classification
Loads brain MRI slices and prepares them for ConvNeXt training
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ADNIDataset(Dataset):
    """
    ADNI Alzheimer's Disease Dataset
    
    Loads grayscale MRI images for binary classification:
    - Class 0: NC (Normal Control)
    - Class 1: AD (Alzheimer's Disease)
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory containing train/ and test/ folders
            split: 'train' or 'test'
            transform: Torchvision transforms to apply
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Class names and mapping
        self.classes = ['AD', 'NC']
        self.class_to_idx = {'AD': 1, 'NC': 0}  # AD=1, NC=0
        
        # Load all samples
        self.samples = []
        self._load_samples()
        
        # Print dataset statistics
        self._print_stats()
        
    def _load_samples(self):
        """Load all image paths and labels"""
        split_dir = os.path.join(self.root_dir, self.split)
        
        if not os.path.exists(split_dir):
            raise RuntimeError(f"Dataset directory not found: {split_dir}")
        
        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Load all image files
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def _print_stats(self):
        """Print dataset statistics"""
        print(f"\n{'='*60}")
        print(f"ADNI Dataset - {self.split.upper()} Split")
        print(f"{'='*60}")
        
        # Count samples per class
        class_counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] += 1
        
        print(f"Total samples: {len(self.samples)}")
        for class_name in self.classes:
            count = class_counts[class_name]
            percentage = 100 * count / len(self.samples) if len(self.samples) > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            image: Tensor of shape [1, H, W] (grayscale)
            label: Integer class label (0=NC, 1=AD)
        """
        img_path, label = self.samples[idx]
        
        # Load image as grayscale (1 channel)
        # Our ConvNeXt model will handle conversion to 3 channels internally
        image = Image.open(img_path).convert('L')  # L = grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split='train', img_size=224):
    """
    Get data transforms for train or test split
    
    For grayscale images, we use single-channel normalization
    
    Args:
        split: 'train' or 'test'
        img_size: Target image size (default: 224)
    
    Returns:
        torchvision.transforms.Compose object
    """
    
    if split == 'train':
        # Training augmentations
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            
            # Geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            
            # Convert to tensor (grayscale: [1, H, W])
            transforms.ToTensor(),
            
            # Normalize grayscale image (mean and std for single channel)
            # Using approximate statistics for brain MRI
            transforms.Normalize(mean=[0.5], std=[0.5]),
            
            # Random erasing (cutout augmentation)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:
        # Test/validation: no augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    
    return transform


def get_dataloaders(data_dir, batch_size=32, num_workers=4, img_size=224):
    """
    Create train and test dataloaders
    
    Args:
        data_dir: Root directory (e.g., '/home/groups/comp3710/ADNI/AD_NC')
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        img_size: Target image size
    
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
    """
    
    print("Creating datasets...")
    
    # Create datasets
    train_dataset = ADNIDataset(
        root_dir=data_dir,
        split='train',
        transform=get_transforms('train', img_size)
    )
    
    test_dataset = ADNIDataset(
        root_dir=data_dir,
        split='test',
        transform=get_transforms('test', img_size)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batch for stable training
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Batch size: {batch_size}\n")
    
    return train_loader, test_loader


# Test the dataset
if __name__ == '__main__':
    print("Testing ADNI Dataset Loader...")
    print("="*80)
    
    # Test with actual data path
    data_dir = '/home/groups/comp3710/ADNI/AD_NC'
    
    if os.path.exists(data_dir):
        # Create dataloaders
        train_loader, test_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=16,
            num_workers=2,
            img_size=224
        )
        
        # Get a batch
        images, labels = next(iter(train_loader))
        
        print("Sample batch:")
        print(f"  Images shape: {images.shape}")  # Should be [16, 1, 224, 224]
        print(f"  Labels shape: {labels.shape}")  # Should be [16]
        print(f"  Image dtype: {images.dtype}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Label values: {labels.tolist()}")
        
        # Count classes in batch
        ad_count = (labels == 1).sum().item()
        nc_count = (labels == 0).sum().item()
        print(f"\n  Batch composition:")
        print(f"    AD: {ad_count}")
        print(f"    NC: {nc_count}")
        
        print("\n✓ Dataset test passed!")
    else:
        print(f"⚠ Warning: Data directory not found: {data_dir}")
        print("This is expected if not running on Rangpur cluster")
        print("\nTo test with your data, update the data_dir path")