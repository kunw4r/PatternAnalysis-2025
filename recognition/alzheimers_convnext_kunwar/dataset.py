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
    
    Args:
        root_dir: Path to AD_NC folder
        split: 'train' or 'test'
        transform: Optional transform to apply to images
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Define class mapping
        self.classes = ['AD', 'NC']  # 0: Alzheimer's, 1: Normal Control
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load image paths and labels
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """Load all image paths and their labels"""
        split_dir = os.path.join(self.root_dir, self.split)
        
        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get all jpeg files in this class directory
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpeg') or img_name.endswith('.jpg'):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        print(f"Loaded {len(self.samples)} images for {self.split} split")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')  # Convert greyscale to RGB
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split='train', img_size=224):
    """
    Get data transforms for training or testing
    
    Args:
        split: 'train' or 'test'
        img_size: Target image size (default 224 for ConvNeXt)
    """
    
    if split == 'train':
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def get_dataloaders(data_dir, batch_size=32, num_workers=4, img_size=224):
    """
    Create train and test dataloaders
    
    Args:
        data_dir: Path to AD_NC directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        img_size: Image size for resizing
    
    Returns:
        train_loader, test_loader
    """
    
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
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


# Test the dataset
if __name__ == '__main__':
    # Test loading
    data_dir = './data/ADNI/AD_NC'
    
    print("Testing ADNI Dataset Loader...")
    print("-" * 50)
    
    # Create datasets
    train_loader, test_loader = get_dataloaders(data_dir, batch_size=16)
    
    # Check class distribution
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a sample batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:5]}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    print("\n✓ Dataset loader working correctly!")