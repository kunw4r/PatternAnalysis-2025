"""
ADNI Dataset Loader for Alzheimer's Disease Classification (patient-level split)
Loads brain MRI slices and prepares them for ConvNeXt training.

Key points:
- Binary classes: NC=0, AD=1 (kept from your work)
- Patient-level (subject-level) split to avoid leakage
- Stratified by label to keep class balance in train/val
- Deterministic, reproducible splits (random_state=42)
"""

import os
import re
from typing import List, Tuple, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import GroupShuffleSplit

class ADNIDataset(Dataset):
    """
    ADNI Alzheimer's Disease Dataset

    Binary classification:
    - Class 0: NC (Normal Control)
    - Class 1: AD (Alzheimer's Disease)
    """

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        """
        Args:
            samples: List of (image_path, label) tuples
            transform: Torchvision transforms to apply
        """
        self.samples = samples
        self.transform = transform
        self.classes = ['NC', 'AD']  # order for reporting

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Robust open: rare I/O hiccups on clusters can throw; retry once
        try:
            image = Image.open(img_path).convert('L')  # grayscale
        except Exception:
            # simple retry with next index
            idx = (idx + 1) % len(self.samples)
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label
    
def _sorted_image_list(dir_path: str) -> List[str]:
    if not os.path.exists(dir_path):
        return []
    return [
        f for f in sorted(os.listdir(dir_path))
        if f.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp', '.tif', '.tiff'))
    ]


def load_samples_from_folder(data_dir: str, split: str = 'train') -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    """
    Load all image paths and labels from a split folder.

    Folder structure expected:
        data_dir/
          train/
            AD/*.png
            NC/*.png
          test/
            AD/*.png
            NC/*.png

    Returns:
        samples: list of (path, label) — with AD=1, NC=0
        counts: dict with per-class counts
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        raise RuntimeError(f"Dataset directory not found: {split_dir}")

    class_to_idx = {'AD': 1, 'NC': 0}
    classes = ['AD', 'NC']  # enumerate AD first to match your prior mapping when printing too
    counts = {cls: 0 for cls in classes}
    samples: List[Tuple[str, int]] = []

    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Warning: Class directory not found: {cls_dir}")
            continue
        for name in _sorted_image_list(cls_dir):
            p = os.path.join(cls_dir, name)
            samples.append((p, class_to_idx[cls]))
            counts[cls] += 1

    return samples, counts


def get_transforms(split: str = 'train', img_size: int = 224):
    """
    Get data transforms for different splits.

    Note: we must call the torchvision API `transforms.Normalize` (US spelling);
    comments/logs use Australian spelling (“normalise”).
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # intensity normalisation
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])
    else:  # 'val' or 'test'
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


def extract_subject_id(path: str) -> str:
    """
    Extract a subject/patient identifier from a filename.

    We try a few common ADNI-like patterns, then fall back to the token before first underscore,
    else the basename without extension.

    Adapt this to your exact naming if needed.
    """
    b = os.path.basename(path)

    # Common patterns to try (add/modify as needed)
    patterns = [
        r'(sub-[A-Za-z0-9]+)',       # e.g., sub-12345
        r'(Patient[0-9]+)',          # e.g., Patient123
        r'(Subject[0-9]+)',          # e.g., Subject001
        r'([A-Za-z]+[0-9]{3,})',     # e.g., ADNI12345 or AD1234
    ]
    for pat in patterns:
        m = re.search(pat, b)
        if m:
            return m.group(1)

    # Otherwise: token before first underscore
    if '_' in b:
        return b.split('_')[0]

    # Otherwise: use basename without extension
    return os.path.splitext(b)[0]


def _print_counts(title: str, samples: List[Tuple[str, int]]):
    ad = sum(1 for _, y in samples if y == 1)
    nc = sum(1 for _, y in samples if y == 0)
    total = len(samples)
    print(f"{title}")
    print(f"  Total: {total}")
    print(f"  AD:    {ad}")
    print(f"  NC:    {nc}")

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    val_split: float = 0.2,
):
    """
    Create train, validation, and test dataloaders with **patient-level** splitting
    and **stratification** on labels.

    Process:
      1. Load all 'train' samples (file paths + labels).
      2. Build 'groups' by extracting subject IDs from filenames.
      3. Group-stratified split (GroupShuffleSplit) into train/val (e.g., 80/20).
      4. Load 'test' samples separately and never touch them during training.

    Returns:
      train_loader, val_loader, test_loader
    """
    print("\n" + "=" * 80)
    print("CREATING DATASETS WITH TRAIN/VAL/TEST SPLIT (patient-level)")
    print("=" * 80)

    # Load training set (to be split into train/val by patient)
    print(f"\nLoading samples from: {data_dir}")
    train_all, train_counts = load_samples_from_folder(data_dir, 'train')
    _print_counts("\nOriginal train folder:", train_all)

    # Build arrays for stratified group split
    paths = [p for p, _ in train_all]
    labels = [y for _, y in train_all]
    groups = [extract_subject_id(p) for p in paths]

    # Grouped split by subject with stratification on labels (as much as GroupShuffleSplit allows)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    idx_train, idx_val = next(gss.split(paths, labels, groups))

    train_samples = [train_all[i] for i in idx_train]
    val_samples = [train_all[i] for i in idx_val]

    print("\nSplitting into train/val (patient-level):")
    print(f"  Train: {len(train_samples)} ({(1 - val_split) * 100:.0f}%)")
    print(f"  Val:   {len(val_samples)} ({val_split * 100:.0f}%)")

    # Test set is totally separate
    test_samples, test_counts = load_samples_from_folder(data_dir, 'test')

    # Transforms
    t_train = get_transforms('train', img_size)
    t_eval = get_transforms('val', img_size)

    # Datasets
    ds_train = ADNIDataset(train_samples, transform=t_train)
    ds_val = ADNIDataset(val_samples, transform=t_eval)
    ds_test = ADNIDataset(test_samples, transform=t_eval)

    # Sanity prints
    _print_counts("\nTrain split (after grouping):", train_samples)
    _print_counts("Val split (after grouping):  ", val_samples)
    print("\nTest folder (held-out):")
    print(f"  Total: {len(test_samples)}")
    print(f"  AD:    {test_counts.get('AD', 0)}")
    print(f"  NC:    {test_counts.get('NC', 0)}")

    # DataLoaders
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataLoader Summary:")
    print(f"  Train batches: {len(train_loader)} (batch_size={batch_size})")
    print(f"  Val batches:   {len(val_loader)} (batch_size={batch_size})")
    print(f"  Test batches:  {len(test_loader)} (batch_size={batch_size})")
    print("=" * 80 + "\n")

    return train_loader, val_loader, test_loader


# Simple self-test (optional)
if __name__ == '__main__':
    print("Testing ADNI Dataset Loader (patient-level split)...")
    print("=" * 80)
    data_dir_ = '/home/groups/comp3710/ADNI/AD_NC'
    if os.path.exists(data_dir_):
        tl, vl, te = get_dataloaders(
            data_dir=data_dir_,
            batch_size=16,
            num_workers=2,
            img_size=224,
            val_split=0.2
        )
        x, y = next(iter(tl))
        print("Train batch:", x.shape, y.tolist()[:8])
        x, y = next(iter(vl))
        print("Val batch:  ", x.shape, y.tolist()[:8])
        x, y = next(iter(te))
        print("Test batch: ", x.shape, y.tolist()[:8])
        print("\n✓ Dataset test passed!")
    else:
        print(f"Warning: Data directory not found: {data_dir_} (expected on non-cluster runs)")
