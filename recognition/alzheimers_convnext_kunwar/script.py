"""
Visualise latent space using t-SNE
"""

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from modules import get_model
from dataset import get_dataloaders


def extract_features(model, loader, device):
    """Extract features from the model's backbone"""
    model.eval()
    features = []
    labels = []
    
    # Hook to capture features before classifier
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hook on the pooling layer
    model.pool.register_forward_hook(get_activation('pool'))
    
    with torch.no_grad():
        for images, labs in tqdm(loader):
            images = images.to(device)
            _ = model(images)
            
            # Get features after pooling
            feat = activation['pool'].squeeze().cpu().numpy()
            features.append(feat)
            labels.extend(labs.numpy())
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    return features, labels


def visualise_latent_space(checkpoint_path='./checkpoints/best_model.pth'):
    """Create t-SNE visualisation of latent space"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = get_model('convnext_base', num_classes=2, pretrained=False, dropout=0.4)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    
    # Load data
    train_loader, test_loader = get_dataloaders(
        '/home/groups/comp3710/ADNI/AD_NC',
        batch_size=32,
        num_workers=4
    )
    
    print("Extracting train features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    
    print("Extracting test features...")
    test_features, test_labels = extract_features(model, test_loader, device)
    
    # Sample subset for visualisation (t-SNE is slow)
    n_samples = 2000
    train_idx = np.random.choice(len(train_features), min(n_samples, len(train_features)), replace=False)
    test_idx = np.random.choice(len(test_features), min(n_samples, len(test_features)), replace=False)
    
    train_features_sample = train_features[train_idx]
    train_labels_sample = train_labels[train_idx]
    test_features_sample = test_features[test_idx]
    test_labels_sample = test_labels[test_idx]
    
    print("Running t-SNE on train features...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    train_embedded = tsne.fit_transform(train_features_sample)
    
    print("Running t-SNE on test features...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    test_embedded = tsne.fit_transform(test_features_sample)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Train set
    for label, name, color in [(0, 'AD', 'red'), (1, 'NC', 'blue')]:
        mask = train_labels_sample == label
        ax1.scatter(train_embedded[mask, 0], train_embedded[mask, 1], 
                   c=color, label=name, alpha=0.6, s=20)
    ax1.set_title('Training Set Latent Space (t-SNE)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test set
    for label, name, color in [(0, 'AD', 'red'), (1, 'NC', 'blue')]:
        mask = test_labels_sample == label
        ax2.scatter(test_embedded[mask, 0], test_embedded[mask, 1], 
                   c=color, label=name, alpha=0.6, s=20)
    ax2.set_title('Test Set Latent Space (t-SNE)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latent_space_visualisation.png', dpi=150)
    print("Saved latent_space_visualisation.png")
    plt.close()
    
    # Check separability
    print("\n" + "="*60)
    print("LATENT SPACE ANALYSIS")
    print("="*60)
    
    # Compute distances between class centroids
    train_ad_centroid = train_embedded[train_labels_sample == 0].mean(axis=0)
    train_nc_centroid = train_embedded[train_labels_sample == 1].mean(axis=0)
    train_separation = np.linalg.norm(train_ad_centroid - train_nc_centroid)
    
    test_ad_centroid = test_embedded[test_labels_sample == 0].mean(axis=0)
    test_nc_centroid = test_embedded[test_labels_sample == 1].mean(axis=0)
    test_separation = np.linalg.norm(test_ad_centroid - test_nc_centroid)
    
    print(f"Train set class separation: {train_separation:.2f}")
    print(f"Test set class separation: {test_separation:.2f}")
    
    # Compute within-class variance
    train_ad_var = train_embedded[train_labels_sample == 0].var(axis=0).mean()
    train_nc_var = train_embedded[train_labels_sample == 1].var(axis=0).mean()
    test_ad_var = test_embedded[test_labels_sample == 0].var(axis=0).mean()
    test_nc_var = test_embedded[test_labels_sample == 1].var(axis=0).mean()
    
    print(f"\nTrain AD variance: {train_ad_var:.2f}")
    print(f"Train NC variance: {train_nc_var:.2f}")
    print(f"Test AD variance: {test_ad_var:.2f}")
    print(f"Test NC variance: {test_nc_var:.2f}")
    
    if train_ad_var > test_ad_var * 1.5:
        print("\n⚠ WARNING: Train AD samples are MORE spread out than test AD samples")
        print("   This suggests the model memorized diverse AD patterns in training")
        print("   but can't generalise to unseen AD cases!")
    
    if train_separation > test_separation * 1.2:
        print("\n⚠ WARNING: Classes are BETTER separated in training than testing")
        print("   This confirms overfitting - model found spurious patterns!")


if __name__ == '__main__':
    visualise_latent_space()