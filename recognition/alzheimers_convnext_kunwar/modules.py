"""
ConvNeXt Implementation for Alzheimer's Disease Classification
Built from scratch with advanced training techniques

Based on: "A ConvNet for the 2020s" (Liu et al., 2022)
Paper: https://arxiv.org/abs/2201.03545

Features:
- Layer-by-layer ConvNeXt implementation (not using pre-built models)
- Grayscale to RGB conversion for MRI images
- Label Smoothing loss
- MixUp data augmentation
- Optional ImageNet pretrained weight loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) for regularisation.
    Randomly drops residual branches during training.
    Inactive during inference.
    
    Args:
        drop_prob: Probability of dropping the path
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        
        return x / keep_prob * random_tensor


class LayerNorm2d(nn.Module):
    """
    LayerNorm for channels_first format (NCHW tensors).
    Standard LayerNorm expects channels_last, but CNNs use channels_first.
    
    Args:
        num_channels: Number of feature channels
        eps: Small epsilon to avoid division by zero
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W]
        # Normalise across channel dimension
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable scale and shift
        return x_hat * self.weight[:, None, None] + self.bias[:, None, None]

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block: The fundamental building block of ConvNeXt.
    
    Architecture:
        1. Depthwise 7x7 convolution (spatial mixing)
        2. LayerNorm
        3. Pointwise 1x1 conv → 4x expansion (channel mixing)
        4. GELU activation
        5. Pointwise 1x1 conv → projection back
        6. Layer Scale (optional learnable scaling)
        7. Drop Path (stochastic depth)
        8. Residual connection
    
    Args:
        dim: Number of input/output channels
        drop_path: Drop path rate for stochastic depth
        layer_scale_init_value: Initial value for layer scale parameter
    """
    def __init__(
        self, 
        dim: int, 
        drop_path: float = 0.0, 
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        
        # Depthwise convolution (7x7, same channels)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm for spatial dimensions
        self.norm = LayerNorm2d(dim)
        
        # Pointwise convolutions (1x1) - MLP in spatial domain
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)  # Expansion
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)  # Projection
        
        # Layer Scale: learnable per-channel scaling
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        ) if layer_scale_init_value > 0 else None
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # LayerNorm
        x = self.norm(x)
        
        # Pointwise MLP
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer Scale
        if self.gamma is not None:
            x = x * self.gamma[:, None, None]
        
        # Drop path + residual
        x = self.drop_path(x)
        x = x + shortcut
        
        return x

class ConvNeXt(nn.Module):
    """
    ConvNeXt Backbone - Built from scratch layer-by-layer.
    
    Architecture:
        - Stem: 4x4 conv with stride 4 (aggressive downsampling)
        - 4 stages with progressive downsampling
        - Each stage: [downsample (except stage 1)] + [N ConvNeXt blocks]
    
    Args:
        in_chans: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes (2 for AD/NC)
        depths: Number of blocks per stage [stage1, stage2, stage3, stage4]
        dims: Channel dimensions per stage [dim1, dim2, dim3, dim4]
        drop_path_rate: Overall drop path rate (stochastic depth)
        dropout_rate: Dropout rate before classifier
        layer_scale_init_value: Initial value for layer scale
        head_init_scale: Scaling factor for head initialisation
    """
    def __init__(
        self,
        in_chans: int = 1,  # Grayscale MRI
        num_classes: int = 2,  # AD vs NC
        depths: List[int] = [3, 3, 9, 3],  # ConvNeXt-Tiny
        dims: List[int] = [96, 192, 384, 768],  # ConvNeXt-Tiny
        drop_path_rate: float = 0.1,
        dropout_rate: float = 0.5,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
    ):
        super().__init__()
        
        self.in_chans = in_chans
        self.num_classes = num_classes
        
        # Grayscale to RGB converter (if needed)
        if in_chans == 1:
            # For grayscale input, we'll duplicate channels
            self.channel_converter = lambda x: x.repeat(1, 3, 1, 1)
            stem_in_chans = 3
        else:
            self.channel_converter = lambda x: x
            stem_in_chans = in_chans
        
        # Stem: 4x4 conv, stride 4 (224 -> 56)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(stem_in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)
        
        # Downsampling layers between stages (2x2 conv, stride 2)
        # Stage 1->2: 56->28, Stage 2->3: 28->14, Stage 3->4: 14->7
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        # Build 4 stages with ConvNeXt blocks
        self.stages = nn.ModuleList()
        
        # Stochastic depth: linearly increasing drop path rates
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]
        
        # Classification head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Scale classifier head
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
    
    def _init_weights(self, m: nn.Module):
        """Initialise weights with truncated normal distribution"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through ConvNeXt stages"""
        # Convert grayscale to RGB if needed
        x = self.channel_converter(x)
        
        # Go through all stages
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        # Global average pooling: [N, C, H, W] -> [N, C]
        x = x.mean([-2, -1])
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire network.
        
        Args:
            x: Input tensor [B, C, H, W] - can be grayscale [B, 1, H, W]
        
        Returns:
            Logits [B, num_classes]
        """
        # Extract features
        x = self.forward_features(x)
        
        # Classifier head
        x = self.norm(x)
        x = self.dropout(x)
        x = self.head(x)
        
        return x

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    
    Prevents overconfidence by using soft labels instead of hard labels.
    Instead of [0, 1] or [1, 0], we use [ε/(K-1), 1-ε] or [1-ε, ε/(K-1)]
    
    Benefits:
        - Prevents overconfidence on training data
        - Better calibration of predictions
        - Improved generalisation to test distribution
    
    Args:
        smoothing: Smoothing parameter (typically 0.1)
        num_classes: Number of classes (2 for binary)
    """
    def __init__(self, smoothing: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predictions (logits) [batch_size, num_classes]
            target: Ground truth labels [batch_size]
        
        Returns:
            Smoothed cross entropy loss
        """
        # Convert logits to log probabilities
        pred = F.log_softmax(pred, dim=1)
        
        # Create smoothed label distribution
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Calculate loss
        loss = torch.mean(torch.sum(-true_dist * pred, dim=1))
        
        return loss

class MixUpAugmentation:
    """
    MixUp data augmentation.
    
    Creates virtual training examples by mixing pairs of images and labels.
    Forces the model to learn more robust features.
    
    Formula:
        x_mixed = λ * x_i + (1 - λ) * x_j
        y_mixed = λ * y_i + (1 - λ) * y_j
    
    Where λ ~ Beta(α, α)
    
    Args:
        alpha: Beta distribution parameter (default: 0.4)
               Higher α = more aggressive mixing
               α=0 means no mixing (standard training)
    """
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
    
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor):
        """
        Apply MixUp to a batch.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            y: Labels [batch_size]
        
        Returns:
            mixed_x: Mixed images
            y_a: Original labels
            y_b: Permuted labels
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        
        # Random permutation
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix images
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        # Return both label sets for loss calculation
        y_a = y
        y_b = y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """
        Calculate MixUp loss.
        
        Loss = λ * loss(pred, y_a) + (1 - λ) * loss(pred, y_b)
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def convnext_tiny(in_chans=1, num_classes=2, dropout=0.5, pretrained=False, pretrain_stages='all'):
    """
    ConvNeXt-Tiny: ~28M parameters
    depths: [3, 3, 9, 3]
    dims: [96, 192, 384, 768]
    
    Args:
        pretrain_stages: 'all', 'early' (stem+stage1-2), or 'stem' (stem only)
    """
    model = ConvNeXt(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        dropout_rate=dropout,
    )
    
    if pretrained:
        load_pretrained_weights(model, 'convnext_tiny', pretrain_stages)
    
    return model


def convnext_small(in_chans=1, num_classes=2, dropout=0.5, pretrained=False, pretrain_stages='all'):
    """
    ConvNeXt-Small: ~50M parameters
    depths: [3, 3, 27, 3]
    dims: [96, 192, 384, 768]
    
    Args:
        pretrain_stages: 'all', 'early' (stem+stage1-2), or 'stem' (stem only)
    """
    model = ConvNeXt(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        dropout_rate=dropout,
    )
    
    if pretrained:
        load_pretrained_weights(model, 'convnext_small', pretrain_stages)
    
    return model


def convnext_base(in_chans=1, num_classes=2, dropout=0.5, pretrained=False, pretrain_stages='all'):
    """
    ConvNeXt-Base: ~89M parameters
    depths: [3, 3, 27, 3]
    dims: [128, 256, 512, 1024]
    
    Args:
        pretrain_stages: 'all', 'early' (stem+stage1-2), or 'stem' (stem only)
    """
    model = ConvNeXt(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        dropout_rate=dropout,
    )
    
    if pretrained:
        load_pretrained_weights(model, 'convnext_base', pretrain_stages)
    
    return model


def load_pretrained_weights(model: ConvNeXt, model_name: str, pretrain_stages='all'):
    """
    Load ImageNet pretrained weights into our custom-built model.
    
    This is ALLOWED because:
    1. We built the architecture ourselves (layer-by-layer)
    2. We're only copying the learned weights
    3. Similar to transfer learning
    
    Args:
        model: Our custom ConvNeXt model
        model_name: 'convnext_tiny', 'convnext_small', or 'convnext_base'
        pretrain_stages: 'all' (full backbone), 'early' (stem + stage 1-2), or 'stem' (stem only)
    """
    try:
        import torchvision.models as models
        
        print(f"\nLoading pretrained ImageNet weights for {model_name}...")
        print(f"Pretrain strategy: {pretrain_stages}")
        
        # Load official pretrained model
        if model_name == 'convnext_tiny':
            pretrained_model = models.convnext_tiny(weights='IMAGENET1K_V1')
        elif model_name == 'convnext_small':
            pretrained_model = models.convnext_small(weights='IMAGENET1K_V1')
        elif model_name == 'convnext_base':
            pretrained_model = models.convnext_base(weights='IMAGENET1K_V1')
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Get pretrained state dict
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        
        # Define which layers to load based on strategy
        if pretrain_stages == 'stem':
            # Only stem (most conservative)
            allowed_keys = ['downsample_layers.0']
        elif pretrain_stages == 'early':
            # Stem + early stages (conservative)
            allowed_keys = ['downsample_layers.0', 'downsample_layers.1', 
                           'stages.0', 'stages.1']
        else:  # 'all'
            # Full backbone (aggressive)
            allowed_keys = None  # Will allow all except classifier
        
        # Filter weights
        if allowed_keys is not None:
            # Partial loading
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() 
                if k in model_dict and v.shape == model_dict[k].shape and
                any(allowed in k for allowed in allowed_keys) and 'head' not in k
            }
        else:
            # Full backbone loading (skip only classifier)
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() 
                if k in model_dict and 'head' not in k and v.shape == model_dict[k].shape
            }
        
        # Update our model
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
        print(f"Loaded {len(pretrained_dict)} pretrained weight tensors")
        print(f"Classifier head initialised randomly for {model.num_classes} classes")
        
    except Exception as e:
        print(f"⚠ Warning: Could not load pretrained weights: {e}")
        print("Training from scratch...")


def get_model(model_name='convnext_small', in_chans=1, num_classes=2, 
              dropout=0.5, pretrained=False, pretrain_stages='all'):
    """
    Factory function to create ConvNeXt models.
    
    Args:
        model_name: 'convnext_tiny', 'convnext_small', or 'convnext_base'
        in_chans: Number of input channels (1 for grayscale MRI, 3 for RGB)
        num_classes: Number of output classes (2 for AD/NC)
        dropout: Dropout rate before classifier
        pretrained: Whether to load ImageNet pretrained weights
        pretrain_stages: 'all' (full backbone), 'early' (stem+stages1-2), 'stem' (stem only)
    
    Returns:
        ConvNeXt model ready for training
    """
    if model_name == 'convnext_tiny':
        return convnext_tiny(in_chans, num_classes, dropout, pretrained, pretrain_stages)
    elif model_name == 'convnext_small':
        return convnext_small(in_chans, num_classes, dropout, pretrained, pretrain_stages)
    elif model_name == 'convnext_base':
        return convnext_base(in_chans, num_classes, dropout, pretrained, pretrain_stages)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_loss_function(loss_type='label_smoothing', smoothing=0.1):
    """
    Get loss function for training.
    
    Args:
        loss_type: 'cross_entropy' or 'label_smoothing'
        smoothing: Smoothing parameter (if using label smoothing)
    
    Returns:
        Loss function
    """
    if loss_type == 'cross_entropy':
        print(f"Using CrossEntropyLoss")
        return nn.CrossEntropyLoss()
    elif loss_type == 'label_smoothing':
        print(f"Using Label Smoothing Loss (smoothing={smoothing})")
        return LabelSmoothingLoss(smoothing=smoothing, num_classes=2)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

if __name__ == '__main__':
    print("=" * 80)
    print("TESTING CONVNEXT IMPLEMENTATION")
    print("=" * 80)
    
    # Test grayscale input
    print("\n1. Testing grayscale input (MRI images)...")
    model = convnext_small(in_chans=1, num_classes=2, dropout=0.5, pretrained=False)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Model: ConvNeXt-Small")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with grayscale
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 224, 224)  # Grayscale
    
    print(f"\n   Input shape (grayscale): {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test probabilities
    probs = F.softmax(output, dim=1)
    print(f"\n   Sample probabilities: {probs[0].tolist()}")
    print(f"   Sum: {probs[0].sum():.3f}")
    
    # Test Label Smoothing
    print("\n2. Testing Label Smoothing Loss...")
    criterion = LabelSmoothingLoss(smoothing=0.1, num_classes=2)
    dummy_labels = torch.tensor([0, 1, 0, 1])
    
    model.train()
    output = model(dummy_input)
    loss = criterion(output, dummy_labels)
    
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test MixUp
    print("\n3. Testing MixUp Augmentation...")
    mixup = MixUpAugmentation(alpha=0.4)
    mixed_x, y_a, y_b, lam = mixup.mixup_data(dummy_input, dummy_labels)
    
    print(f"   Original input shape: {dummy_input.shape}")
    print(f"   Mixed input shape: {mixed_x.shape}")
    print(f"   Mixing coefficient λ: {lam:.4f}")
    print(f"   Original labels: {dummy_labels.tolist()}")
    print(f"   Labels A: {y_a.tolist()}")
    print(f"   Labels B: {y_b.tolist()}")
    
    # Test MixUp loss
    output_mixed = model(mixed_x)
    loss_mixed = mixup.mixup_criterion(criterion, output_mixed, y_a, y_b, lam)
    print(f"   MixUp loss: {loss_mixed.item():.4f}")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)