"""
ConvNeXt Implementation for Alzheimer's Disease Classification
Based on: "A ConvNet for the 2020s" (Liu et al., 2022)
Paper: https://arxiv.org/abs/2201.03545
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block
    
    Architecture:
    - Depthwise Conv 7x7
    - LayerNorm
    - Linear (1x1 conv) - expansion
    - GELU activation
    - Linear (1x1 conv) - projection
    - Layer Scale
    - Residual connection
    """
    
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        
        # Depthwise convolution (7x7)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm (channels first)
        self.norm = LayerNorm(dim, eps=1e-6)
        
        # Pointwise/Inverted Bottleneck (1x1 convs)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Expansion
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)  # Projection
        
        # Layer Scale (learnable scaling parameter)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        
    def forward(self, x):
        input = x
        
        # Depthwise convolution
        x = self.dwconv(x)
        
        # Permute for LayerNorm: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        # LayerNorm + MLP
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer Scale
        if self.gamma is not None:
            x = self.gamma * x
        
        # Permute back: (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Residual connection
        x = input + x
        
        return x


class ConvNeXt(nn.Module):
    """
    ConvNeXt Model
    
    Args:
        in_chans: Number of input channels (3 for RGB)
        num_classes: Number of output classes (2 for AD/NC)
        depths: Number of blocks at each stage
        dims: Number of channels at each stage
        drop_path_rate: Stochastic depth rate
    """
    
    def __init__(
        self,
        in_chans=3,
        num_classes=2,
        depths=[3, 3, 9, 3],      # ConvNeXt-Tiny
        dims=[96, 192, 384, 768], # ConvNeXt-Tiny
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6
    ):
        super().__init__()
        
        # Stem: 4x4 conv with stride 4 (aggressive downsampling)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # 4 stages
        self.stages = nn.ModuleList()
        
        for i in range(4):
            # Downsampling layer (except first stage)
            if i > 0:
                downsample = nn.Sequential(
                    LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2)
                )
            else:
                downsample = nn.Identity()
            
            # Stack of ConvNeXt blocks
            stage = nn.Sequential(
                downsample,
                *[ConvNeXtBlock(dims[i], layer_scale_init_value) for _ in range(depths[i])]
            )
            
            self.stages.append(stage)
        
        # Classifier head
        self.norm = LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Stem
        x = self.stem(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        
        # Stages
        for stage in self.stages:
            # Permute back for conv layers
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            x = stage(x)
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        
        # Global average pooling
        x = x.mean(dim=[1, 2])  # (N, H, W, C) -> (N, C)
        
        # Classifier
        x = self.norm(x)
        x = self.head(x)
        
        return x


def convnext_tiny(num_classes=2):
    """ConvNeXt-Tiny model"""
    return ConvNeXt(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        num_classes=num_classes
    )


def convnext_small(num_classes=2):
    """ConvNeXt-Small model"""
    return ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        num_classes=num_classes
    )


# Test the model
if __name__ == '__main__':
    print("Testing ConvNeXt Model...")
    print("-" * 50)
    
    # Create model
    model = convnext_tiny(num_classes=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test with softmax
    probs = F.softmax(output, dim=1)
    print(f"\nProbabilities shape: {probs.shape}")
    print(f"Sample probabilities:\n{probs[0]}")
    print(f"Sum of probabilities: {probs[0].sum():.3f}")
    
    print("\n✓ ConvNeXt model working correctly!")