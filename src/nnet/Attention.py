# Attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):
    """Squeeze-and-Excitation Attention Module"""
    
    def __init__(self, channels, reduction_ratio=16):
        """
        Args:
            channels (int): Number of input channels
            reduction_ratio (int): Reduction ratio for the bottleneck layer
        """
        super(SE, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: Two fully connected layers
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width)
        Returns:
            torch.Tensor: Output tensor with channel-wise attention weights applied
        """
        batch_size, channels, height, width = x.size()
        
        # Squeeze: Global average pooling
        squeeze = self.global_avg_pool(x).view(batch_size, channels)
        
        # Excitation: Get channel weights
        excitation = self.excitation(squeeze).view(batch_size, channels, 1, 1)
        
        # Scale: Apply channel weights to input
        return x * excitation.expand_as(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial Attention)"""
    
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        """
        Args:
            channels (int): Number of input channels
            reduction_ratio (int): Reduction ratio for channel attention
            kernel_size (int): Kernel size for spatial attention
        """
        super(CBAM, self).__init__()
        self.channels = channels
        
        # Channel Attention Module
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        
        # Spatial Attention Module
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width)
        Returns:
            torch.Tensor: Output tensor with both channel and spatial attention applied
        """
        # Apply channel attention first
        x = self.channel_attention(x)
        
        # Then apply spatial attention
        x = self.spatial_attention(x)
        
        return x

class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM"""
    
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.channels = channels
        
        # Both average and max pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average pooling branch
        avg_out = self.mlp(self.avg_pool(x))
        
        # Max pooling branch
        max_out = self.mlp(self.max_pool(x))
        
        # Combine and apply sigmoid
        channel_weights = self.sigmoid(avg_out + max_out)
        
        return x * channel_weights

class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # Convolution layer for spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Concatenate average and max pooling along channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        spatial_features = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply convolution and sigmoid
        spatial_weights = self.sigmoid(self.conv(spatial_features))
        
        return x * spatial_weights

# Factory function to create attention modules
def create_attention(attention_type, channels, **kwargs):
    """
    Factory function to create attention modules
    
    Args:
        attention_type (str): Type of attention ('se', 'cbam', or None)
        channels (int): Number of input channels
        **kwargs: Additional arguments for the attention module
    
    Returns:
        nn.Module or None: Attention module instance
    """
    if attention_type is None or attention_type.lower() == 'none':
        return None
    
    attention_type = attention_type.lower()
    
    if attention_type == 'se':
        reduction_ratio = kwargs.get('reduction_ratio', 16)
        return SE(channels, reduction_ratio)
    
    elif attention_type == 'cbam':
        reduction_ratio = kwargs.get('reduction_ratio', 16)
        kernel_size = kwargs.get('kernel_size', 7)
        return CBAM(channels, reduction_ratio, kernel_size)
    
    else:
        raise ValueError(f"Unsupported attention type: {attention_type}")
    
class fwChannelAttention(nn.Module):
    """frequency-Wise Channel Attention Module"""
