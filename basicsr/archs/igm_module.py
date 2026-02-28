"""
Illumination Guidance Module (IGM) for MambaIRv2-Enhanced Low-Light Super-Resolution

Adapted from UltraIS paper implementation to work with MambaIRv2 architecture.

This module implements:
1. Illumination Estimation Network (IENet) - lightweight U-Net
2. Multi-scale feature extraction I_f^(k)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ContextBlock(nn.Module):
    """Context attention block from UltraIS."""
    
    def __init__(self, n_feat: int, bias: bool = False):
        super(ContextBlock, self).__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=bias, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=3, bias=bias, padding=1, groups=2)
        )

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )
        
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def modeling(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, C, H, W]
        inp = x
        inp = self.head(inp)
        
        # [N, C, 1, 1]
        context = self.modeling(inp)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        inp = inp + channel_add_term
        x = x + self.act(inp)

        return x


class RCBdown(nn.Module):
    """Residual Context Block for downsampling path."""
    
    def __init__(self, n_feat: int, kernel_size: int = 3, reduction: int = 8, 
                 bias: bool = False, groups: int = 1):
        super(RCBdown, self).__init__()
        
        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential( 
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act
        )

        self.act = act
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        res = self.act(self.gcnet(x))
        res += x
        return res


class RCBup(nn.Module):
    """Residual Context Block for upsampling path."""
    
    def __init__(self, n_feat: int, kernel_size: int = 3, reduction: int = 8, 
                 bias: bool = False, groups: int = 1):
        super(RCBup, self).__init__()
        
        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(2*n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act
        )

        self.act = act
        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        res = self.act(self.gcnet(x))
        res += x
        return res


class IENet(nn.Module):
    """
    Illumination Estimation Network (IENet) - Lightweight U-Net.
    
    Takes 2-channel input: [guidance, grayscale] and 
    outputs multi-scale illumination features.
    """
    
    def __init__(self, inp_channels: int = 2, out_channels: int = 1, 
                 n_feat: int = 48, scale: int = 1, bias: bool = False):
        super(IENet, self).__init__()
        
        self.scale = scale
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        
        # Input conv
        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, stride=1, padding=1)
        
        # Encoder
        self.conv1 = RCBdown(n_feat=n_feat)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = RCBdown(n_feat=n_feat)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = RCBdown(n_feat=n_feat)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = RCBdown(n_feat=n_feat)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # Bottleneck
        self.conv5 = RCBdown(n_feat=n_feat)
        
        # Decoder
        self.upv6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        self.conv6 = RCBup(n_feat=n_feat)
        
        self.upv7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        self.conv7 = RCBup(n_feat=n_feat)
        
        self.upv8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        self.conv8 = RCBup(n_feat=n_feat)
        
        self.upv9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
        self.conv9 = RCBup(n_feat=n_feat)
        
        # Final layers
        self.conv10_1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1)
        self.conv_out_r = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias)
        
        # Second lrelu definition (matches UltraIS line 464 - different inplace setting)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, refl: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass of IENet.
        
        Args:
            refl: 2-channel input [B, 2, H, W] from IlluminationGuidance
            
        Returns:
            Tuple of:
            - Multi-scale features list [conv5, conv6, conv7, conv8, conv9]
            - Final illumination map [B, 1, H, W]
        """
        # Input
        conv1 = self.lrelu(self.conv_in(refl))        
        conv1 = self.conv1(conv1)
        pool1 = self.pool1(conv1)        
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)        
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)        
        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)        
        
        # Bottleneck
        conv5 = self.conv5(pool4)         
        
        # Decoder with skip connections
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)        
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)        
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)        
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)        
        
        # Final output
        out = self.conv10_1(conv9) 
        out = torch.sigmoid(self.conv_out_r(out))
        
        # Return multi-scale features for ISDM-lite modulation
        # conv5: k=1 (deepest, global context)
        # conv6: k=2 (global context)  
        # conv7: k=3 (medium context)
        # conv8: k=4 (local details)
        # conv9: k=5 (finest details)
        multi_scale_features = [conv5, conv6, conv7, conv8, conv9]
        
        return multi_scale_features, out
    
    def _initialize_weights(self):
        """Initialize weights for all modules"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)


class IGMModule(nn.Module):
    """
    Complete Illumination Guidance Module (IGM).
    
    Takes precomputed 2-channel illumination guidance [guidance, grayscale]
    (computed in paired_image_dataset.py) and produces:
    1. Multi-scale illumination features I_f^(k) for ISDM-lite modulation
    2. Final illumination map for potential supervision
    """
    
    def __init__(self, n_feat: int = 48, scale: int = 1, bias: bool = False):
        super(IGMModule, self).__init__()
        
        self.ienet = IENet(inp_channels=2, out_channels=1, n_feat=n_feat, scale=scale, bias=bias)
        
    def forward(self, gray: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass of complete IGM.
        
        Args:
            gray: Precomputed 2-channel illumination guidance [B, 2, H, W]
                  Channel 0: IG(L_g) = A(L_g) + B(L_g) (guidance)
                  Channel 1: L_g (grayscale)
                  Computed in paired_image_dataset.py
            
        Returns:
            Tuple of:
            - I_fk_list: Multi-scale illumination features for ISDM-lite [5 tensors]
            - illumination_map: Final illumination estimate [B, 1, H, W]
        """
        I_fk_list, illumination_map = self.ienet(gray)
        
        return I_fk_list, illumination_map


if __name__ == "__main__":
    # Unit test
    import torch
    import numpy as np
    
    def max_operation(img):
        """Structure Expansion A(L_g) - same as in paired_image_dataset.py"""
        img_np = img.float().numpy()
        x = np.maximum(img_np[:, :, :-1, :], img_np[:, :, 1:, :])
        x = np.concatenate((x, np.expand_dims(img_np[:, :, -1, :], 2)), 2)
        y = np.maximum(x[:, :, :, :-1], x[:, :, :, 1:])
        y = np.concatenate((y, np.expand_dims(x[:, :, :, -1], 3)), 3)
        return torch.from_numpy(y)

    def edge_operation(img):
        """Edge Preservation B(L_g) - same as in paired_image_dataset.py"""
        img_np = img.float().numpy()
        x1 = img_np[:, :, :-1, :] - img_np[:, :, 1:, :]
        x1 = np.concatenate((x1, np.expand_dims(img_np[:, :, -1, :], 2)), 2)
        x2 = img_np[:, :, 1:, :] - img_np[:, :, :-1, :]
        x2 = np.concatenate((np.expand_dims(img_np[:, :, 0, :], 2), x2), 2)
        y1 = img_np[:, :, :, :-1] - img_np[:, :, :, 1:]
        y1 = np.concatenate((y1, np.expand_dims(img_np[:, :, :, -1], 3)), 3)
        y2 = img_np[:, :, :, 1:] - img_np[:, :, :, :-1]
        y2 = np.concatenate((np.expand_dims(img_np[:, :, :, 0], 3), y2), 3)
        img_np = (np.abs(x1) + np.abs(x2) + np.abs(y1) + np.abs(y2)) / 4.0
        return torch.from_numpy(img_np)

    def simulate_dataset_preprocessing(img_lq):
        """Simulates paired_image_dataset.py illumination guidance preprocessing.
        
        Note: In the actual dataset, img_lq is [C, H, W] (single image, no batch dim).
        Here we accept [1, C, H, W] for convenience and return [2, H, W].
        """
        # Remove batch dim to match dataset behavior (dataset works per-image)
        img_lq = img_lq.squeeze(0)  # [C, H, W]
        r, g, b = img_lq[0] + 1, img_lq[1] + 1, img_lq[2] + 1
        A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
        A_gray = torch.unsqueeze(A_gray, 0)  # [1, H, W]
        A_gray = torch.unsqueeze(A_gray, 0)  # [1, 1, H, W]
        max_out = max_operation(A_gray)
        edge_out = edge_operation(A_gray)
        guidance = max_out + edge_out
        gray = torch.cat([guidance, A_gray], 1).squeeze(0)  # [2, H, W]
        return gray

    # Test dataset preprocessing simulation
    print("Testing dataset preprocessing simulation...")
    x = torch.randn(2, 3, 64, 64)  # Batch of 2, RGB, 64x64
    # Simulate per-image preprocessing as dataset would do
    gray_list = [simulate_dataset_preprocessing(x[i:i+1]) for i in range(x.shape[0])]
    gray = torch.stack(gray_list)  # [B, 2, H, W]
    print(f"Input shape: {x.shape}")
    print(f"Illumination Guidance (gray) shape: {gray.shape}")
    assert gray.shape == (2, 2, 64, 64), f"Expected gray shape (2, 2, 64, 64), got {gray.shape}"
    
    # Test IENet with n_feat=48
    print("\nTesting IENet (n_feat=48)...")
    ienet = IENet(n_feat=48)
    I_fk_list, ill_map = ienet(gray)
    print(f"Number of multi-scale features: {len(I_fk_list)}")
    for i, feat in enumerate(I_fk_list):
        print(f"Feature {i+1} shape: {feat.shape}")
    print(f"Illumination map shape: {ill_map.shape}")
    assert len(I_fk_list) == 5, f"Expected 5 features, got {len(I_fk_list)}"
    assert ill_map.shape == (2, 1, 64, 64), f"Expected illumination map shape (2, 1, 64, 64), got {ill_map.shape}"
    # Verify all features have 48 channels (matching embed_dim)
    for i, feat in enumerate(I_fk_list):
        assert feat.shape[1] == 48, f"Feature {i+1} has {feat.shape[1]} channels, expected 48"
    print("All features have 48 channels - matches MambaIRv2 embed_dim!")
    
    # Test complete IGMModule with n_feat=48
    print("\nTesting complete IGMModule (n_feat=48)...")
    igm = IGMModule(n_feat=48, scale=4)
    I_fk_list, ill_map = igm(gray)
    print(f"IGM Multi-scale features: {len(I_fk_list)}")
    print(f"IGM Illumination map shape: {ill_map.shape}")
    
    print("\n All tests passed! IGM module is ready for integration.")
    print("  - Preprocessing moved to paired_image_dataset.py")
    print("  - IENet n_feat=48 matches MambaIRv2 embed_dim (no channel adaptation needed)")