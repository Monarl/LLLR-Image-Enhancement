"""
Illumination Guidance Module (IGM) for MambaIRv2-Enhanced Low-Light Super-Resolution

Adapted from UltraIS paper implementation to work with MambaIRv2 architecture.
This module implements:
1. Grayscale conversion from RGB
2. Structure Expansion A(L_g) via max-pooling
3. Edge Preservation B(L_g) via local gradients  
4. Illumination Estimation Network (IENet) - lightweight U-Net
5. Multi-scale feature extraction I_f^(k)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


class IlluminationGuidance(nn.Module): # Might put this into pair_image_dataset.py
    """
    Illumination Guidance preprocessing module.
    
    Implements:
    - RGB to grayscale conversion 
    - Structure Expansion A(L_g): Max-pooling operations (horizontal & vertical)
    - Edge Preservation B(L_g): Local gradient differences 
    - IG(L_g) = A(L_g) + B(L_g)
    """
    
    def __init__(self):
        super(IlluminationGuidance, self).__init__()
    
    def rgb_to_grayscale(self, img_rgb: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB to grayscale using standard weights.
        
        Args:
            img_rgb: RGB image tensor [B, 3, H, W] in range [0, 1]
            
        Returns:
            Grayscale tensor [B, 1, H, W]
        """
        r, g, b = img_rgb[:, 0:1] + 1, img_rgb[:, 1:2] + 1, img_rgb[:, 2:3] + 1
        A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
        return A_gray
    
    def structure_expansion_A(self, L_g: torch.Tensor) -> torch.Tensor:
        """
        Structure Expansion A(L_g): Max-pooling strategy on local neighborhoods.
        
        Enlarges pixel distribution of bright regions and suppresses dark noise.
        
        Args:
            L_g: Grayscale tensor [B, 1, H, W]
            
        Returns:
            Structure expanded tensor [B, 1, H, W]
        """
        # Convert to numpy for max operations as in original UltraIS
        img = L_g.float().cpu().numpy()
        
        # Horizontal max operation
        x = np.maximum(img[:, :, :-1, :], img[:, :, 1:, :])  # Adjacent pixels max
        x = np.concatenate((x, np.expand_dims(img[:, :, -1, :], 2)), 2)  # Restore last row
        
        # Vertical max operation  
        y = np.maximum(x[:, :, :, :-1], x[:, :, :, 1:])  # Adjacent pixels max
        y = np.concatenate((y, np.expand_dims(x[:, :, :, -1], 3)), 3)  # Restore last column
        
        return torch.from_numpy(y).to(L_g.device)
    
    def edge_preservation_B(self, L_g: torch.Tensor) -> torch.Tensor:
        """
        Edge Preservation B(L_g): Local gradient differences.
        
        Captures high-frequency edge information and texture boundaries.
        
        Args:
            L_g: Grayscale tensor [B, 1, H, W]
            
        Returns:
            Edge preserved tensor [B, 1, H, W]  
        """
        # Convert to numpy for gradient operations as in original UltraIS
        img = L_g.float().cpu().numpy()

        # Horizontal gradients
        x1 = img[:, :, :-1, :] - img[:, :, 1:, :]  # Forward difference
        x1 = np.concatenate((x1, np.expand_dims(img[:, :, -1, :], 2)), 2)

        x2 = img[:, :, 1:, :] - img[:, :, :-1, :]  # Backward difference  
        x2 = np.concatenate((np.expand_dims(img[:, :, 0, :], 2), x2), 2)

        # Vertical gradients
        y1 = img[:, :, :, :-1] - img[:, :, :, 1:]  # Forward difference
        y1 = np.concatenate((y1, np.expand_dims(img[:, :, :, -1], 3)), 3)

        y2 = img[:, :, :, 1:] - img[:, :, :, :-1]  # Backward difference
        y2 = np.concatenate((np.expand_dims(img[:, :, :, 0], 3), y2), 3)

        # Average absolute gradients
        img = (np.abs(x1) + np.abs(x2) + np.abs(y1) + np.abs(y2)) / 4.0

        return torch.from_numpy(img).to(L_g.device)
    
    def forward(self, img_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of IlluminationGuidance.
        
        Args:
            img_rgb: RGB image [B, 3, H, W]
            
        Returns:
            Tuple of:
            - L_g: Grayscale image [B, 1, H, W] 
            - IG: Illumination guidance map [B, 2, H, W] = [A(L_g) + B(L_g), L_g]
        """
        # Step 1: RGB to grayscale
        L_g = self.rgb_to_grayscale(img_rgb)
        
        # Step 2: Structure expansion A(L_g) 
        A_Lg = self.structure_expansion_A(L_g)
        
        # Step 3: Edge preservation B(L_g)
        B_Lg = self.edge_preservation_B(L_g)
        
        # Step 4: Combine guidance
        guidance = A_Lg + B_Lg
        
        # Step 5: Concatenate guidance and grayscale as 2-channel input for IENet
        IG = torch.cat([guidance, L_g], dim=1)  # [B, 2, H, W]
        
        return L_g, IG


class IENet(nn.Module):
    """
    Illumination Estimation Network (IENet) - Lightweight U-Net.
    
    Takes 2-channel input: [guidance, grayscale] and outputs multi-scale illumination features.
    Fixed to 12-channel architecture for lightweight processing.
    """
    
    def __init__(self, inp_channels: int = 2, out_channels: int = 1, 
                 n_feat: int = 64, scale: int = 1, bias: bool = False):
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
    
    Combines IlluminationGuidance preprocessing with IENet to produce:
    1. Multi-scale illumination features I_f^(k) for ISDM-lite modulation
    2. Final illumination map for potential supervision
    """
    
    def __init__(self, n_feat: int = 64, scale: int = 1, bias: bool = False):
        super(IGMModule, self).__init__()
        
        self.illumination_guidance = IlluminationGuidance()
        self.ienet = IENet(inp_channels=2, out_channels=1, n_feat=n_feat, scale=scale, bias=bias)
        
    def forward(self, img_rgb: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass of complete IGM.
        
        Args:
            img_rgb: Input RGB low-light image [B, 3, H, W]
            
        Returns:
            Tuple of:
            - I_fk_list: Multi-scale illumination features for ISDM-lite [5 tensors]
            - illumination_map: Final illumination estimate [B, 1, H, W] 
            - L_g: Grayscale version [B, 1, H, W]
        """
        # Step 1: Generate illumination guidance
        L_g, IG = self.illumination_guidance(img_rgb)
        
        # Step 2: Extract multi-scale illumination features
        I_fk_list, illumination_map = self.ienet(IG)
        
        return I_fk_list, illumination_map, L_g


if __name__ == "__main__":
    # Unit test
    import torch
    
    # Test IlluminationGuidance
    print("Testing IlluminationGuidance...")
    ig = IlluminationGuidance()
    x = torch.randn(2, 3, 64, 64)  # Batch of 2, RGB, 64x64
    L_g, IG = ig(x)
    print(f"Input shape: {x.shape}")
    print(f"Grayscale shape: {L_g.shape}")
    print(f"Illumination Guidance shape: {IG.shape}")
    assert IG.shape == (2, 2, 64, 64), f"Expected IG shape (2, 2, 64, 64), got {IG.shape}"
    
    # Test IENet
    print("\nTesting IENet...")
    ienet = IENet(n_feat=64)
    I_fk_list, ill_map = ienet(IG)
    print(f"Number of multi-scale features: {len(I_fk_list)}")
    for i, feat in enumerate(I_fk_list):
        print(f"Feature {i+1} shape: {feat.shape}")
    print(f"Illumination map shape: {ill_map.shape}")
    assert len(I_fk_list) == 5, f"Expected 5 features, got {len(I_fk_list)}"
    assert ill_map.shape == (2, 1, 64, 64), f"Expected illumination map shape (2, 1, 64, 64), got {ill_map.shape}"
    
    # Test complete IGMModule
    print("\nTesting complete IGMModule...")
    igm = IGMModule(n_feat=64, scale=4)
    I_fk_list, ill_map, L_g = igm(x)
    print(f"IGM Multi-scale features: {len(I_fk_list)}")
    print(f"IGM Illumination map shape: {ill_map.shape}")
    print(f"IGM Grayscale shape: {L_g.shape}")
    
    print("\n✅ All tests passed! IGM module is ready for integration.")