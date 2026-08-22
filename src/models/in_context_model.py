import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.fusion_module import FusionModule
from src.models.backbone import load_universeg_backbone, get_frozen_features


class ConvBlock(nn.Module):
    """Dual Convolution Block: (Conv2d -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class InContextSegmentationModel(nn.Module):
    """
    In-Context Medical Image Segmentation Model with Frozen Pretrained Backbone.

    ARCHITECTURE OVERVIEW:
    =======================
    1. Frozen Pretrained Backbone (UniverSeg):
       Extracts multi-scale 2D spatial feature maps at resolutions [H, W], [H/2, W/2],
       and bottleneck [H/4, W/4]. All backbone parameters are frozen (requires_grad = False).

    2. Bottleneck In-Context Fusion (Trainable):
       Passes target Query latent features [B, C, H/4, W/4] and mask-conditioned Support
       latent features [B, S, C, H/4, W/4] into FusionModule (cross-attention).

    3. Decoder (Trainable):
       Combines fused bottleneck features with skip-connected Query features from the frozen
       backbone stage to decode full-resolution binary segmentation logits [B, 1, H, W].
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_channels: list = [64, 64, 64],
        num_heads: int = 4,
        dropout: float = 0.1,
        backbone: nn.Module = None
    ):
        """
        Args:
            in_channels (int): Input image channels (1 for grayscale/CT/FLAIR, 4 for multi-modal MRI).
            feature_channels (list): Features at [stage1, stage2, bottleneck]. Default: [64, 64, 64].
            num_heads (int): Cross-attention heads in FusionModule.
            dropout (float): Dropout probability.
            backbone (nn.Module, optional): Pretrained UniverSeg backbone module. Loaded automatically if None.
        """
        super().__init__()
        self.in_channels = in_channels
        c1, c2, c3 = feature_channels

        # 1. Frozen Pretrained Backbone (UniverSeg)
        if backbone is None:
            self.backbone = load_universeg_backbone()
        else:
            self.backbone = backbone

        # Ensure all backbone parameters are frozen
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 2. Bottleneck Cross-Attention Fusion Module (Trainable)
        self.fusion = FusionModule(channels=c3, num_heads=num_heads, dropout=dropout)

        # 3. Decoder stages (Trainable, with skip connections from Query frozen backbone)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        # Final 1x1 Conv Segmentation Head (Trainable)
        self.final_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def extract_features(self, x: torch.Tensor):
        """
        Extracts multi-scale frozen features for an image batch using UniverSeg backbone.
        Args:
            x (torch.Tensor): Shape [B, in_channels, H, W]
        Returns:
            f1 [B, 64, H, W], f2 [B, 64, H/2, W/2], f3 [B, 64, H/4, W/4]
        """
        return get_frozen_features(x, self.backbone)

    def get_parameter_count_breakdown(self):
        """
        Returns parameter counts broken down by frozen backbone vs trainable fusion & decoder.
        """
        frozen_params = sum(p.numel() for p in self.backbone.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        decoder_params = (
            sum(p.numel() for p in self.up2.parameters()) +
            sum(p.numel() for p in self.dec2.parameters()) +
            sum(p.numel() for p in self.up1.parameters()) +
            sum(p.numel() for p in self.dec1.parameters()) +
            sum(p.numel() for p in self.final_conv.parameters())
        )
        trainable_params = fusion_params + decoder_params
        total_params = frozen_params + trainable_params

        return {
            "frozen_backbone": frozen_params,
            "trainable_fusion": fusion_params,
            "trainable_decoder": decoder_params,
            "total_trainable": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": (trainable_params / total_params) * 100.0 if total_params > 0 else 0.0
        }

    def print_parameter_summary(self):
        counts = self.get_parameter_count_breakdown()
        print("==================================================")
        print(" IN-CONTEXT MODEL PARAMETER COUNT BREAKDOWN")
        print("==================================================")
        print(f" Frozen Backbone (UniverSeg): {counts['frozen_backbone']:,} ({100.0 - counts['trainable_percentage']:.2f}%)")
        print(f" Trainable Fusion Module:    {counts['trainable_fusion']:,}")
        print(f" Trainable Decoder:          {counts['trainable_decoder']:,}")
        print(f" Total Trainable Params:     {counts['total_trainable']:,} ({counts['trainable_percentage']:.2f}%)")
        print(f" Total Model Parameters:     {counts['total_parameters']:,}")
        print("==================================================")

    def forward(
        self,
        query_img: torch.Tensor,
        support_imgs: torch.Tensor,
        support_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward Pass of InContextSegmentationModel.

        Args:
            query_img (torch.Tensor): Target image [B, C_in, H, W]
            support_imgs (torch.Tensor): Support set images [B, S, C_in, H, W]
            support_masks (torch.Tensor): Support set masks [B, S, 1, H, W]

        Returns:
            torch.Tensor: Segmentation logits [B, 1, H, W]
        """
        B, C_in, H, W = query_img.shape
        _, S, _, _, _ = support_imgs.shape

        # Step 1: Extract Query multi-scale frozen features
        q_f1, q_f2, q_f3 = self.extract_features(query_img)  # q_f3 is [B, 64, H/4, W/4]

        # Step 2: Extract Support set multi-scale bottleneck frozen features
        support_imgs_flat = support_imgs.view(B * S, C_in, H, W)
        _, _, s_f3_flat = self.extract_features(support_imgs_flat)  # [B*S, 64, H/4, W/4]
        
        c3_spatial_h, c3_spatial_w = q_f3.shape[2], q_f3.shape[3]
        s_f3 = s_f3_flat.view(B, S, -1, c3_spatial_h, c3_spatial_w)  # [B, S, 64, H/4, W/4]

        # Step 3: Resize Support Masks to Bottleneck Spatial Resolution
        support_masks_flat = support_masks.view(B * S, 1, H, W)
        s_masks_down_flat = F.interpolate(
            support_masks_flat,
            size=(c3_spatial_h, c3_spatial_w),
            mode="nearest"
        )
        s_masks_down = s_masks_down_flat.view(B, S, 1, c3_spatial_h, c3_spatial_w)

        # Step 4: Apply Bottleneck In-Context Cross-Attention Fusion (Trainable)
        fused_bottleneck = self.fusion(q_f3, s_f3, s_masks_down)  # [B, 64, H/4, W/4]

        # Step 5: Decoder with Skip Connections (Trainable)
        # Stage 2 Up-sampling
        x = self.up2(fused_bottleneck)              # [B, 64, H/2, W/2]
        x = torch.cat([x, q_f2], dim=1)            # [B, 64 + 64, H/2, W/2]
        x = self.dec2(x)                           # [B, 64, H/2, W/2]

        # Stage 1 Up-sampling
        x = self.up1(x)                            # [B, 64, H, W]
        x = torch.cat([x, q_f1], dim=1)            # [B, 64 + 64, H, W]
        x = self.dec1(x)                           # [B, 64, H, W]

        # Final Logits Output
        logits = self.final_conv(x)                 # [B, 1, H, W]

        return logits


if __name__ == "__main__":
    print("==================================================")
    print(" Testing InContextSegmentationModel with Frozen Backbone")
    print("==================================================")

    model = InContextSegmentationModel(in_channels=1, feature_channels=[64, 64, 64], num_heads=4)
    model.print_parameter_summary()

    # Test single-channel input
    B, S, C_in, H, W = 2, 2, 1, 128, 128
    query_img = torch.randn(B, C_in, H, W)
    support_imgs = torch.randn(B, S, C_in, H, W)
    support_masks = torch.randint(0, 2, (B, S, 1, H, W)).float()

    model.eval()
    with torch.no_grad():
        out_logits = model(query_img, support_imgs, support_masks)

    print(f"Single-Channel Input Logits Shape: {list(out_logits.shape)}")
    assert out_logits.shape == (B, 1, H, W), f"Expected {(B, 1, H, W)}, got {out_logits.shape}"

    # Test multi-channel input (4-channel BrainTumour MRI)
    C_in_multi = 4
    query_img_m = torch.randn(B, C_in_multi, H, W)
    support_imgs_m = torch.randn(B, S, C_in_multi, H, W)

    with torch.no_grad():
        out_logits_m = model(query_img_m, support_imgs_m, support_masks)

    print(f"Multi-Channel (4D MRI) Logits Shape: {list(out_logits_m.shape)}")
    assert out_logits_m.shape == (B, 1, H, W), f"Expected {(B, 1, H, W)}, got {out_logits_m.shape}"

    print("--------------------------------------------------")
    print("SUCCESS: InContextSegmentationModel forward pass verified for single-channel and multi-channel inputs!")
    print("==================================================")
