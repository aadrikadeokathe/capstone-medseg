import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionBlock(nn.Module):
    """
    A single cross-attention + FFN layer block for feature fusion.
    """

    def __init__(self, channels: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # 1. Mask Projection Layer: Maps 1-channel binary mask into C-channel embedding space
        self.mask_proj = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=1)

        # 2. Multi-Head Cross-Attention Block
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3. Normalization Layers for Residual Connections
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        # 4. Feed-Forward Network (FFN) for feature refinement after attention
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout)
        )

    def forward(self, query_feat: torch.Tensor, support_feats: torch.Tensor, support_masks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of a single FusionBlock.

        Args:
            query_feat (torch.Tensor): Query feature map of shape [B, C, H, W].
            support_feats (torch.Tensor): Support set feature maps of shape [B, S, C, H, W].
            support_masks (torch.Tensor): Support set binary masks of shape [B, S, 1, H, W].

        Returns:
            torch.Tensor: Updated Query feature map of shape [B, C, H, W].
        """
        B, C, H, W = query_feat.shape
        _, S, _, _, _ = support_feats.shape

        # Step 1: Mask Conditioning on Support Set Features
        support_masks_flat = support_masks.view(B * S, 1, H, W)
        mask_embed = self.mask_proj(support_masks_flat)  # [B*S, C, H, W]
        mask_embed = mask_embed.view(B, S, C, H, W)      # [B, S, C, H, W]

        conditioned_support = support_feats * (1.0 + torch.sigmoid(mask_embed))

        # Step 2: Prepare Query, Key, Value Tensors
        Q = query_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        K = conditioned_support.permute(0, 1, 3, 4, 2).reshape(B, S * H * W, C)
        V = K

        # Step 3: Multi-Head Cross Attention
        attn_out, _ = self.cross_attn(query=Q, key=K, value=V)  # [B, H*W, C]

        # Step 4: First Residual Connection & Layer Normalization
        x = self.norm1(Q + attn_out)  # [B, H*W, C]

        # Step 5: Feed-Forward Refinement & Second Residual Connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)    # [B, H*W, C]

        # Step 6: Reshape back to 2D Feature Map [B, C, H, W]
        fused_feat = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return fused_feat


class FusionModule(nn.Module):
    """
    Stacked In-Context Fusion Module for Few-Shot / In-Context Medical Image Segmentation.

    ARCHITECTURE OVERVIEW (for Report / Viva / Paper):
    ===================================================
    This module stacks multiple cross-attention FusionBlock layers (default depth `num_layers=3`),
    chained sequentially like transformer encoder/decoder layers.

    Each layer block N:
    1. Mask Conditioning:
       Projects support masks into feature space to gate support image features.
    2. Multi-Head Cross-Attention:
       Query feature tokens (from target image or previous block output) attend to
       Key/Value tokens from the conditioned support set.
    3. Residual Connections & FFN:
       Layer Normalization, Feed-Forward refinement, and skip connections.
    4. Sequential Chaining:
       Output feature map of block N feeds directly as input Query into block N+1.
    """

    def __init__(self, channels: int = 64, num_heads: int = 4, num_layers: int = 3, dropout: float = 0.1):
        """
        Args:
            channels (int): Number of feature channels (C). Default: 64.
            num_heads (int): Number of attention heads. Default: 4.
            num_layers (int): Number of stacked FusionBlock layers. Default: 3.
            dropout (float): Dropout probability. Default: 0.1.
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            FusionBlock(channels=channels, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query_feat: torch.Tensor, support_feats: torch.Tensor, support_masks: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass across stacked FusionBlock layers.

        Args:
            query_feat (torch.Tensor): Initial Query feature map [B, C, H, W].
            support_feats (torch.Tensor): Support set feature maps [B, S, C, H, W].
            support_masks (torch.Tensor): Support set binary masks [B, S, 1, H, W].

        Returns:
            torch.Tensor: Final fused feature map [B, C, H, W].
        """
        x = query_feat
        for layer in self.layers:
            x = layer(x, support_feats, support_masks)
        return x


if __name__ == "__main__":
    print("==================================================")
    print(" Testing Stacked FusionModule with Dummy Data")
    print("==================================================")

    # Realistic test dimensions
    B, C, H, W, S = 1, 64, 32, 32, 2

    dummy_query_feat = torch.randn(B, C, H, W)
    dummy_support_feats = torch.randn(B, S, C, H, W)
    dummy_support_masks = torch.randint(0, 2, (B, S, 1, H, W)).float()

    print(f"Input Query Features shape:  {list(dummy_query_feat.shape)}")
    print(f"Input Support Features shape: {list(dummy_support_feats.shape)}")
    print(f"Input Support Masks shape:    {list(dummy_support_masks.shape)}")
    print("--------------------------------------------------")

    for num_layers in [1, 2, 3, 4]:
        model = FusionModule(channels=C, num_heads=4, num_layers=num_layers, dropout=0.1)
        model.eval()

        with torch.no_grad():
            fused_output = model(dummy_query_feat, dummy_support_feats, dummy_support_masks)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"Depth num_layers={num_layers}: Output shape {list(fused_output.shape)} | Parameters: {param_count:,}")
        assert fused_output.shape == (B, C, H, W), f"Shape mismatch for num_layers={num_layers}!"

    print("--------------------------------------------------")
    print("SUCCESS: Stacked FusionModule verified for num_layers = 1, 2, 3, 4!")
    print("==================================================")

