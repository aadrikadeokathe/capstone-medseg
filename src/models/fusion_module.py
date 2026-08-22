import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    """
    In-Context Fusion Module for Few-Shot / In-Context Medical Image Segmentation.

    ARCHITECTURE OVERVIEW (for Report / Viva):
    ===========================================
    This module performs cross-attention between a Target Query image's feature map
    and a Support Set of reference images + ground-truth binary masks.

    1. Mask Conditioning:
       The support masks (indicating target anatomical regions in reference slices)
       are projected into the feature embedding space and combined with support image
       features. This ensures the attention mechanism focuses on target structure features.

    2. Cross-Attention Mechanism:
       - Query (Q): Target image feature tokens [Batch, H*W, Channels]
       - Key (K) & Value (V): Mask-conditioned support set tokens [Batch, S*H*W, Channels]
       The Query features "attend to" relevant support regions to retrieve guidance on
       what pixels belong to the target segmentation structure.

    3. Feature Fusion & Residual Output:
       The attended support representation is combined with original Query features via a
       residual skip connection, followed by Layer Normalization and a Feed-Forward
       Projection Network (FFN) to yield the final fused feature map ready for a decoder.
    """

    def __init__(self, channels: int = 64, num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            channels (int): Number of feature channels (C). Default: 64.
            num_heads (int): Number of attention heads for multi-head cross attention. Default: 4.
            dropout (float): Dropout probability for regularization. Default: 0.1.
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # 1. Mask Projection Layer: Maps 1-channel binary mask into C-channel embedding space
        self.mask_proj = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=1)

        # 2. Multi-Head Cross-Attention Block
        # Query comes from target image; Key & Value come from support set features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 3. Normalization Layers for Residual Connection
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
        Forward Pass of the FusionModule.

        Args:
            query_feat (torch.Tensor): Query image feature map of shape [B, C, H, W].
            support_feats (torch.Tensor): Support set feature maps of shape [B, S, C, H, W].
            support_masks (torch.Tensor): Support set binary masks of shape [B, S, 1, H, W].

        Returns:
            torch.Tensor: Fused feature map ready for segmentation decoder, shape [B, C, H, W].
        """
        B, C, H, W = query_feat.shape
        _, S, _, _, _ = support_feats.shape

        # Step 1: Mask Conditioning on Support Set Features
        # Reshape support masks [B, S, 1, H, W] -> [B*S, 1, H, W] to run 2D Conv
        support_masks_flat = support_masks.view(B * S, 1, H, W)
        mask_embed = self.mask_proj(support_masks_flat)  # [B*S, C, H, W]
        mask_embed = mask_embed.view(B, S, C, H, W)      # [B, S, C, H, W]

        # Condition support features by adding mask embedding and multiplying by mask gate
        # (Focuses attention heavily on foreground target regions while preserving feature context)
        conditioned_support = support_feats * (1.0 + torch.sigmoid(mask_embed))

        # Step 2: Prepare Query, Key, Value Tensors for Multi-Head Attention
        # Query (Q): Flatten target image spatial dims -> [B, H*W, C]
        Q = query_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Key (K) & Value (V): Flatten support set & spatial dims -> [B, S*H*W, C]
        K = conditioned_support.permute(0, 1, 3, 4, 2).reshape(B, S * H * W, C)
        V = K  # Values are the mask-conditioned support tokens

        # Step 3: Multi-Head Cross Attention
        # Query tokens attend to Key/Value tokens from the support set
        attn_out, _ = self.cross_attn(query=Q, key=K, value=V)  # [B, H*W, C]

        # Step 4: First Residual Connection & Layer Normalization
        x = self.norm1(Q + attn_out)  # [B, H*W, C]

        # Step 5: Feed-Forward Refinement & Second Residual Connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)    # [B, H*W, C]

        # Step 6: Reshape back to 2D Feature Map [B, C, H, W]
        fused_feat = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return fused_feat


if __name__ == "__main__":
    print("==================================================")
    print(" Testing FusionModule with Dummy Data Tensors")
    print("==================================================")

    # Define realistic dimensions: Batch size=1, Channels=64, Height=32, Width=32, Support count=2
    B, C, H, W, S = 1, 64, 32, 32, 2

    # Create dummy input tensors
    dummy_query_feat = torch.randn(B, C, H, W)
    dummy_support_feats = torch.randn(B, S, C, H, W)
    dummy_support_masks = torch.randint(0, 2, (B, S, 1, H, W)).float()

    print(f"Input Query Features shape:  {list(dummy_query_feat.shape)}")
    print(f"Input Support Features shape: {list(dummy_support_feats.shape)}")
    print(f"Input Support Masks shape:    {list(dummy_support_masks.shape)}")

    # Instantiate FusionModule
    model = FusionModule(channels=C, num_heads=4, dropout=0.1)
    model.eval()

    # Forward pass
    with torch.no_grad():
        fused_output = model(dummy_query_feat, dummy_support_feats, dummy_support_masks)

    print("--------------------------------------------------")
    print(f"Output Fused Features shape:  {list(fused_output.shape)}")
    print("--------------------------------------------------")

    # Assert shape matches expected output shape [B, C, H, W]
    assert fused_output.shape == (B, C, H, W), f"Shape mismatch! Expected {(B, C, H, W)}, got {fused_output.shape}"
    print("SUCCESS: Forward pass executed cleanly and output shape matches [B, C, H, W]!")
