import os
import sys
from typing import Tuple
import torch
import torch.nn as nn

# Ensure external/universeg is in sys.path
UNIVERSEG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../external/universeg"))
if UNIVERSEG_PATH not in sys.path:
    sys.path.insert(0, UNIVERSEG_PATH)

from universeg import universeg


def load_universeg_backbone(device: torch.device = None) -> nn.Module:
    """
    Loads pretrained UniverSeg model as a frozen backbone.
    Sets requires_grad = False for all parameters and sets eval mode.
    """
    model = universeg(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def get_frozen_features(images: torch.Tensor, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts intermediate encoder feature maps for a batch of input images via forward hooks.

    Args:
        images (torch.Tensor): Input batch of images [B, C, H, W] or [B, 1, H, W].
        model (nn.Module): Pretrained UniverSeg backbone model.

    Returns:
        Tuple of feature maps:
            f1: Stage 1 feature map [B, 64, H, W]
            f2: Stage 2 feature map [B, 64, H/2, W/2]
            f3: Stage 3 (bottleneck) feature map [B, 64, H/4, W/4]
    """
    device = images.device
    dtype = images.dtype
    B, C, H, W = images.shape

    # UniverSeg expects 1-channel target input [B, 1, H, W]
    if C > 1:
        img_in = images[:, :1, :, :]
    else:
        img_in = images

    # Create dummy support tensors for UniverSeg forward execution (S=1 dummy support)
    dummy_supp_img = torch.zeros((B, 1, 1, H, W), device=device, dtype=dtype)
    dummy_supp_lbl = torch.zeros((B, 1, 1, H, W), device=device, dtype=dtype)

    captured = {}
    handles = []

    def make_hook(name: str):
        def hook(module, input, output):
            # output of CrossBlock is (target, support) tuple
            # target has shape [B, 1, C_out, H_stage, W_stage]
            target_feat = output[0]
            if target_feat.ndim == 5 and target_feat.shape[1] == 1:
                target_feat = target_feat.squeeze(1)
            captured[name] = target_feat
        return hook

    # Register hooks on encoder blocks 0, 1, 2
    for idx in range(3):
        h = model.enc_blocks[idx].register_forward_hook(make_hook(f"enc_{idx}"))
        handles.append(h)

    model.eval()
    with torch.no_grad():
        _ = model(img_in, dummy_supp_img, dummy_supp_lbl)

    # Remove forward hooks
    for h in handles:
        h.remove()

    f1 = captured["enc_0"]  # [B, 64, H, W]
    f2 = captured["enc_1"]  # [B, 64, H/2, W/2]
    f3 = captured["enc_2"]  # [B, 64, H/4, W/4]

    return f1, f2, f3


if __name__ == "__main__":
    print("==================================================")
    print(" Testing Backbone Module (UniverSeg Feature Extractor)")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading backbone on device: {device}")
    backbone = load_universeg_backbone(device=device)

    # Check requires_grad is False
    grad_params = [name for name, p in backbone.named_parameters() if p.requires_grad]
    assert len(grad_params) == 0, f"Found parameters with requires_grad=True: {grad_params}"
    print("Verification Passed: All backbone parameters have requires_grad=False!")

    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"Frozen Backbone Parameter Count: {total_params:,}")

    # Test feature extraction
    B, C, H, W = 2, 1, 128, 128
    dummy_input = torch.randn(B, C, H, W, device=device)
    f1, f2, f3 = get_frozen_features(dummy_input, backbone)

    print(f"Extracted Stage 1 Features (f1): {list(f1.shape)}")
    print(f"Extracted Stage 2 Features (f2): {list(f2.shape)}")
    print(f"Extracted Stage 3 Features (f3): {list(f3.shape)}")

    assert f1.shape == (B, 64, H, W), f"Expected {(B, 64, H, W)}, got {f1.shape}"
    assert f2.shape == (B, 64, H // 2, W // 2), f"Expected {(B, 64, H // 2, W // 2)}, got {f2.shape}"
    assert f3.shape == (B, 64, H // 4, W // 4), f"Expected {(B, 64, H // 4, W // 4)}, got {f3.shape}"

    print("==================================================")
    print("SUCCESS: backbone.py verified successfully!")
    print("==================================================")
