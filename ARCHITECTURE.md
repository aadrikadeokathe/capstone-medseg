# In-Context Medical Image Segmentation Architecture

This document provides the complete architecture specification, text-based block diagrams, tensor shape tracking, parameter breakdown across fusion module depths, design rationale, and limitations for the **In-Context Medical Image Segmentation Model with Stacked Multi-Layer Cross-Attention Fusion**.

---

## 1. High-Level Pipeline Architecture Diagram

The overall network combines a **Frozen Pretrained Backbone (UniverSeg)**, a **Trainable Stacked Multi-Layer Fusion Module**, and a **Trainable UNet-Style Skip Decoder**.

```
                       [ Target Query Image (B, C_in, H, W) ]
                                         │
                             (Frozen UniverSeg Backbone)
                                         │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
     Stage 1 Features             Stage 2 Features            Bottleneck Features
     q_f1: [B, 64, H, W]       q_f2: [B, 64, H/2, W/2]       q_f3: [B, 64, H/4, W/4]
           │                            │                            │
           │                            │                            ▼
           │                            │                  ┌──────────────────┐
           │                            │                  │ Stacked Fusion   │ ◄── Support Imgs & Masks
           │                            │                  │ (N = 1, 3, or 5) │     [B, S, C_in, H, W], [B, S, 1, H, W]
           │                            │                  └─────────┬────────┘
           │                            │                            │ Fused Bottleneck [B, 64, H/4, W/4]
           │                            │                            ▼
           │                            └──────────────────► ┌──────────────────┐
           │                                                 │ Decoder Stage 2  │ (ConvTranspose + Skip + ConvBlock)
           │                                                 └─────────┬────────┘
           │                                                           │ Up2 Output [B, 64, H/2, W/2]
           │                                                           ▼
           │                                                 ┌──────────────────┐
           └───────────────────────────────────────────────► │ Decoder Stage 1  │ (ConvTranspose + Skip + ConvBlock)
                                                             └─────────┬────────┘
                                                                       │ Up1 Output [B, 64, H, W]
                                                                       ▼
                                                             ┌──────────────────┐
                                                             │ 1x1 Conv Head    │
                                                             └─────────┬────────┘
                                                                       │
                                                                       ▼
                                                           Mask Logits: [B, 1, H, W]
```

### Stacked Fusion Module (Layer $N$ `FusionBlock`):
```
           Query Feature Input x_N-1 [B, C, H/4, W/4]          Support Set Features & Masks
                                │                                 [B, S, C, H/4, W/4], [B, S, 1, H/4, W/4]
                                │                                                  │
                                │                                         (Mask Projection 1x1 Conv)
                                │                                                  │
                                │                                         Conditioned Support Keys/Values
                                │                                                  │
                                ├──────────────────────┐                           │
                                │                      ▼                           │
                                │           Multi-Head Cross-Attention ◄───────────┘
                                │                      │
                                └───► (+) ◄────────────┘
                                       │
                                (LayerNorm 1)
                                       │
                                       ├──────────────────────┐
                                       │                      ▼
                                       │             Feed-Forward Network (FFN)
                                       │                      │
                                       └───► (+) ◄────────────┘
                                              │
                                      (LayerNorm 2)
                                              │
                                              ▼
                                Query Feature Output x_N [B, C, H/4, W/4]
```

---

## 2. Input/Output Shapes Across Every Pipeline Stage

Assuming input batch size $B$, support set count $S=2$, spatial resolution $H \times W = 128 \times 128$, input channels $C_{in}=1$ (or $4$ for multi-sequence MRI), and bottleneck feature channels $C=64$:

| Stage / Layer | Tensor Name | Input Shape | Output Shape | Trainable? |
|---|---|---|---|---|
| Target Input Image | `query_img` | - | `[B, C_in, 128, 128]` | No |
| Support Set Images | `support_imgs` | - | `[B, S, C_in, 128, 128]` | No |
| Support Set Masks | `support_masks` | - | `[B, S, 1, 128, 128]` | No |
| Backbone Stage 1 Features | `q_f1` | `[B, C_in, 128, 128]` | `[B, 64, 128, 128]` | No (Frozen) |
| Backbone Stage 2 Features | `q_f2` | `[B, C_in, 128, 128]` | `[B, 64, 64, 64]` | No (Frozen) |
| Backbone Bottleneck | `q_f3` | `[B, C_in, 128, 128]` | `[B, 64, 32, 32]` | No (Frozen) |
| Downsampled Support Masks | `s_masks_down` | `[B, S, 1, 128, 128]` | `[B, S, 1, 32, 32]` | No |
| Support Mask Projection | `mask_embed` | `[B, S, 1, 32, 32]` | `[B, S, 64, 32, 32]` | Yes |
| Conditioned Support Features | `conditioned_support` | `[B, S, 64, 32, 32]` | `[B, S, 64, 32, 32]` | Yes |
| Query Tokens ($Q$) | `Q` | `[B, 64, 32, 32]` | `[B, 1024, 64]` | Yes |
| Key / Value Tokens ($K, V$) | `K`, `V` | `[B, S, 64, 32, 32]` | `[B, 2048, 64]` | Yes |
| Cross-Attention Output | `attn_out` | $Q, K, V$ | `[B, 1024, 64]` | Yes |
| Fusion Block Output | `fused_feat` | `[B, 1024, 64]` | `[B, 64, 32, 32]` | Yes |
| Stacked FusionModule ($N$ layers) | `fused_bottleneck` | `[B, 64, 32, 32]` | `[B, 64, 32, 32]` | Yes |
| Decoder Stage 2 Up | `up2` | `[B, 64, 32, 32]` | `[B, 64, 64, 64]` | Yes |
| Decoder Stage 2 Cat + Conv | `dec2_out` | `[B, 128, 64, 64]` | `[B, 64, 64, 64]` | Yes |
| Decoder Stage 1 Up | `up1` | `[B, 64, 64, 64]` | `[B, 64, 128, 128]` | Yes |
| Decoder Stage 1 Cat + Conv | `dec1_out` | `[B, 128, 128, 128]` | `[B, 64, 128, 128]` | Yes |
| Final Segmentation Head | `logits` | `[B, 64, 128, 128]` | `[B, 1, 128, 128]` | Yes |

---

## 3. Exact Parameter Count Breakdown across Fusion Module Depths

The model parameters are split between the **Frozen Backbone (UniverSeg)** (which remains fixed at 1,182,785 parameters regardless of depth) and the **Trainable Components** (FusionModule + Decoder):

| Fusion Depth (`num_layers`) | Frozen Backbone Params | Trainable Fusion Params | Trainable Decoder Params | Total Trainable Params (%) | Total Model Parameters | Best Val Dice | Test Loss | Test Dice |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1 Layer** | 1,182,785 | 33,600 | 254,657 | **288,257 (19.60%)** | **1,471,042** | 0.7690 | 0.3125 | **0.7496** |
| **3 Layers** | 1,182,785 | 100,800 | 254,657 | **355,457 (23.11%)** | **1,538,242** | **0.7792** | **0.2594** | **0.7765** |
| **5 Layers** | 1,182,785 | 168,000 | 254,657 | **422,657 (26.33%)** | **1,605,442** | 0.7766 | 0.2612 | **0.7758** |

### Per-Layer Parameter Breakdown of `FusionBlock`:
- `mask_proj` ($1 \rightarrow 64$ 1x1 Conv): $1 \times 64 \times 1 \times 1 + 64 = \mathbf{128}$ params
- `cross_attn` (MultiheadAttention, $C=64, \text{heads}=4$): $4 \times (64 \times 64 + 64) = \mathbf{16,640}$ params
- `norm1` + `norm2` (LayerNorms, $C=64$): $2 \times (64 + 64) = \mathbf{256}$ params
- `ffn` (Linear $64 \rightarrow 128$ + Linear $128 \rightarrow 64$): $(64 \times 128 + 128) + (128 \times 64 + 64) = \mathbf{16,576}$ params
- **Total per `FusionBlock`**: $\mathbf{33,600}$ params

---

## 4. Empirical Rationale for Architectural Design Choices

The core design choices of this architecture stem from three key technical requirements: representation transferability, adaptive feature interaction, and spatial target conditioning. First, a **frozen backbone** (pretrained UniverSeg) was chosen for computational feasibility and to leverage deep, domain-general visual features without destroying pretrained knowledge or suffering from catastrophic forgetting during few-shot adaptation. Second, **cross-attention instead of concatenation** was implemented in the fusion module because cross-attention dynamically computes query-support affinity maps, enabling the target query to selectively attend to relevant anatomical regions in the support set regardless of global spatial alignment or minor pose variations, whereas naive concatenation enforces rigid spatial correspondence. Third, **mask-gating** was applied to the support features prior to key-value projection so that ground-truth target masks act as spatial attention gates, concentrating model focus exclusively on foreground target organs while suppressing irrelevant background visual noise.

---

## 5. Architectural Limitations

Despite its strong few-shot performance, the proposed architecture has three main limitations. First, the model currently operates on **single-channel 2D inputs**, requiring 3D volumetric scans or multi-sequence MRI modalities (e.g., T1, T2, FLAIR) to be evaluated either sequence-by-sequence or channel-by-channel rather than end-to-end natively. Second, **variable support size handling** requires flattening all support set slices into a single key-value token dimension ($S \times H \times W$); while functional for small support sets ($S=2, 4$), memory consumption scales linearly with $S$, placing bounds on large-scale support sets. Third, **scaling with spatial resolution** is constrained by the quadratic memory complexity of standard Multi-Head Cross-Attention ($O(N^2)$); while operating at the 1/4 bottleneck resolution ($32 \times 32$) makes memory manageable for $128 \times 128$ inputs, scaling up to full $512 \times 512$ resolutions would necessitate windowed or linear attention mechanisms to prevent out-of-memory errors on standard GPUs.
