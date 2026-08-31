# Model Architecture Specification & Methods Documentation

This document provides a detailed technical specification of the **In-Context Medical Image Segmentation Model with Stacked Multi-Layer Cross-Attention Fusion**. This specification serves as the Methods section skeleton for project reports and paper submissions.

---

## 1. High-Level Architecture Overview

The model employs a hybrid **Frozen Backbone + Trainable In-Context Fusion + UNet-Style Decoder** architecture designed for few-shot medical image segmentation across arbitrary modalities (CT, MRI).

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
           │                            │                  │ (N=3 Blocks)     │
           │                            │                  └─────────┬────────┘
           │                            │                            │ Fused Bottleneck [B, 64, H/4, W/4]
           │                            │                            ▼
           │                            └──────────────────► ┌──────────────────┐
           │                                                 │ Decoder Stage 2  │ (ConvTranspose + Skip + ConvBlock)
           │                                                 └─────────┬────────┘
           │                                                           │ Up2 Output [B, 64, H/2, W/2]
           │                                                           ▼
           └───────────────────────────────────────────────► ┌──────────────────┐
                                                             │ Decoder Stage 1  │ (ConvTranspose + Skip + ConvBlock)
                                                             └─────────┬────────┘
                                                                       │ Up1 Output [B, 64, H, W]
                                                                       ▼
                                                             ┌──────────────────┐
                                                             │ 1x1 Conv Head    │
                                                             └─────────┬────────┘
                                                                       ▼
                                                           Logits: [B, 1, H, W]
```

---

## 2. Stacked Multi-Layer Fusion Module (`FusionModule`)

The core contribution is the **Stacked Multi-Layer Cross-Attention Fusion Module**. Instead of a shallow single-layer cross-attention mechanism, the module stacks \(N=3\) identical `FusionBlock` layers chained sequentially like Transformer encoder/decoder layers.

### Block Diagram of Layer \(N\) (`FusionBlock`):
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

### Layer Chaining Mechanism:
- **Input to Block 1**: Query bottleneck feature \(x_0 = q\_f3\) extracted from the target image.
- **Input to Block \(N\)**: Refined feature map \(x_{N-1}\) output from Block \(N-1\).
- **Key & Value Tokens**: Shared mask-conditioned support tokens derived from reference images and masks:
  \[
  \text{Conditioned Support} = \mathbf{F}_{\text{supp}} \odot \left(1 + \sigma(\text{Conv}_{1\times1}(\mathbf{M}_{\text{supp}}))\right)
  \]
- **Output**: Final fused bottleneck tensor \(x_N \in \mathbb{R}^{B \times C \times H/4 \times W/4}\).

---

## 3. Data Shapes & Tensor Dimensions Across Pipeline

Assuming standard input batch size \(B\), support set count \(S=2\), target resolution \(H \times W = 128 \times 128\), and feature channels \(C=64\):

| Stage / Layer | Tensor Name | Shape | Trainable? |
|---|---|---|---|
| Input Target Image | `query_img` | `[B, C_in, 128, 128]` | No |
| Input Support Images | `support_imgs` | `[B, S, C_in, 128, 128]` | No |
| Input Support Masks | `support_masks` | `[B, S, 1, 128, 128]` | No |
| Backbone Stage 1 Features | `q_f1` | `[B, 64, 128, 128]` | No (Frozen) |
| Backbone Stage 2 Features | `q_f2` | `[B, 64, 64, 64]` | No (Frozen) |
| Backbone Bottleneck | `q_f3` | `[B, 64, 32, 32]` | No (Frozen) |
| Downsampled Support Masks | `s_masks_down` | `[B, S, 1, 32, 32]` | No |
| Fusion Layer Input Tokens | `Q` | `[B, 1024, 64]` | Yes |
| Support Key/Value Tokens | `K`, `V` | `[B, 2048, 64]` | Yes |
| Stacked Fusion Output | `fused_bottleneck` | `[B, 64, 32, 32]` | Yes |
| Decoder Stage 2 Output | `dec2_out` | `[B, 64, 64, 64]` | Yes |
| Decoder Stage 1 Output | `dec1_out` | `[B, 64, 128, 128]` | Yes |
| Final Output Logits | `logits` | `[B, 1, 128, 128]` | Yes |

---

## 4. Parameter Count Breakdown (`num_layers=3`)

```
==================================================
 IN-CONTEXT MODEL PARAMETER COUNT BREAKDOWN
==================================================
 Frozen Backbone (UniverSeg): 1,182,785 (76.89%)
 Trainable Fusion Module:      100,800
 Trainable Decoder:            254,657
 Total Trainable Params:       355,457 (23.11%)
 Total Model Parameters:     1,538,242
==================================================
```

Each `FusionBlock` layer contributes **33,600** trainable parameters:
- `mask_proj` (Conv2d 1->64): 128 params
- `cross_attn` (MultiheadAttention 64->64, 4 heads): 16,640 params
- `norm1` + `norm2` (LayerNorms): 256 params
- `ffn` (Linear 64->128 + Linear 128->64): 16,576 params

---

## 5. Depth Ablation Study & Empirical Rationale for \(N=3\)

To evaluate whether stacking multiple cross-attention layers improves in-context segmentation performance or hits diminishing returns, we conducted a controlled 5-epoch depth ablation study on local Task09 Spleen CT data (fixed random seed = 42, identical learning rate and optimizer settings):

| Fusion Depth (`num_layers`) | Trainable Params | Total Params | Best Val Dice | Test Loss | Test Dice |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1 Block (Baseline)** | 288,257 (19.60%) | 1,471,042 | 0.7523 | 0.2435 | **0.7489** |
| **3 Blocks (Proposed)** | 355,457 (23.11%) | 1,538,242 | **0.7770** | **0.2113** | **0.7765** |
| **5 Blocks** | 422,657 (26.34%) | 1,605,442 | 0.7766 | 0.2173 | **0.7758** |

### Key Findings & Architectural Justification:
1. **Significant Gain over Single-Layer**: Increasing depth from 1 to 3 blocks provides a notable **+2.76% boost in Test Dice** (0.7489 \(\rightarrow\) 0.7765) and reduces Test Loss from 0.2435 to 0.2113.
2. **Diminishing Returns Beyond 3 Layers**: Extending depth further from 3 to 5 blocks results in a performance plateau (Test Dice 0.7765 vs 0.7758) while adding 67,200 additional parameters.
3. **Optimal Architectural Choice**: Setting `num_layers = 3` provides the optimal trade-off between representational capacity, training speed, parameter efficiency (only 23.1% of total model params are trainable), and generalization capability.
