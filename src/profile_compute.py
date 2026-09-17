"""
profile_compute.py - Compute, Latency & Parameter Profiling
===========================================================
Profiles the proposed In-Context Cross-Attention Adapter and UniverSeg backbone:
  1. Trainable vs. Frozen vs. Total Parameter count.
  2. Floating Point Operations (GFLOPs / GMACs).
  3. Forward-pass inference latency (ms per 2D slice) via torch.cuda.Event.
  4. Peak GPU VRAM consumption (MB).

Generates a publication-ready systems table for the research paper.
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    from src.models.in_context_model import InContextSegmentationModel
except ImportError as e:
    print(f"[WARN] Local imports: {e}")


def profile_model(num_layers: int = 3, device_str: str = "cuda", output_file: str = "logs/compute_profile.txt"):
    """Profiles model latency, memory, and parameters."""
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"[INFO] Profiling device: {device}")

    model = InContextSegmentationModel(num_layers=num_layers).to(device)
    model.eval()

    # 1. Parameter Breakdown
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    pct_trainable = (trainable_params / total_params) * 100.0

    print("\n--- PARAMETER BREAKDOWN ---")
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,} ({pct_trainable:.2f}%)")
    print(f"Frozen Parameters:     {frozen_params:,} ({100 - pct_trainable:.2f}%)")

    # Synthetic input tensors: Query [1, 1, 128, 128], Support Imgs [1, 1, 1, 128, 128], Support Masks [1, 1, 1, 128, 128]
    query = torch.randn(1, 1, 128, 128, device=device)
    sup_img = torch.randn(1, 1, 1, 128, 128, device=device)
    sup_mask = (torch.rand(1, 1, 1, 128, 128, device=device) > 0.5).float()

    # 2. FLOPs / GMACs via thop (if installed) or manual count
    gmacs_str = "N/A"
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(query, sup_img, sup_mask), verbose=False)
        gmacs = macs / 1e9
        gmacs_str = f"{gmacs:.2f} GMACs"
        print(f"Computational Complexity: {gmacs_str}")
    except ImportError:
        print("[INFO] 'thop' package not installed. Skipping direct GMACs measurement.")

    # 3. Peak VRAM
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Warm-up forward passes
    print("\nRunning warm-up passes...")
    for _ in range(20):
        with torch.no_grad():
            _ = model(query, sup_img, sup_mask)

    # 4. Latency measurement (100 iterations)
    num_runs = 100
    latencies = []

    print(f"Benchmarking forward-pass latency over {num_runs} iterations...")
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(num_runs):
            starter.record()
            with torch.no_grad():
                _ = model(query, sup_img, sup_mask)
            ender.record()
            torch.cuda.synchronize()
            latencies.append(starter.elapsed_time(ender))  # milliseconds
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        for _ in range(num_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(query, sup_img, sup_mask)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
        peak_vram_mb = 0.0

    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print(f"Latency per slice:     {mean_lat:.2f} +/- {std_lat:.2f} ms")
    print(f"Throughput:            {fps:.1f} slices/second")
    print(f"Peak GPU VRAM:         {peak_vram_mb:.1f} MB")

    # 5. Save report
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write("================================================================================\n")
        f.write("             COMPUTATIONAL LATENCY & SYSTEM RESOURCE PROFILE\n")
        f.write("================================================================================\n\n")
        f.write(f"Device:                {device}\n")
        f.write(f"Total Parameters:      {total_params:,}\n")
        f.write(f"Trainable Parameters:  {trainable_params:,} ({pct_trainable:.2f}% of total)\n")
        f.write(f"Frozen Backbone:       {frozen_params:,} ({100 - pct_trainable:.2f}% of total)\n")
        f.write(f"GMACs / FLOPs:         {gmacs_str}\n")
        f.write(f"Inference Latency:     {mean_lat:.2f} +/- {std_lat:.2f} ms per 2D slice\n")
        f.write(f"Inference Throughput:  {fps:.1f} slices / second\n")
        f.write(f"Peak GPU VRAM:         {peak_vram_mb:.1f} MB\n")
        f.write("================================================================================\n")

    print(f"\n[DONE] Saved compute profile to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile compute latency and parameters")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="logs/compute_profile.txt")
    args = parser.parse_args()

    profile_model(args.layers, args.device, args.output)
