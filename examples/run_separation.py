#!/usr/bin/env python
"""
DAS Multi-Event Separation – end-to-end example.

This script:
  1. Simulates a synthetic DAS gather with two overlapping seismic events.
  2. Applies bandpass filtering and tapering.
  3. Separates the events using three different algorithms (FK, NMF, Sparse).
  4. Evaluates quality with SNR and correlation.
  5. Saves a figure summarising the results.

Usage
-----
    python examples/run_separation.py

The output figure is saved as ``das_separation_demo.png`` in the current
working directory.
"""

import numpy as np

from das_separation import (
    simulate_das_data,
    bandpass_filter,
    taper,
    DASEventSeparator,
    evaluate_separation,
)

# ---------------------------------------------------------------------------
# 1. Simulate data
# ---------------------------------------------------------------------------

print("Simulating DAS data …")
events = [
    {"velocity": 2000.0, "t0": 0.10, "frequency": 15.0, "amplitude": 1.0},
    {"velocity": 3800.0, "t0": 0.06, "frequency": 28.0, "amplitude": 0.8},
]

mixed, ground_truth = simulate_das_data(
    n_channels=64,
    n_samples=512,
    dt=0.002,
    dx=10.0,
    events=events,
    noise_std=0.05,
    seed=42,
)
print(f"  Mixed record shape : {mixed.data.shape}  (channels × samples)")

# ---------------------------------------------------------------------------
# 2. Pre-process
# ---------------------------------------------------------------------------

print("Preprocessing …")
nyquist = 0.5 / mixed.dt
mixed_pre = bandpass_filter(mixed, f_low=5.0, f_high=min(45.0, nyquist * 0.9))
mixed_pre = taper(mixed_pre, taper_fraction=0.05)

# ---------------------------------------------------------------------------
# 3. Separate with three methods
# ---------------------------------------------------------------------------

results: dict = {}

print("Running FK separation …")
sep_fk = DASEventSeparator(
    method="fk",
    n_components=2,
    velocity_ranges=[(800.0, 2800.0), (2800.0, 5000.0)],
    return_all=False,
)
results["FK"] = sep_fk.separate(mixed_pre)

print("Running NMF separation …")
sep_nmf = DASEventSeparator(method="nmf", n_components=2, max_iter=400, random_state=0)
results["NMF"] = sep_nmf.separate(mixed_pre)

print("Running Sparse separation …")
sep_sparse = DASEventSeparator(
    method="sparse",
    n_components=2,
    freq_hints=[15.0, 28.0],
    n_iterations=150,
    threshold=0.05,
)
results["Sparse"] = sep_sparse.separate(mixed_pre)

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------

print("\nSeparation quality (SNR / correlation vs. noise-free ground truth):\n")
header = f"{'Method':<10} {'Event':<8} {'SNR (dB)':>10} {'Correlation':>14}"
print(header)
print("-" * len(header))

for method_name, separated in results.items():
    metrics = evaluate_separation(separated, ground_truth)
    for evt_idx, m in enumerate(metrics):
        print(
            f"{method_name:<10} {evt_idx:<8} {m['snr_db']:>10.2f} {m['correlation']:>14.4f}"
        )

# ---------------------------------------------------------------------------
# 5. Save figure
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")  # headless rendering
    from das_separation import plot_separation_result

    print("\nSaving figure …")
    fig = plot_separation_result(
        mixed=mixed_pre,
        separated=results["NMF"],
        references=ground_truth,
        figsize=(14, 5),
    )
    fig.savefig("das_separation_demo.png", dpi=120)
    print("  Saved: das_separation_demo.png")
except ImportError:
    print("  (matplotlib not available – skipping figure)")

print("\nDone.")
