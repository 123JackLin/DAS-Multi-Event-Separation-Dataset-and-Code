# Philo-flow DAS Multi-Event Separation

**DAS Multi-Event Separation** – a Python toolkit for separating overlapping
seismic events recorded by Distributed Acoustic Sensing (DAS) arrays.

---

## Overview

Distributed Acoustic Sensing (DAS) converts a standard fibre-optic cable into
thousands of virtual seismic sensors at metre-scale spacing.  In active seismic
surveys and passive monitoring applications, multiple events often overlap in
time and space, making it difficult to analyse them individually.

This package provides:

| Module | Contents |
|--------|----------|
| `das_separation.data` | `DASData` container + synthetic data simulator |
| `das_separation.preprocessing` | Bandpass filter, normalisation, cosine taper |
| `das_separation.separation` | FK filter, NMF, sparse decomposition, `DASEventSeparator` |
| `das_separation.evaluation` | SNR, Pearson correlation, multi-event scoring |
| `das_separation.visualization` | Gather plot, side-by-side separation figure |

---

## Installation

```bash
pip install -e ".[dev]"
```

**Dependencies:** `numpy`, `scipy`, `scikit-learn`, `matplotlib`

---

## Quick start

```python
from das_separation import (
    simulate_das_data,
    bandpass_filter,
    DASEventSeparator,
    evaluate_separation,
    plot_separation_result,
)

# 1. Simulate a mixed DAS record (2 overlapping events)
mixed, ground_truth = simulate_das_data(n_channels=64, n_samples=512, seed=42)

# 2. Pre-process
mixed = bandpass_filter(mixed, f_low=5.0, f_high=45.0)

# 3. Separate with NMF
sep = DASEventSeparator(method="nmf", n_components=2)
separated = sep.separate(mixed)

# 4. Evaluate quality
metrics = evaluate_separation(separated, ground_truth)
for i, m in enumerate(metrics):
    print(f"Event {i}: SNR={m['snr_db']:.1f} dB  CC={m['correlation']:.3f}")

# 5. Visualise
fig = plot_separation_result(mixed, separated, references=ground_truth)
fig.savefig("result.png", dpi=120)
```

---

## Separation methods

### FK filtering (`method='fk'`)
Classical frequency-wavenumber masking.  Events are separated by their
apparent velocities.  Pass `velocity_ranges=[(v_min1, v_max1), ...]`.

### NMF (`method='nmf'`)
Non-negative Matrix Factorisation of the FK magnitude spectrum.  Works well
when events have different spatial frequency content.

### Sparse decomposition (`method='sparse'`)
Channel-wise matching pursuit in a Ricker wavelet dictionary.  Atoms are
grouped by dominant frequency to reconstruct individual events.

---

## Running the example

```bash
python examples/run_separation.py
```

This prints a quality table and saves `das_separation_demo.png`.

---

## Running tests

```bash
pytest tests/ -v
```

---

## Repository structure

```
src/das_separation/     # Python package
  __init__.py
  data.py               # DASData, simulate_das_data
  preprocessing.py      # bandpass_filter, normalize, taper
  separation.py         # fk_filter, nmf_separation, sparse_separation, DASEventSeparator
  evaluation.py         # snr, correlation_coefficient, evaluate_separation
  visualization.py      # plot_gather, plot_separation_result
tests/
  test_das_separation.py
examples/
  run_separation.py
```
