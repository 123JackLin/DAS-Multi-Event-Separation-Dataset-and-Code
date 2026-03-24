"""
Data module for DAS Multi-Event Separation.

Contains the DASData container class and a synthetic data simulator that
generates realistic mixed DAS records from multiple seismic events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DASData:
    """Container for a DAS record.

    Attributes
    ----------
    data : np.ndarray
        2-D array of shape ``(n_channels, n_samples)`` containing strain-rate
        or velocity waveforms.
    dt : float
        Sample interval in seconds.
    dx : float
        Channel spacing in metres.
    t0 : float
        Start time in seconds (default 0).
    channel_offset : float
        Distance of the first channel from the source reference in metres
        (default 0).
    labels : list of str, optional
        Human-readable label for each event component (used for separation
        results).
    """

    data: np.ndarray
    dt: float
    dx: float
    t0: float = 0.0
    channel_offset: float = 0.0
    labels: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def times(self) -> np.ndarray:
        """Time axis in seconds."""
        return self.t0 + np.arange(self.n_samples) * self.dt

    @property
    def offsets(self) -> np.ndarray:
        """Channel offset axis in metres."""
        return self.channel_offset + np.arange(self.n_channels) * self.dx

    def copy(self) -> "DASData":
        return DASData(
            data=self.data.copy(),
            dt=self.dt,
            dx=self.dx,
            t0=self.t0,
            channel_offset=self.channel_offset,
            labels=list(self.labels),
        )


# ---------------------------------------------------------------------------
# Synthetic data simulator
# ---------------------------------------------------------------------------


def _ricker_wavelet(f: float, dt: float, n_samples: int) -> np.ndarray:
    """Return a Ricker (Mexican hat) wavelet centred at the array midpoint."""
    t = (np.arange(n_samples) - n_samples // 2) * dt
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def simulate_das_data(
    n_channels: int = 64,
    n_samples: int = 512,
    dt: float = 0.002,
    dx: float = 10.0,
    events: Optional[List[dict]] = None,
    noise_std: float = 0.05,
    seed: Optional[int] = 42,
) -> Tuple[DASData, List[DASData]]:
    """Simulate a mixed DAS record containing multiple seismic events.

    Each event is modelled as a linear move-out (constant apparent velocity)
    wavelet whose amplitude decays with offset.

    Parameters
    ----------
    n_channels : int
        Number of DAS channels.
    n_samples : int
        Number of time samples.
    dt : float
        Sample interval in seconds.
    dx : float
        Channel spacing in metres.
    events : list of dict, optional
        Each dict may contain:

        ``velocity`` : float
            Apparent velocity in m/s (default 2000).
        ``t0`` : float
            Arrival time at channel 0 in seconds (default 0.1).
        ``frequency`` : float
            Dominant frequency of the Ricker wavelet in Hz (default 20).
        ``amplitude`` : float
            Peak amplitude (default 1.0).
        ``decay`` : float
            Exponential amplitude decay per channel (default 0.0).

        If *events* is ``None`` two default events are used.
    noise_std : float
        Standard deviation of additive white Gaussian noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    mixed : DASData
        Mixed record (sum of all events + noise).
    components : list of DASData
        Individual (noise-free) event records in the same order as *events*.
    """
    rng = np.random.default_rng(seed)

    if events is None:
        events = [
            {"velocity": 2000.0, "t0": 0.10, "frequency": 15.0, "amplitude": 1.0},
            {"velocity": 3500.0, "t0": 0.05, "frequency": 25.0, "amplitude": 0.7},
        ]

    offsets = np.arange(n_channels) * dx
    wavelet_cache: dict = {}

    components: List[DASData] = []
    mixed_data = np.zeros((n_channels, n_samples))

    for idx, ev in enumerate(events):
        velocity = float(ev.get("velocity", 2000.0))
        t0_ev = float(ev.get("t0", 0.10))
        freq = float(ev.get("frequency", 20.0))
        amplitude = float(ev.get("amplitude", 1.0))
        decay = float(ev.get("decay", 0.0))

        # Cache wavelets by frequency to avoid recomputing
        if freq not in wavelet_cache:
            wavelet_cache[freq] = _ricker_wavelet(freq, dt, n_samples)
        wavelet = wavelet_cache[freq]

        ev_data = np.zeros((n_channels, n_samples))
        for ch, offset in enumerate(offsets):
            delay_samples = int(round((t0_ev + offset / velocity) / dt))
            amp = amplitude * np.exp(-decay * ch)
            shifted = np.zeros(n_samples)
            if delay_samples < n_samples:
                src_start = max(0, -delay_samples)
                dst_start = max(0, delay_samples)
                length = min(n_samples - dst_start, n_samples - src_start)
                shifted[dst_start : dst_start + length] = (
                    wavelet[src_start : src_start + length]
                )
            ev_data[ch] = amp * shifted

        component = DASData(
            data=ev_data,
            dt=dt,
            dx=dx,
            labels=[f"event_{idx}"],
        )
        components.append(component)
        mixed_data += ev_data

    # Add noise
    if noise_std > 0:
        mixed_data += rng.normal(0, noise_std, mixed_data.shape)

    mixed = DASData(
        data=mixed_data,
        dt=dt,
        dx=dx,
        labels=[f"event_{i}" for i in range(len(events))],
    )

    return mixed, components
