"""
Preprocessing utilities for DAS data.

Includes bandpass filtering, amplitude normalization, and cosine tapering.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .data import DASData


def bandpass_filter(
    das: DASData,
    f_low: float,
    f_high: float,
    order: int = 4,
    zero_phase: bool = True,
) -> DASData:
    """Apply a Butterworth bandpass filter along the time axis.

    Parameters
    ----------
    das : DASData
        Input record.
    f_low : float
        Low corner frequency in Hz.
    f_high : float
        High corner frequency in Hz.
    order : int
        Filter order (default 4).
    zero_phase : bool
        If ``True`` apply the filter forward and backward (zero-phase,
        doubles the effective order).

    Returns
    -------
    DASData
        Filtered record with the same metadata.
    """
    nyq = 0.5 / das.dt
    if f_low <= 0 or f_high >= nyq:
        raise ValueError(
            f"Corner frequencies must satisfy 0 < f_low < f_high < {nyq:.1f} Hz"
        )
    sos = signal.butter(order, [f_low / nyq, f_high / nyq], btype="band", output="sos")
    if zero_phase:
        filtered = signal.sosfiltfilt(sos, das.data, axis=-1)
    else:
        filtered = signal.sosfilt(sos, das.data, axis=-1)

    out = das.copy()
    out.data = filtered
    return out


def normalize(
    das: DASData,
    mode: str = "trace",
    eps: float = 1e-10,
) -> DASData:
    """Normalise amplitudes.

    Parameters
    ----------
    das : DASData
        Input record.
    mode : {'trace', 'global', 'rms'}
        * ``'trace'``  – each trace is divided by its own maximum absolute value.
        * ``'global'`` – all traces are divided by the global maximum absolute value.
        * ``'rms'``    – each trace is normalised to unit RMS amplitude.
    eps : float
        Small constant added to denominators to avoid division by zero.

    Returns
    -------
    DASData
        Normalised record.
    """
    out = das.copy()
    if mode == "trace":
        peak = np.max(np.abs(out.data), axis=-1, keepdims=True)
        out.data = out.data / (peak + eps)
    elif mode == "global":
        peak = np.max(np.abs(out.data))
        out.data = out.data / (peak + eps)
    elif mode == "rms":
        rms = np.sqrt(np.mean(out.data ** 2, axis=-1, keepdims=True))
        out.data = out.data / (rms + eps)
    else:
        raise ValueError(f"Unknown normalization mode: '{mode}'")
    return out


def taper(
    das: DASData,
    taper_fraction: float = 0.05,
    side: str = "both",
) -> DASData:
    """Apply a cosine (Tukey) taper along the time axis.

    Parameters
    ----------
    das : DASData
        Input record.
    taper_fraction : float
        Fraction of the trace length used for tapering on *each* side
        (e.g. 0.05 tapers 5 % of each end).
    side : {'both', 'left', 'right'}
        Which ends of the trace to taper.

    Returns
    -------
    DASData
        Tapered record.
    """
    n = das.n_samples
    window = signal.windows.tukey(n, alpha=2 * taper_fraction)

    if side == "left":
        window[n // 2 :] = 1.0
    elif side == "right":
        window[: n // 2] = 1.0
    elif side != "both":
        raise ValueError(f"Unknown side: '{side}'")

    out = das.copy()
    out.data = out.data * window[np.newaxis, :]
    return out
