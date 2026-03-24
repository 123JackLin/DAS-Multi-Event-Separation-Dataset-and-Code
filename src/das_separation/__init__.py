"""
DAS Multi-Event Separation package.

Provides tools to simulate, preprocess, separate and evaluate
overlapping seismic events recorded by Distributed Acoustic Sensing (DAS) arrays.
"""

from .data import DASData, simulate_das_data
from .preprocessing import bandpass_filter, normalize, taper
from .separation import (
    fk_filter,
    nmf_separation,
    sparse_separation,
    DASEventSeparator,
)
from .evaluation import snr, correlation_coefficient, evaluate_separation
from .visualization import plot_gather, plot_separation_result

__all__ = [
    "DASData",
    "simulate_das_data",
    "bandpass_filter",
    "normalize",
    "taper",
    "fk_filter",
    "nmf_separation",
    "sparse_separation",
    "DASEventSeparator",
    "snr",
    "correlation_coefficient",
    "evaluate_separation",
    "plot_gather",
    "plot_separation_result",
]
