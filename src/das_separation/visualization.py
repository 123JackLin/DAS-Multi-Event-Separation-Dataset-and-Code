"""
Visualization utilities for DAS data and separation results.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .data import DASData

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def _check_matplotlib() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install it with: pip install matplotlib"
        )


def plot_gather(
    das: DASData,
    title: str = "DAS Gather",
    clip: float = 0.95,
    cmap: str = "seismic",
    ax: Optional["plt.Axes"] = None,
    figsize: tuple = (8, 6),
) -> "Figure":
    """Plot a DAS record as a wiggle-image gather.

    Parameters
    ----------
    das : DASData
        Record to plot.
    title : str
        Figure title.
    clip : float
        Percentile used to clip the color scale (0 < clip <= 1).
    cmap : str
        Matplotlib colormap name.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  A new figure is created if ``None``.
    figsize : tuple
        ``(width, height)`` in inches (used only if *ax* is ``None``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    vmax = np.percentile(np.abs(das.data), clip * 100)
    ax.imshow(
        das.data.T,
        aspect="auto",
        origin="lower",
        extent=[
            das.offsets[0],
            das.offsets[-1],
            das.times[0],
            das.times[-1],
        ],
        vmin=-vmax,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",
    )
    ax.set_xlabel("Channel offset (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_separation_result(
    mixed: DASData,
    separated: List[DASData],
    references: Optional[List[DASData]] = None,
    clip: float = 0.95,
    cmap: str = "seismic",
    figsize: Optional[tuple] = None,
) -> "Figure":
    """Plot the mixed record alongside separated (and optionally reference) components.

    Parameters
    ----------
    mixed : DASData
        The observed mixed record.
    separated : list of DASData
        Separated event components.
    references : list of DASData, optional
        Ground-truth event components to compare against.
    clip : float
        Color clip percentile.
    cmap : str
        Matplotlib colormap.
    figsize : tuple, optional
        Figure size.  Auto-computed if ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_matplotlib()

    n_sep = len(separated)
    n_ref = len(references) if references is not None else 0
    n_cols = 1 + n_sep + n_ref
    if figsize is None:
        figsize = (4 * n_cols, 5)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=True)
    if n_cols == 1:
        axes = [axes]

    col = 0
    plot_gather(mixed, title="Mixed", clip=clip, cmap=cmap, ax=axes[col])
    col += 1

    for i, comp in enumerate(separated):
        label = comp.labels[0] if comp.labels else f"Separated {i}"
        plot_gather(comp, title=f"Sep: {label}", clip=clip, cmap=cmap, ax=axes[col])
        col += 1

    if references is not None:
        for i, ref in enumerate(references):
            label = ref.labels[0] if ref.labels else f"Reference {i}"
            plot_gather(ref, title=f"Ref: {label}", clip=clip, cmap=cmap, ax=axes[col])
            col += 1

    fig.tight_layout()
    return fig
