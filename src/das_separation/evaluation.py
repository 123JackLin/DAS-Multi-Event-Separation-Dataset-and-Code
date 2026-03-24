"""
Evaluation metrics for DAS separation quality.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .data import DASData


def snr(estimated: DASData, reference: DASData, eps: float = 1e-10) -> float:
    """Signal-to-Noise Ratio between an estimated and reference DAS record.

    .. math::

        \\mathrm{SNR} = 10 \\log_{10}\\!\\left(
            \\frac{\\|\\mathbf{x}_{\\text{ref}}\\|^2}
                 {\\|\\mathbf{x}_{\\text{ref}} - \\mathbf{x}_{\\text{est}}\\|^2}
        \\right)

    Parameters
    ----------
    estimated : DASData
        Estimated (separated) record.
    reference : DASData
        Ground-truth reference record.
    eps : float
        Small regularisation constant.

    Returns
    -------
    float
        SNR in dB.
    """
    signal_power = np.sum(reference.data ** 2)
    noise_power = np.sum((reference.data - estimated.data) ** 2)
    return float(10 * np.log10((signal_power + eps) / (noise_power + eps)))


def correlation_coefficient(estimated: DASData, reference: DASData) -> float:
    """Pearson correlation coefficient between estimated and reference data.

    Parameters
    ----------
    estimated : DASData
        Estimated record.
    reference : DASData
        Ground-truth record.

    Returns
    -------
    float
        Correlation coefficient in ``[-1, 1]``.
    """
    x = estimated.data.ravel()
    y = reference.data.ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def evaluate_separation(
    estimated_list: List[DASData],
    reference_list: List[DASData],
) -> List[Dict[str, float]]:
    """Evaluate separation quality for each event component.

    Each estimated component is matched to the reference component with the
    highest correlation (a greedy best-match assignment).

    Parameters
    ----------
    estimated_list : list of DASData
        Separated components produced by an algorithm.
    reference_list : list of DASData
        Ground-truth event components (noise-free).

    Returns
    -------
    list of dict
        One dict per reference component with keys ``'snr_db'`` and
        ``'correlation'``, as well as ``'matched_index'`` (which estimated
        component was matched).
    """
    n_ref = len(reference_list)
    n_est = len(estimated_list)

    # Build correlation matrix
    corr_matrix = np.zeros((n_ref, n_est))
    for i, ref in enumerate(reference_list):
        for j, est in enumerate(estimated_list):
            corr_matrix[i, j] = abs(correlation_coefficient(est, ref))

    used_est = set()
    results = []
    for i in range(n_ref):
        # Find the best unmatched estimated component
        order = np.argsort(corr_matrix[i])[::-1]
        matched_j = None
        for j in order:
            if j not in used_est:
                matched_j = int(j)
                used_est.add(matched_j)
                break
        if matched_j is None:
            results.append({"snr_db": float("-inf"), "correlation": 0.0, "matched_index": -1})
        else:
            results.append(
                {
                    "snr_db": snr(estimated_list[matched_j], reference_list[i]),
                    "correlation": float(corr_matrix[i, matched_j]),
                    "matched_index": matched_j,
                }
            )
    return results
