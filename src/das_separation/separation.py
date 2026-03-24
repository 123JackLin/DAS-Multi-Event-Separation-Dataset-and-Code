"""
Separation algorithms for DAS Multi-Event Separation.

Three complementary approaches are implemented:

1. **FK filtering** – classical frequency-wavenumber masking to isolate events
   with different apparent velocities.
2. **NMF separation** – non-negative matrix factorisation of the FK spectrum
   magnitude.
3. **Sparse separation** – iterative sparse decomposition in a wavelet
   dictionary (matching pursuit style).

A high-level :class:`DASEventSeparator` class ties them together and selects
the most appropriate algorithm based on the available information.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.decomposition import NMF

from .data import DASData


# ---------------------------------------------------------------------------
# 1. FK filtering
# ---------------------------------------------------------------------------


def _fk_spectrum(data: np.ndarray, dt: float, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the 2-D FK spectrum and its frequency / wavenumber axes."""
    n_channels, n_samples = data.shape
    fk = np.fft.fftshift(np.fft.fft2(data))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_samples, d=dt))
    wavenums = np.fft.fftshift(np.fft.fftfreq(n_channels, d=dx))
    return fk, freqs, wavenums


def fk_filter(
    das: DASData,
    velocity_ranges: List[Tuple[float, float]],
    taper_width: float = 0.05,
    return_all: bool = True,
) -> List[DASData]:
    """Separate events by FK velocity filtering.

    Parameters
    ----------
    das : DASData
        Mixed input record.
    velocity_ranges : list of (v_min, v_max) tuples
        Apparent velocity pass-bands (in m/s) for each target event.
        Use positive values for forward-propagating and negative for
        backward-propagating events.  Sign is preserved.
    taper_width : float
        Fractional width of the cosine taper applied to mask edges
        (relative to the full wavenumber range).
    return_all : bool
        If ``True``, also return the residual (unfiltered energy not captured
        by any mask) as the last element.

    Returns
    -------
    list of DASData
        One separated record per velocity range, plus an optional residual.
    """
    data = das.data
    n_channels, n_samples = data.shape
    fk, freqs, wavenums = _fk_spectrum(data, das.dt, das.dx)

    # fk has shape (n_channels, n_samples): axis-0 is spatial (wavenumber),
    # axis-1 is temporal (frequency).
    KK, FF = np.meshgrid(wavenums, freqs, indexing="ij")
    # Apparent velocity at each (f, k) point; guard against k=0
    with np.errstate(divide="ignore", invalid="ignore"):
        v_apparent = np.where(KK != 0, FF / KK, np.inf)

    cumulative_mask = np.zeros(fk.shape, dtype=float)
    separated = []

    for v_min, v_max in velocity_ranges:
        if v_min > v_max:
            v_min, v_max = v_max, v_min
        mask = ((v_apparent >= v_min) & (v_apparent <= v_max)).astype(float)

        # Smooth the mask edges with a cosine taper
        if taper_width > 0:
            from scipy.ndimage import uniform_filter
            sigma = max(1, int(taper_width * min(n_samples, n_channels)))
            mask = uniform_filter(mask, size=sigma)
            mask = np.clip(mask, 0.0, 1.0)

        filtered_fk = fk * mask
        cumulative_mask = np.maximum(cumulative_mask, mask)

        recovered = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fk)))
        out = das.copy()
        out.data = recovered
        separated.append(out)

    if return_all:
        residual_mask = 1.0 - cumulative_mask
        residual_fk = fk * residual_mask
        residual_data = np.real(np.fft.ifft2(np.fft.ifftshift(residual_fk)))
        residual = das.copy()
        residual.data = residual_data
        residual.labels = ["residual"]
        separated.append(residual)

    return separated


# ---------------------------------------------------------------------------
# 2. NMF separation
# ---------------------------------------------------------------------------


def nmf_separation(
    das: DASData,
    n_components: int = 2,
    max_iter: int = 500,
    random_state: int = 0,
) -> List[DASData]:
    """Separate events via Non-negative Matrix Factorisation (NMF).

    The FK magnitude spectrum is factorised into ``n_components`` basis
    images.  Each component is then back-projected into the time-space domain
    using the original FK phases.

    Parameters
    ----------
    das : DASData
        Mixed input record.
    n_components : int
        Number of source events to extract.
    max_iter : int
        Maximum NMF iterations.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    list of DASData
        Separated event records (length = ``n_components``).
    """
    data = das.data
    fk, freqs, wavenums = _fk_spectrum(data, das.dt, das.dx)
    magnitude = np.abs(fk)
    phase = np.angle(fk)

    n_freq, n_wav = magnitude.shape
    X = magnitude.reshape(n_freq, n_wav)

    model = NMF(
        n_components=n_components,
        max_iter=max_iter,
        random_state=random_state,
        init="nndsvda",
    )
    W = model.fit_transform(X)  # (n_freq, n_components)
    H = model.components_       # (n_components, n_wav)

    separated = []
    for i in range(n_components):
        component_magnitude = np.outer(W[:, i], H[i, :]).reshape(n_freq, n_wav)
        component_fk = component_magnitude * np.exp(1j * phase)
        recovered = np.real(np.fft.ifft2(np.fft.ifftshift(component_fk)))
        out = das.copy()
        out.data = recovered
        out.labels = [f"nmf_component_{i}"]
        separated.append(out)

    return separated


# ---------------------------------------------------------------------------
# 3. Sparse separation (matching pursuit in Ricker dictionary)
# ---------------------------------------------------------------------------


def _make_ricker_dictionary(
    dt: float, n_samples: int, freqs: List[float]
) -> np.ndarray:
    """Build a dictionary of Ricker wavelets at different frequencies and shifts."""
    atoms = []
    for f in freqs:
        t = (np.arange(n_samples) - n_samples // 2) * dt
        a = (np.pi * f * t) ** 2
        wavelet = (1 - 2 * a) * np.exp(-a)
        wavelet /= np.linalg.norm(wavelet) + 1e-12
        for shift in range(0, n_samples, max(1, n_samples // 32)):
            atom = np.zeros(n_samples)
            end = min(n_samples, shift + len(wavelet))
            atom[shift:end] = wavelet[: end - shift]
            if np.linalg.norm(atom) > 1e-6:
                atom /= np.linalg.norm(atom)
                atoms.append(atom)
    return np.array(atoms)  # (n_atoms, n_samples)


def sparse_separation(
    das: DASData,
    n_components: int = 2,
    freq_hints: Optional[List[float]] = None,
    n_iterations: int = 100,
    threshold: float = 0.1,
) -> List[DASData]:
    """Separate events using channel-wise sparse decomposition.

    Each channel is decomposed into a sparse sum of atoms from a Ricker
    wavelet dictionary.  Atoms are then grouped by their dominant frequency
    to reconstruct individual event records.

    Parameters
    ----------
    das : DASData
        Mixed input record.
    n_components : int
        Number of source events.
    freq_hints : list of float, optional
        Approximate dominant frequencies (Hz) for each event.  If ``None``,
        frequencies are auto-estimated from the data spectrum.
    n_iterations : int
        Maximum matching pursuit iterations per channel.
    threshold : float
        Stop iterating when the residual norm falls below this fraction of
        the original signal norm.

    Returns
    -------
    list of DASData
        Separated event records (length = ``n_components``).
    """
    data = das.data
    dt = das.dt
    n_channels, n_samples = data.shape

    # Auto-estimate dominant frequencies if not provided
    if freq_hints is None:
        freqs_axis = np.fft.rfftfreq(n_samples, d=dt)
        spectrum = np.mean(np.abs(np.fft.rfft(data, axis=-1)), axis=0)
        # Find local maxima
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(spectrum, distance=max(1, len(freqs_axis) // (2 * n_components)))
        peak_freqs = freqs_axis[peaks]
        # Keep the n_components strongest peaks
        if len(peak_freqs) >= n_components:
            strengths = spectrum[peaks]
            idx = np.argsort(strengths)[::-1][:n_components]
            freq_hints = sorted(peak_freqs[idx].tolist())
        else:
            # Fallback: evenly spread
            fmax = 0.4 / dt
            freq_hints = [fmax * (i + 1) / (n_components + 1) for i in range(n_components)]

    # Build dictionary
    dictionary = _make_ricker_dictionary(dt, n_samples, freq_hints)
    n_atoms = len(dictionary)

    # Determine which atoms belong to which component
    atom_components = []
    for a_idx in range(n_atoms):
        t = (np.arange(n_samples) - n_samples // 2) * dt
        atom = dictionary[a_idx]
        # Estimate dominant frequency via peak of atom's spectrum
        spec = np.abs(np.fft.rfft(atom))
        freqs_axis = np.fft.rfftfreq(n_samples, d=dt)
        f_dom = freqs_axis[np.argmax(spec)] if np.any(spec > 0) else 0.0
        # Assign to nearest freq hint
        dists = [abs(f_dom - fh) for fh in freq_hints]
        atom_components.append(int(np.argmin(dists)))

    components_data = [np.zeros((n_channels, n_samples)) for _ in range(n_components)]

    for ch in range(n_channels):
        residual = data[ch].copy()
        coefficients = np.zeros(n_atoms)
        original_norm = np.linalg.norm(residual) + 1e-12

        for _ in range(n_iterations):
            if np.linalg.norm(residual) / original_norm < threshold:
                break
            # Correlate residual with all atoms
            correlations = dictionary @ residual
            best = int(np.argmax(np.abs(correlations)))
            coeff = correlations[best]
            coefficients[best] += coeff
            residual -= coeff * dictionary[best]

        # Reconstruct components from their assigned atoms
        for a_idx in range(n_atoms):
            if coefficients[a_idx] != 0:
                c = atom_components[a_idx]
                components_data[c][ch] += coefficients[a_idx] * dictionary[a_idx]

    result = []
    for i in range(n_components):
        out = das.copy()
        out.data = components_data[i]
        out.labels = [f"sparse_component_{i}"]
        result.append(out)
    return result


# ---------------------------------------------------------------------------
# High-level separator
# ---------------------------------------------------------------------------


class DASEventSeparator:
    """High-level interface for DAS multi-event separation.

    Parameters
    ----------
    method : {'fk', 'nmf', 'sparse'}
        Separation algorithm to use.
    n_components : int
        Expected number of seismic events.
    **kwargs
        Additional keyword arguments forwarded to the chosen algorithm.

    Examples
    --------
    >>> from das_separation import simulate_das_data, DASEventSeparator
    >>> mixed, _ = simulate_das_data(n_channels=32, n_samples=256)
    >>> sep = DASEventSeparator(method='nmf', n_components=2)
    >>> separated = sep.separate(mixed)
    """

    METHODS = ("fk", "nmf", "sparse")

    def __init__(
        self,
        method: str = "nmf",
        n_components: int = 2,
        **kwargs,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got '{method}'")
        self.method = method
        self.n_components = n_components
        self.kwargs = kwargs

    def separate(self, das: DASData) -> List[DASData]:
        """Run separation on *das* and return a list of separated components.

        Parameters
        ----------
        das : DASData
            Mixed DAS record.

        Returns
        -------
        list of DASData
            Separated event records.
        """
        if self.method == "fk":
            velocity_ranges = self.kwargs.get(
                "velocity_ranges",
                [(500.0, 2500.0), (2500.0, 5000.0)][: self.n_components],
            )
            return fk_filter(
                das,
                velocity_ranges=velocity_ranges,
                taper_width=self.kwargs.get("taper_width", 0.05),
                return_all=self.kwargs.get("return_all", False),
            )
        elif self.method == "nmf":
            return nmf_separation(
                das,
                n_components=self.n_components,
                max_iter=self.kwargs.get("max_iter", 500),
                random_state=self.kwargs.get("random_state", 0),
            )
        else:  # sparse
            return sparse_separation(
                das,
                n_components=self.n_components,
                freq_hints=self.kwargs.get("freq_hints", None),
                n_iterations=self.kwargs.get("n_iterations", 100),
                threshold=self.kwargs.get("threshold", 0.1),
            )
