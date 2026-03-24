"""
Unit tests for das_separation package.
"""

import numpy as np
import pytest

from das_separation.data import DASData, simulate_das_data
from das_separation.preprocessing import bandpass_filter, normalize, taper
from das_separation.separation import (
    fk_filter,
    nmf_separation,
    sparse_separation,
    DASEventSeparator,
)
from das_separation.evaluation import snr, correlation_coefficient, evaluate_separation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_mixed():
    """Return a small mixed DAS record and its ground-truth components."""
    mixed, components = simulate_das_data(
        n_channels=16,
        n_samples=128,
        dt=0.004,
        dx=10.0,
        noise_std=0.02,
        seed=0,
    )
    return mixed, components


# ---------------------------------------------------------------------------
# Data tests
# ---------------------------------------------------------------------------


class TestDASData:
    def test_shape(self, simple_mixed):
        mixed, _ = simple_mixed
        assert mixed.data.shape == (16, 128)

    def test_properties(self, simple_mixed):
        mixed, _ = simple_mixed
        assert len(mixed.times) == 128
        assert len(mixed.offsets) == 16

    def test_copy_independence(self, simple_mixed):
        mixed, _ = simple_mixed
        copy = mixed.copy()
        copy.data[0, 0] = 9999.0
        assert mixed.data[0, 0] != 9999.0


class TestSimulateDASData:
    def test_default(self):
        mixed, components = simulate_das_data()
        assert mixed.data.shape[0] == 64
        assert mixed.data.shape[1] == 512
        assert len(components) == 2

    def test_custom_events(self):
        events = [
            {"velocity": 1500.0, "t0": 0.05, "frequency": 10.0, "amplitude": 1.0},
            {"velocity": 3000.0, "t0": 0.08, "frequency": 30.0, "amplitude": 0.5},
            {"velocity": 5000.0, "t0": 0.12, "frequency": 20.0, "amplitude": 0.8},
        ]
        mixed, components = simulate_das_data(
            n_channels=32, n_samples=256, events=events, seed=1
        )
        assert len(components) == 3
        assert mixed.data.shape == (32, 256)

    def test_mixed_is_sum_of_components(self):
        """Without noise the mixed record equals the sum of components."""
        mixed, components = simulate_das_data(
            n_channels=8, n_samples=64, noise_std=0.0, seed=2
        )
        reconstructed = sum(c.data for c in components)
        np.testing.assert_allclose(mixed.data, reconstructed, atol=1e-10)

    def test_reproducibility(self):
        m1, _ = simulate_das_data(seed=42)
        m2, _ = simulate_das_data(seed=42)
        np.testing.assert_array_equal(m1.data, m2.data)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


class TestBandpassFilter:
    def test_output_shape(self, simple_mixed):
        mixed, _ = simple_mixed
        filtered = bandpass_filter(mixed, f_low=5.0, f_high=40.0)
        assert filtered.data.shape == mixed.data.shape

    def test_metadata_preserved(self, simple_mixed):
        mixed, _ = simple_mixed
        filtered = bandpass_filter(mixed, f_low=5.0, f_high=40.0)
        assert filtered.dt == mixed.dt
        assert filtered.dx == mixed.dx

    def test_original_unchanged(self, simple_mixed):
        mixed, _ = simple_mixed
        original = mixed.data.copy()
        _ = bandpass_filter(mixed, f_low=5.0, f_high=40.0)
        np.testing.assert_array_equal(mixed.data, original)

    def test_bad_frequencies(self, simple_mixed):
        mixed, _ = simple_mixed
        with pytest.raises(ValueError):
            bandpass_filter(mixed, f_low=0.0, f_high=40.0)


class TestNormalize:
    def test_trace_mode(self, simple_mixed):
        mixed, _ = simple_mixed
        normed = normalize(mixed, mode="trace")
        peaks = np.max(np.abs(normed.data), axis=-1)
        np.testing.assert_allclose(peaks, np.ones(mixed.n_channels), atol=1e-6)

    def test_global_mode(self, simple_mixed):
        mixed, _ = simple_mixed
        normed = normalize(mixed, mode="global")
        assert np.max(np.abs(normed.data)) <= 1.0 + 1e-6

    def test_rms_mode(self, simple_mixed):
        mixed, _ = simple_mixed
        normed = normalize(mixed, mode="rms")
        rms = np.sqrt(np.mean(normed.data ** 2, axis=-1))
        np.testing.assert_allclose(rms, np.ones(mixed.n_channels), atol=1e-6)

    def test_bad_mode(self, simple_mixed):
        mixed, _ = simple_mixed
        with pytest.raises(ValueError):
            normalize(mixed, mode="unknown")


class TestTaper:
    def test_output_shape(self, simple_mixed):
        mixed, _ = simple_mixed
        tapered = taper(mixed)
        assert tapered.data.shape == mixed.data.shape

    def test_edges_attenuated(self, simple_mixed):
        mixed, _ = simple_mixed
        tapered = taper(mixed, taper_fraction=0.1)
        # Endpoints should be smaller in absolute terms than middle
        mid = mixed.n_samples // 2
        assert np.max(np.abs(tapered.data[:, 0])) < np.max(np.abs(tapered.data[:, mid]))


# ---------------------------------------------------------------------------
# Separation tests
# ---------------------------------------------------------------------------


class TestFKFilter:
    def test_output_count_no_residual(self, simple_mixed):
        mixed, _ = simple_mixed
        out = fk_filter(mixed, [(500, 2500), (2500, 5000)], return_all=False)
        assert len(out) == 2

    def test_output_count_with_residual(self, simple_mixed):
        mixed, _ = simple_mixed
        out = fk_filter(mixed, [(500, 2500), (2500, 5000)], return_all=True)
        assert len(out) == 3

    def test_output_shape(self, simple_mixed):
        mixed, _ = simple_mixed
        out = fk_filter(mixed, [(500, 2500)], return_all=False)
        assert out[0].data.shape == mixed.data.shape


class TestNMFSeparation:
    def test_output_count(self, simple_mixed):
        mixed, _ = simple_mixed
        out = nmf_separation(mixed, n_components=2)
        assert len(out) == 2

    def test_output_shape(self, simple_mixed):
        mixed, _ = simple_mixed
        out = nmf_separation(mixed, n_components=2)
        for comp in out:
            assert comp.data.shape == mixed.data.shape


class TestSparseSeparation:
    def test_output_count(self, simple_mixed):
        mixed, _ = simple_mixed
        out = sparse_separation(mixed, n_components=2, n_iterations=20)
        assert len(out) == 2

    def test_output_shape(self, simple_mixed):
        mixed, _ = simple_mixed
        out = sparse_separation(mixed, n_components=2, n_iterations=20)
        for comp in out:
            assert comp.data.shape == mixed.data.shape


class TestDASEventSeparator:
    @pytest.mark.parametrize("method", ["nmf", "fk", "sparse"])
    def test_methods(self, simple_mixed, method):
        mixed, _ = simple_mixed
        kwargs = {}
        if method == "fk":
            kwargs["velocity_ranges"] = [(500.0, 2500.0), (2500.0, 5000.0)]
            kwargs["return_all"] = False
        sep = DASEventSeparator(method=method, n_components=2, **kwargs)
        out = sep.separate(mixed)
        assert len(out) == 2

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            DASEventSeparator(method="invalid")


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_snr_perfect(self, simple_mixed):
        _, components = simple_mixed
        ref = components[0]
        result = snr(ref, ref)
        assert result > 50  # identical signal → very high SNR

    def test_snr_orthogonal(self, simple_mixed):
        _, components = simple_mixed
        ref = components[0]
        noise = ref.copy()
        noise.data = np.random.default_rng(7).standard_normal(ref.data.shape)
        result = snr(noise, ref)
        # Random noise vs signal → low SNR
        assert result < 20

    def test_correlation_identical(self, simple_mixed):
        _, components = simple_mixed
        ref = components[0]
        cc = correlation_coefficient(ref, ref)
        assert abs(cc - 1.0) < 1e-6

    def test_correlation_range(self, simple_mixed):
        _, components = simple_mixed
        ref = components[0]
        random = ref.copy()
        random.data = np.random.default_rng(8).standard_normal(ref.data.shape)
        cc = correlation_coefficient(random, ref)
        assert -1.0 <= cc <= 1.0

    def test_evaluate_separation(self, simple_mixed):
        mixed, components = simple_mixed
        # Use NMF with few components for speed
        estimated = nmf_separation(mixed, n_components=2, max_iter=50)
        results = evaluate_separation(estimated, components)
        assert len(results) == len(components)
        for r in results:
            assert "snr_db" in r
            assert "correlation" in r
            assert "matched_index" in r
