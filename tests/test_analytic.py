import numpy as np
import pytest

from ultrashear import analytic as A
from ultrashear.model import StartLength

LAMS = np.array([1 / 2000, 1 / 600, 1 / 250])


def test_number_average_formula():
    for L0 in (1_000, 5_000, 50_000):
        expected = 1 / (LAMS + 1 / L0)
        np.testing.assert_allclose(A.number_average(LAMS, StartLength(L0)), expected)


@pytest.mark.parametrize("k", [1.0, 2.0, 5.0])
def test_long_molecule_limits(k):
    hmw = StartLength(1e8)
    np.testing.assert_allclose(A.weight_average(LAMS, hmw, k), (1 + 1 / k) / LAMS, rtol=1e-4)
    np.testing.assert_allclose(A.number_average(LAMS, hmw), 1 / LAMS, rtol=1e-4)


def test_weight_to_number_ratio_is_two_for_poisson():
    ratio = A.weight_average(LAMS, StartLength(1e8), 1) / A.number_average(LAMS, StartLength(1e8))
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-4)


def test_k1_weight_average_closed_form():
    # Poisson cuts on a molecule of length L: L_w = (2/lam) * (1 - (1 - exp(-lam L)) / (lam L)).
    L = 3_000.0
    x = LAMS * L
    expected = 2 / LAMS * (1 - (1 - np.exp(-x)) / x)
    np.testing.assert_allclose(A.weight_average(LAMS, StartLength(L), 1), expected, rtol=1e-9)


@pytest.mark.parametrize("definition", ["number", "weight", "mode"])
def test_no_cuts_returns_start_length(definition):
    L0 = 2_000.0
    size = A.reported_size(np.array([1e-9]), StartLength(L0), 1.0, definition)[0]
    assert size == pytest.approx(L0, rel=0.02)


@pytest.mark.parametrize("definition", ["number", "weight", "mode"])
def test_converges_to_hmw_curve(definition):
    lam = np.array([1 / 400])
    far = A.reported_size(lam, StartLength(1e7), 1.0, definition)
    near = A.reported_size(lam, StartLength(2e5), 1.0, definition)
    np.testing.assert_allclose(near, far, rtol=0.01)


def test_shorter_start_gives_smaller_fragments():
    lam = np.array([1 / 800])
    sizes = [A.weight_average(lam, StartLength(L), 1)[0] for L in (1_000, 3_000, 10_000, 1e6)]
    assert np.all(np.diff(sizes) > 0)


def test_fractions_sum_to_one_and_spectrum_integrates():
    for start in (StartLength(4_000), StartLength(20_000, "lognormal", 0.7)):
        m, n = A.fraction_in_window(LAMS, start, 2.0, 0, 1e12)
        np.testing.assert_allclose(m, 1, atol=1e-9)
        np.testing.assert_allclose(n, 1, atol=1e-9)
        edges, _ = A.size_bins(start, x_min=1.0, n_bins=3000)
        _, dens = A.mass_spectrum(LAMS, start, 2.0, edges, resolution=0)
        np.testing.assert_allclose((dens * np.diff(np.log(edges))).sum(-1), 1, atol=5e-3)


def test_mode_long_molecules():
    # Mass per ln(x) for Gamma(k) spacings peaks at (1 + 1/k) / lam.
    for k in (1.0, 3.0):
        got = A.mode(LAMS, StartLength(1e6), k, resolution=0)
        np.testing.assert_allclose(got, (1 + 1 / k) / LAMS, rtol=0.03)


@pytest.mark.parametrize("definition", ["number", "weight", "mode"])
def test_invert_roundtrip(definition):
    start = StartLength(50_000)
    for lam in (1e-3, 4e-3):
        size = A.reported_size(np.array([lam]), start, 1.0, definition)[0]
        assert A.invert_lambda(size, start, 1.0, definition) == pytest.approx(lam, rel=1e-3)


def test_invert_rejects_size_above_start_length():
    with pytest.raises(ValueError):
        A.invert_lambda(5_000, StartLength(3_000))
