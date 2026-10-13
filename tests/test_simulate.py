import numpy as np
import pytest

from ultrashear import analytic as A
from ultrashear import metrics as M
from ultrashear.model import StartLength
from ultrashear.simulate import simulate

CASES = [
    (StartLength(50_000), 1 / 300, 1.0),
    (StartLength(2_000), 1 / 500, 1.0),
    (StartLength(2_000), 1 / 500, 4.0),
    (StartLength(15_000, "lognormal", 0.8), 1 / 800, 2.0),
]


@pytest.mark.parametrize("start,lam,k", CASES)
def test_monte_carlo_matches_analytic(start, lam, k):
    res = simulate(lam, start, k, n_molecules=20_000, seed=1)
    f = res.fragments
    lam_a = np.array([lam])
    # Mass is conserved exactly.
    assert f.sum() == pytest.approx(res.start_lengths.sum(), rel=1e-9)
    # Fragment count: Monte Carlo error is ~ sqrt(sum(lam * L)) cuts.
    n_expected = np.sum(1 + lam * res.start_lengths)
    assert abs(f.size - n_expected) < 5 * np.sqrt(lam * res.start_lengths.sum() / k + 1)
    assert M.number_average(f) == pytest.approx(A.number_average(lam_a, start)[0], rel=0.02)
    assert M.weight_average(f) == pytest.approx(A.weight_average(lam_a, start, k)[0], rel=0.03)
    mf, nf = M.fraction_in_window(f, 250, 500)
    am, an = A.fraction_in_window(lam_a, start, k, 250, 500)
    assert mf == pytest.approx(am[0], abs=0.01)
    assert nf == pytest.approx(an[0], abs=0.01)
    assert M.mode(f) == pytest.approx(A.mode(lam_a, start, k)[0], rel=0.06)


def test_gamma_k1_is_exponential():
    f = simulate(1 / 400, StartLength(1e7), 1.0, n_molecules=20, seed=3).fragments
    # Exponential spacings: CV = 1 and weight/number average ratio = 2.
    assert np.std(f) / np.mean(f) == pytest.approx(1.0, abs=0.03)
    assert M.weight_average(f) / M.number_average(f) == pytest.approx(2.0, abs=0.05)


def test_gamma_shape_narrows_distribution():
    f = simulate(1 / 400, StartLength(1e7), 4.0, n_molecules=20, seed=3).fragments
    assert np.std(f) / np.mean(f) == pytest.approx(0.5, abs=0.03)


def test_zero_cut_density_returns_molecules():
    res = simulate(0.0, StartLength(5_000), n_molecules=100)
    np.testing.assert_allclose(res.fragments, 5_000)


def test_runtime_is_fast():
    import time

    t0 = time.perf_counter()
    simulate(1 / 300, StartLength(50_000), 1.0, n_molecules=20_000)
    assert time.perf_counter() - t0 < 5.0
