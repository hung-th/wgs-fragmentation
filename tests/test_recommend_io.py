import io

import numpy as np
import pandas as pd
import pytest

from ultrashear.calibration import calibrate
from ultrashear.io import fit_trace, read_points, read_trace
from ultrashear.metrics import log_histogram
from ultrashear.model import StartLength
from ultrashear.recommend import time_for_target_size, time_maximizing_window
from ultrashear.simulate import simulate


@pytest.fixture(scope="module")
def cal():
    return calibrate(n_boot=60)


def test_time_for_target_hits_target(cal):
    start = StartLength(50_000)
    r = time_for_target_size(cal, 350, start)
    assert r.within_calibrated_range
    assert r.time_lo_min < r.time_min < r.time_hi_min
    size = cal.predict([r.time_min], start)["size_bp"].iloc[0]
    assert size == pytest.approx(350, rel=0.01)


def test_short_dna_needs_less_time(cal):
    hmw = time_for_target_size(cal, 350, StartLength(50_000)).time_min
    short = time_for_target_size(cal, 350, StartLength(1_500)).time_min
    assert short < hmw


def test_target_above_start_length_is_unreachable(cal):
    r = time_for_target_size(cal, 900, StartLength(800))
    assert np.isnan(r.time_min)
    assert not r.within_calibrated_range


def test_extrapolated_target_is_flagged(cal):
    r = time_for_target_size(cal, 180, StartLength(50_000))
    assert r.time_min > 45
    assert not r.within_calibrated_range
    assert "OUTSIDE" in r.message


def test_window_optimum(cal):
    w = time_maximizing_window(cal, 250, 500, StartLength(50_000))
    assert 0 < w.max_fraction < 1
    assert w.time_range_min[0] <= w.time_min <= w.time_range_min[1]


def test_read_points_aliases():
    text = "Time,Size,Input (ng),L0\n10,800,20,\n30,350,20,8000\n"
    df = read_points(io.StringIO(text))
    assert {"time_min", "size_bp", "input_ng", "start_length_bp"} <= set(df.columns)
    assert df["input_ng"].tolist() == [20, 20]
    assert np.isnan(df["start_length_bp"].iloc[0])


def test_read_points_requires_columns():
    with pytest.raises(ValueError):
        read_points(io.StringIO("a,b\n1,2\n"))


def test_trace_fit_recovers_parameters():
    start = StartLength(50_000)
    f = simulate(0.004, start, 3.0, n_molecules=5_000, seed=2).fragments
    edges = np.geomspace(35, 20_000, 400)
    sizes = np.sqrt(edges[1:] * edges[:-1])
    csv = pd.DataFrame({"Size [bp]": sizes, "A1": log_histogram(f, edges) * 5}).to_csv(index=False)
    trace, cols = read_trace(io.StringIO(csv))
    assert cols == ["A1"]
    fit = fit_trace(trace, start, (50, 10_000), resolution=0.0)
    assert fit.lam == pytest.approx(0.004, rel=0.03)
    assert fit.k == pytest.approx(3.0, rel=0.1)
