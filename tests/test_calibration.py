import numpy as np
import pandas as pd
import pytest

from ultrashear.calibration import calibrate, load_neb_points
from ultrashear.model import ModelSettings, StartLength


@pytest.fixture(scope="module")
def cal():
    return calibrate(n_boot=100)


def test_neb_data_loaded():
    df = load_neb_points()
    assert list(df["time_min"]) == [5, 10, 15, 20, 25, 30, 35, 40, 45]
    assert df["size_bp"].iloc[0] == 1260


@pytest.mark.parametrize("definition", ["number", "weight", "mode"])
def test_reproduces_neb_within_10_percent(definition):
    c = calibrate(settings=ModelSettings(definition=definition), n_boot=20)
    assert np.max(np.abs(c.points["rel_error"])) < 0.10


def test_loo_reported_and_reasonable(cal):
    table = cal.summary_table()
    chosen = table[table["selected"]].iloc[0]
    assert chosen["LOO RMSE (size, %)"] < 10
    assert chosen["AICc"] == table["AICc"].min()


def test_prediction_band_brackets_central(cal):
    pred = cal.predict([10, 30, 45], StartLength(50_000))
    assert np.all(pred["size_lo_bp"] < pred["size_bp"])
    assert np.all(pred["size_bp"] < pred["size_hi_bp"])
    assert np.all(np.diff(pred["size_bp"]) < 0)


def test_short_start_length_reduces_size(cal):
    hmw = cal.predict([10], StartLength(50_000))["size_bp"].iloc[0]
    short = cal.predict([10], StartLength(1_500))["size_bp"].iloc[0]
    assert short < hmw


def test_alpha_zero_means_no_mass_effect(cal):
    a = cal.predict([20], StartLength(50_000), mass_ng=10)["size_bp"].iloc[0]
    b = cal.predict([20], StartLength(50_000), mass_ng=200)["size_bp"].iloc[0]
    assert a == pytest.approx(b)


def test_alpha_positive_lower_mass_fragments_faster():
    c = calibrate(settings=ModelSettings(alpha=0.5), n_boot=20)
    low = c.predict([20], StartLength(50_000), mass_ng=10)["size_bp"].iloc[0]
    high = c.predict([20], StartLength(50_000), mass_ng=200)["size_bp"].iloc[0]
    assert low < high


def test_user_points_with_own_start_length_are_combined():
    user = pd.DataFrame(
        {"time_min": [10, 30], "size_bp": [700, 300], "start_length_bp": [8000, 8000]}
    )
    pts = pd.concat([load_neb_points(), user], ignore_index=True)
    c = calibrate(pts, n_boot=20)
    assert len(c.points) == 11
    assert set(c.points["source"]) == {"NEB", "user"}


def test_impossible_point_is_dropped_with_note():
    user = pd.DataFrame({"time_min": [10], "size_bp": [5000], "start_length_bp": [2000]})
    pts = pd.concat([load_neb_points(), user], ignore_index=True)
    c = calibrate(pts, n_boot=10)
    assert len(c.points) == 9
    assert c.notes
