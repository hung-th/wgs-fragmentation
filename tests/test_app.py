from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_runs_with_defaults():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any("bp" in m.value for m in at.metric)


def test_app_extrapolation_and_degraded_dna_warnings():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.slider[0].set_value(55.0)
    at.sidebar.radio[0].set_value("Lognormal")
    at.run()
    at.sidebar.number_input[1].set_value(3000.0)
    at.run()
    assert not at.exception
    text = " ".join(w.value for w in at.warning)
    assert "outside the calibrated" in text
    assert "far from the high-molecular-weight" in text


def test_app_mode_definition():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.selectbox[0].set_value("mode")
    at.run()
    assert not at.exception
