"""Streamlit UI for the UltraShear fragmentation simulator.

Run with:  streamlit run app/streamlit_app.py
All science lives in the ``ultrashear`` package; this file only wires inputs to it and draws.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import ultrashear  # noqa: F401
except ModuleNotFoundError:  # running from a clone without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ultrashear import analytic, metrics
from ultrashear.calibration import LAMBDA_FORMS, calibrate, load_neb_points
from ultrashear.io import fit_trace, read_points, read_trace
from ultrashear.model import (
    CALIBRATED_TIME_RANGE_MIN,
    HMW_WARNING_LENGTH_BP,
    REF_MASS_NG,
    REF_START_LENGTH_BP,
    VALIDATED_MASS_RANGE_NG,
    ModelSettings,
    StartLength,
)
from ultrashear.recommend import (
    DEFAULT_TIME_GRID,
    time_for_target_size,
    time_maximizing_window,
    window_fraction_curve,
)
from ultrashear.simulate import simulate

# Reference data-viz palette (validated categorical order; blue sequential ramp).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_BAND = "rgba(42,120,214,0.18)"
GRAY_BAND = "rgba(128,128,128,0.10)"
SEQ_BLUE = [
    [0.0, "#cde2fb"],
    [0.25, "#86b6ef"],
    [0.5, "#3987e5"],
    [0.75, "#1c5cab"],
    [1.0, "#0d366b"],
]
DEF_LABEL = {
    "weight": "weight (mass) average",
    "number": "number average",
    "mode": "peak (mode)",
}
POINT_COLUMNS = ["time_min", "size_bp", "start_length_bp", "input_ng", "label"]

st.set_page_config(page_title="UltraShear fragmentation planner", page_icon="🧬", layout="wide")


# --------------------------------------------------------------------------------------------
# Cached computations
# --------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Calibrating...")
def get_calibration(points: pd.DataFrame, settings: ModelSettings, model: str):
    return calibrate(points, settings, model=model, n_boot=300)


@st.cache_data(show_spinner=False)
def run_mc(lam: float, start: StartLength, k: float, n: int) -> np.ndarray:
    return simulate(lam, start, k, n_molecules=n, seed=0).fragments


LOG_TICKS = [
    100,
    150,
    200,
    300,
    500,
    700,
    1000,
    1500,
    2000,
    3000,
    5000,
    7000,
    10_000,
    15_000,
    20_000,
    30_000,
    50_000,
    70_000,
    100_000,
    150_000,
    200_000,
]


def size_axis(fig: go.Figure, title: str = "Fragment length (bp)") -> None:
    fig.update_xaxes(type="log", title=title, tickvals=LOG_TICKS, showgrid=True)


def base_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


# --------------------------------------------------------------------------------------------
# Sidebar inputs
# --------------------------------------------------------------------------------------------
st.sidebar.header("Reaction")
t_min = st.sidebar.slider(
    "Fragmentation time at 37 °C (min)",
    1.0,
    60.0,
    20.0,
    0.5,
    help="NEB calibration covers 5-45 min. Outside that range predictions are extrapolated.",
)
mass_ng = st.sidebar.number_input(
    "Input DNA (ng)",
    min_value=0.01,
    max_value=5000.0,
    value=REF_MASS_NG,
    step=5.0,
    help="Validated range per NEB manual: 5-250 ng in a 44 µl reaction.",
)

st.sidebar.header("Starting DNA")
dist_kind = st.sidebar.radio(
    "Length distribution",
    ["Fixed", "Lognormal"],
    horizontal=True,
    help="Lognormal: use for degraded or polydisperse DNA. The median is the mass-weighted "
    "median, i.e. roughly the peak of a gel or genomic-DNA TapeStation trace.",
)
start_median = st.sidebar.number_input(
    "Starting length (bp)" if dist_kind == "Fixed" else "Median length (bp, mass-weighted)",
    min_value=200.0,
    max_value=5_000_000.0,
    value=REF_START_LENGTH_BP,
    step=1000.0,
)
start_cv = 0.5
if dist_kind == "Lognormal":
    start_cv = st.sidebar.slider("Coefficient of variation of length", 0.1, 2.0, 0.6, 0.05)
start = StartLength(start_median, "lognormal" if dist_kind == "Lognormal" else "fixed", start_cv)

st.sidebar.header("Target")
target_bp = st.sidebar.number_input("Target size (bp)", 50.0, 20_000.0, 350.0, 10.0)
c1, c2 = st.sidebar.columns(2)
win_lo = c1.number_input("Window low (bp)", 10.0, 50_000.0, 250.0, 10.0)
win_hi = c2.number_input("Window high (bp)", 20.0, 100_000.0, 500.0, 10.0)
if win_hi <= win_lo:
    st.sidebar.error("Window high must exceed window low.")
    st.stop()

with st.sidebar.expander("Advanced model options"):
    definition = st.selectbox(
        "Measured 'average size' corresponds to",
        ["weight", "number", "mode"],
        format_func=lambda d: DEF_LABEL[d],
        help="TapeStation/Bioanalyzer region averages are signal- (mass-) weighted, so 'weight' "
        "is the default. The calibration is redone under whichever definition you choose.",
    )
    if "gamma_k" not in st.session_state:
        st.session_state["gamma_k"] = 1.0
    gamma_k = st.slider(
        "Gamma shape k of cut spacing",
        1.0,
        10.0,
        step=0.1,
        key="gamma_k",
        help="k = 1: random (Poisson) cuts. k > 1 suppresses very short fragments (narrower "
        "distribution). Uncalibrated; fit it from a trace in the Calibration tab.",
    )
    alpha = st.slider(
        "Input-mass exponent α",
        0.0,
        1.0,
        0.0,
        0.05,
        help="Assumption: t_eff = t·(50 ng / input)^α. α = 0 (default): input amount has no "
        "effect, consistent with the NEB manual, which does not state one.",
    )
    model_choice = st.selectbox(
        "λ(t) model",
        ["auto"] + list(LAMBDA_FORMS),
        format_func=lambda m: "auto (lowest AICc)" if m == "auto" else LAMBDA_FORMS[m].formula,
    )
    ref_L0 = st.number_input(
        "Assumed starting length of the NEB calibration DNA (bp)",
        5_000.0,
        1e7,
        REF_START_LENGTH_BP,
        5_000.0,
    )
    resolution = st.slider(
        "Instrument sizing resolution (SD, ln units)",
        0.0,
        0.3,
        analytic.DEFAULT_RESOLUTION,
        0.01,
        help="Blur applied to simulated electropherograms and the 'peak' statistic.",
    )
    n_mc = st.select_slider("Monte Carlo molecules", [2_000, 5_000, 10_000, 20_000, 50_000], 20_000)

with st.sidebar.expander("Your calibration points"):
    use_neb = st.checkbox("Include NEB reference points", value=True)
    up = st.file_uploader(
        "Upload CSV (time_min, size_bp[, start_length_bp, input_ng, label])", type=["csv", "tsv"]
    )
    st.caption("…or type them in (blank start length/input = calibration reference):")
    if "manual_points" not in st.session_state:
        st.session_state["manual_points"] = pd.DataFrame(
            {c: pd.Series(dtype="float" if c != "label" else "str") for c in POINT_COLUMNS}
        )
    manual = st.data_editor(
        st.session_state["manual_points"],
        num_rows="dynamic",
        key="points_editor",
        use_container_width=True,
    )

user_points = []
if up is not None:
    try:
        user_points.append(read_points(up))
    except ValueError as exc:
        st.sidebar.error(f"Could not read points: {exc}")
manual_clean = manual.dropna(subset=["time_min", "size_bp"])
if len(manual_clean):
    user_points.append(manual_clean.assign(source="user"))
user_df = pd.concat(user_points, ignore_index=True) if user_points else pd.DataFrame()

frames = ([load_neb_points()] if use_neb else []) + ([user_df] if len(user_df) else [])
if not frames:
    st.error("No calibration data: include the NEB points or add your own.")
    st.stop()
points = pd.concat(frames, ignore_index=True)

settings = ModelSettings(
    definition=definition,
    gamma_shape=float(gamma_k),
    alpha=float(alpha),
    ref_start_length_bp=float(ref_L0),
)
try:
    cal = get_calibration(points, settings, model_choice)
except ValueError as exc:
    st.error(f"Calibration failed: {exc}")
    st.stop()

# --------------------------------------------------------------------------------------------
# Core predictions
# --------------------------------------------------------------------------------------------
n_band = 40 if definition == "mode" else 300
t_grid = np.arange(1.0, 60.0 + 1e-9, 1.0 if definition == "mode" else 0.5)
curve = cal.predict(t_grid, start, mass_ng, n_samples=n_band)
now = cal.predict([t_min], start, mass_ng, n_samples=n_band).iloc[0]
lam_now = float(now["lambda_per_bp"])
frags = run_mc(lam_now, start, float(gamma_k), int(n_mc))
mc = metrics.summarize(frags, win_lo, win_hi)
mass_frac_a, num_frac_a = analytic.fraction_in_window(
    np.array([lam_now]), start, gamma_k, win_lo, win_hi
)
rec = time_for_target_size(cal, target_bp, start, mass_ng, n_samples=n_band)
win = time_maximizing_window(cal, win_lo, win_hi, start, mass_ng)

# --------------------------------------------------------------------------------------------
# Header, headline numbers and warnings
# --------------------------------------------------------------------------------------------
st.title("UltraShear fragmentation planner")
st.caption(
    "Predicts the fragment-size outcome of NEBNext UltraShear enzymatic fragmentation from "
    "time, starting DNA length and input amount, and suggests a time to reach a target size. "
    "**Planning aid only. It does not replace a test fragmentation of your own samples.**"
)

with st.popover("How length and input affect fragmentation"):
    st.markdown(
        """
**Starting length**
- Fragment size depends on cuts per bp: `1/size ≈ λ(t) + 1/L0`.
- Above ~50 kb, `1/L0` is negligible: 50 kb, 1 Mb and 40 Mb give the same sizes (within ~1 %).
- Length only matters below ~10 kb, where molecule ends make fragments shorter, so less time is
  needed.

**Input amount**
- Cutting rate per bp is set by the free enzyme concentration, not by the amount of DNA. Less
  DNA means fewer cuts in total, but also fewer bp to cut, so the size is the same (default,
  α = 0).
- Less DNA fragments faster only if the DNA ties up a large share of the enzyme (saturation).
  The α setting models this; α = 1 is the fully saturated extreme.
- NEB uses one protocol for 5-250 ng, which suggests the effect is small.

**Check it yourself:** fragment the same DNA at a low and a high input (e.g. 10 and 200 ng)
for the same time. Equal sizes → α ≈ 0. Smaller at low input → add both as calibration
points and raise α until they fit.
"""
    )

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    f"Predicted size at {t_min:g} min",
    f"{now['size_bp']:.0f} bp",
    help=f"{DEF_LABEL[definition].capitalize()}. Band = 5-95 % bootstrap prediction interval.",
)
m1.caption(f"{DEF_LABEL[definition]}; 90 % band {now['size_lo_bp']:.0f}-{now['size_hi_bp']:.0f} bp")
m2.metric(f"Mass in {win_lo:.0f}-{win_hi:.0f} bp", f"{mass_frac_a[0]:.0%}")
m2.caption(f"Molecules in window: {num_frac_a[0]:.0%}")
m3.metric(
    f"Time for {target_bp:.0f} bp",
    "n/a" if np.isnan(rec.time_min) else f"{rec.time_min:.1f} min",
)
m3.caption(
    "" if np.isnan(rec.time_min) else f"range {rec.time_lo_min:.1f}-{rec.time_hi_min:.1f} min"
)
m4.metric("Time maximising mass in window", f"{win.time_min:.1f} min")
m4.caption(f"≥90 % of max: {win.time_range_min[0]:.1f}-{win.time_range_min[1]:.1f} min")

warnings: list[str] = []
lo_t, hi_t = CALIBRATED_TIME_RANGE_MIN
if not lo_t <= t_min <= hi_t:
    warnings.append(
        f"Time {t_min:g} min is outside the calibrated {lo_t:g}-{hi_t:g} min range: extrapolated."
    )
if not VALIDATED_MASS_RANGE_NG[0] <= mass_ng <= VALIDATED_MASS_RANGE_NG[1]:
    warnings.append(
        f"Input {mass_ng:g} ng is outside NEB's validated {VALIDATED_MASS_RANGE_NG[0]:g}-"
        f"{VALIDATED_MASS_RANGE_NG[1]:g} ng range."
    )
if alpha > 0:
    warnings.append(
        f"Input-mass effect α = {alpha:g} is an assumption, not supported by calibration data."
    )
if start.median_bp < HMW_WARNING_LENGTH_BP or (start.kind == "lognormal" and start.cv > 1):
    warnings.append(
        "Starting DNA is far from the high-molecular-weight reference. The starting-length "
        "effect comes from the random-cleavage model and has not been validated against data. "
        "For degraded/FFPE DNA, NEB suggests shorter times (5-15 min) with optimisation."
    )
if gamma_k != 1:
    warnings.append(f"Gamma shape k = {gamma_k:g} is user-set (uncalibrated unless fitted).")
if not rec.within_calibrated_range and not np.isnan(rec.time_min):
    warnings.append(f"Recommended time for {target_bp:.0f} bp is outside the calibrated range.")
if np.isnan(rec.time_min):
    warnings.append(rec.message)
warnings += cal.notes
for w in warnings:
    st.warning(w, icon="⚠️")

# --------------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------------
tab_time, tab_dist, tab_heat, tab_rec, tab_cal, tab_about = st.tabs(
    ["Size vs time", "Size distribution", "Heatmaps", "Recommend", "Calibration", "Assumptions"]
)
figures: dict[str, go.Figure] = {}

with tab_time:
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=lo_t, fillcolor=GRAY_BAND, line_width=0, layer="below")
    fig.add_vrect(
        x0=hi_t,
        x1=61,
        fillcolor=GRAY_BAND,
        line_width=0,
        layer="below",
        annotation_text="extrapolated",
        annotation_position="top right",
    )
    fig.add_trace(
        go.Scatter(
            x=np.r_[curve["time_min"], curve["time_min"][::-1]],
            y=np.r_[curve["size_hi_bp"], curve["size_lo_bp"][::-1]],
            fill="toself",
            fillcolor=BLUE_BAND,
            line=dict(width=0),
            hoverinfo="skip",
            name="90 % prediction band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve["time_min"],
            y=curve["size_bp"],
            mode="lines",
            line=dict(color=BLUE, width=2),
            name="Prediction (your DNA)",
            hovertemplate="%{y:.0f} bp",
        )
    )
    if start != StartLength(ref_L0) or mass_ng != REF_MASS_NG:
        ref_curve = cal.predict(t_grid, StartLength(ref_L0), REF_MASS_NG, n_samples=1)
        fig.add_trace(
            go.Scatter(
                x=ref_curve["time_min"],
                y=ref_curve["size_bp"],
                mode="lines",
                line=dict(color=BLUE, width=2, dash="dot"),
                name="Reference (HMW, 50 ng)",
                hovertemplate="%{y:.0f} bp",
            )
        )
    for src, color, sym in (("NEB", ORANGE, "circle"), ("user", AQUA, "diamond")):
        p = cal.points[cal.points["source"] == src]
        if len(p):
            fig.add_trace(
                go.Scatter(
                    x=p["time_min"],
                    y=p["size_bp"],
                    mode="markers",
                    marker=dict(
                        color=color, size=9, symbol=sym, line=dict(width=1.5, color="white")
                    ),
                    name="NEB data (50 ng HMW)" if src == "NEB" else "Your points",
                    hovertemplate="%{y:.0f} bp measured",
                )
            )
    fig.add_hline(
        y=target_bp,
        line=dict(dash="dash", width=1, color="gray"),
        annotation_text=f"target {target_bp:.0f} bp",
        annotation_position="bottom left",
    )
    fig.add_vline(x=t_min, line=dict(width=1, color="gray"))
    fig.update_yaxes(
        type="log",
        title=f"Predicted {DEF_LABEL[definition]} (bp)",
        tickvals=LOG_TICKS,
        minor=dict(showgrid=False),
    )
    fig.update_xaxes(title="Fragmentation time at 37 °C (min)", range=[0, 61])
    st.plotly_chart(base_layout(fig, 460), use_container_width=True)
    figures["size_vs_time"] = fig
    st.caption(
        "NEB points are for 50 ng high-molecular-weight human DNA. Your curve applies the "
        "random-cleavage model to your starting length and input. Shaded areas are extrapolated."
    )

with tab_dist:
    edges = np.geomspace(max(frags.min(), 10), frags.max() * 1.01, 160)
    centers = np.sqrt(edges[1:] * edges[:-1])
    mass_h = metrics.log_histogram(frags, edges, "mass")
    num_h = metrics.log_histogram(frags, edges, "number")
    a_edges = np.geomspace(max(frags.min(), 10), frags.max() * 1.01, 800)
    a_x, a_d = analytic.mass_spectrum(np.array([lam_now]), start, gamma_k, a_edges, resolution)
    fig = go.Figure()
    fig.add_vrect(x0=win_lo, x1=win_hi, fillcolor=BLUE_BAND, line_width=0, layer="below")
    # Annotations on a log axis take log10 coordinates (shapes take data coordinates).
    fig.add_annotation(
        x=np.log10(win_lo),
        y=0.98,
        yref="paper",
        text=" target window",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=11),
    )
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=mass_h,
            mode="lines",
            line_shape="hvh",
            line=dict(color=BLUE, width=2),
            name="Mass-weighted (simulated)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=a_x,
            y=a_d[0],
            mode="lines",
            line=dict(color=BLUE, width=2, dash="dot"),
            name="Mass-weighted (analytic, instrument blur)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=centers,
            y=num_h,
            mode="lines",
            line_shape="hvh",
            line=dict(color=ORANGE, width=2),
            name="Number-weighted (simulated)",
        )
    )
    size_axis(fig)
    x_hi = metrics.weighted_quantile(frags, frags, 0.995) * 1.5
    fig.update_xaxes(range=[np.log10(max(frags.min(), 20)), np.log10(x_hi)])
    fig.update_yaxes(title="Density per ln(length)", rangemode="tozero")
    st.plotly_chart(base_layout(fig), use_container_width=True)
    figures["size_distribution"] = fig
    st.caption(
        f"Monte Carlo: {mc['n_fragments']:,} fragments from {int(n_mc):,} molecules at "
        f"λ = {lam_now:.2e} cuts/bp. The mass-weighted curve is what an electropherogram shows."
    )
    stats = pd.DataFrame(
        {
            "Statistic": [
                "Number average",
                "Weight average",
                "Median (by count)",
                "N50",
                "Peak (mass per log length)",
                "Mass fraction in window",
                "Molecule fraction in window",
            ],
            "Monte Carlo": [
                mc["number_average_bp"],
                mc["weight_average_bp"],
                mc["median_bp"],
                mc["n50_bp"],
                mc["mode_bp"],
                mc["mass_fraction_in_window"],
                mc["number_fraction_in_window"],
            ],
            "Analytic": [
                analytic.number_average(np.array([lam_now]), start)[0],
                analytic.weight_average(np.array([lam_now]), start, gamma_k)[0],
                np.nan,
                np.nan,
                analytic.mode(np.array([lam_now]), start, gamma_k, resolution)[0],
                mass_frac_a[0],
                num_frac_a[0],
            ],
        }
    )
    st.dataframe(
        stats.style.format(
            lambda v: "" if pd.isna(v) else (f"{v:.1%}" if v <= 1 else f"{v:,.0f} bp"),
            subset=["Monte Carlo", "Analytic"],
        ),
        hide_index=True,
        use_container_width=True,
    )

with tab_heat:
    ht = np.arange(2.0, 60.0 + 1e-9, 2.0)
    hL = np.geomspace(500, 200_000, 36)
    s = cal.settings
    lam_grid = cal.lam(s.effective_time(ht, mass_ng))
    Z = np.stack(
        [
            analytic.reported_size(lam_grid, StartLength(L), s.gamma_shape, s.definition)
            if definition != "mode"
            else analytic.reported_size(lam_grid, StartLength(L), s.gamma_shape, "weight")
            for L in hL
        ]
    )
    stat_name = DEF_LABEL[definition] if definition != "mode" else DEF_LABEL["weight"]
    fig = go.Figure(
        go.Heatmap(
            x=ht,
            y=hL,
            z=Z,
            colorscale=SEQ_BLUE,
            reversescale=True,
            colorbar=dict(title="bp"),
            hovertemplate="t = %{x} min<br>L0 = %{y:,.0f} bp<br>%{z:.0f} bp<extra></extra>",
        )
    )
    fig.add_trace(
        go.Contour(
            x=ht,
            y=hL,
            z=Z,
            contours=dict(start=target_bp, end=target_bp, size=1, coloring="none", showlabels=True),
            line=dict(color="white", width=2, dash="dash"),
            showscale=False,
            hoverinfo="skip",
            name=f"{target_bp:.0f} bp",
        )
    )
    fig.update_yaxes(type="log", title="Starting length, fixed (bp)", tickvals=LOG_TICKS[::2])
    fig.update_xaxes(title="Time (min)")
    fig.update_layout(title=f"Predicted {stat_name} vs time and starting length ({mass_ng:g} ng)")
    st.plotly_chart(base_layout(fig, 460), use_container_width=True)
    figures["heatmap_time_length"] = fig
    st.caption(
        "Starting-length dependence is a model prediction (random cleavage), not measured. "
        "Dashed contour: target size."
        + (" Peak statistic shown as weight average for speed." if definition == "mode" else "")
    )

    hm = np.geomspace(1, 500, 30)
    Zm = np.stack(
        [
            analytic.reported_size(
                cal.lam(s.effective_time(ht, m)),
                start,
                s.gamma_shape,
                s.definition if definition != "mode" else "weight",
            )
            for m in hm
        ]
    )
    fig = go.Figure(
        go.Heatmap(
            x=ht,
            y=hm,
            z=Zm,
            colorscale=SEQ_BLUE,
            reversescale=True,
            colorbar=dict(title="bp"),
            hovertemplate="t = %{x} min<br>input = %{y:.1f} ng<br>%{z:.0f} bp<extra></extra>",
        )
    )
    fig.add_hrect(
        y0=VALIDATED_MASS_RANGE_NG[0],
        y1=VALIDATED_MASS_RANGE_NG[1],
        line=dict(color="white", width=1, dash="dot"),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(
        type="log", title="Input DNA (ng)", tickvals=[1, 2, 5, 10, 20, 50, 100, 250, 500]
    )
    fig.update_xaxes(title="Time (min)")
    fig.update_layout(title=f"Predicted {stat_name} vs time and input (α = {alpha:g})")
    st.plotly_chart(base_layout(fig, 420), use_container_width=True)
    figures["heatmap_time_mass"] = fig
    st.caption(
        "**Assumption-driven.** Input-amount effects are not in the calibration data; this map "
        "only reflects the α you set (α = 0 gives no dependence on input). Dotted box: NEB's "
        "validated 5-250 ng range."
    )

with tab_rec:
    st.subheader(f"Time to reach {target_bp:.0f} bp ({DEF_LABEL[definition]})")
    (st.success if rec.within_calibrated_range else st.warning)(rec.message)
    st.subheader(f"Time maximising the fraction in {win_lo:.0f}-{win_hi:.0f} bp")
    (st.success if win.within_calibrated_range else st.warning)(win.message)
    mf, nf = window_fraction_curve(cal, win_lo, win_hi, start, mass_ng, DEFAULT_TIME_GRID)
    fig = go.Figure()
    fig.add_vrect(
        x0=hi_t,
        x1=DEFAULT_TIME_GRID[-1],
        fillcolor=GRAY_BAND,
        line_width=0,
        layer="below",
        annotation_text="extrapolated",
        annotation_position="top right",
    )
    fig.add_vrect(x0=0, x1=lo_t, fillcolor=GRAY_BAND, line_width=0, layer="below")
    fig.add_trace(
        go.Scatter(
            x=DEFAULT_TIME_GRID,
            y=mf,
            name="Mass fraction",
            line=dict(color=BLUE, width=2),
            hovertemplate="%{y:.1%}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=DEFAULT_TIME_GRID,
            y=nf,
            name="Molecule fraction",
            line=dict(color=ORANGE, width=2),
            hovertemplate="%{y:.1%}",
        )
    )
    fig.update_yaxes(title="Fraction in window", tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(title="Time (min)")
    st.plotly_chart(base_layout(fig, 380), use_container_width=True)
    figures["window_fraction"] = fig
    st.info(
        "**Suggested minimal calibration run.** Fragment one high-molecular-weight control DNA "
        "and two of your own samples at two times that bracket the recommended time (e.g. "
        f"{max(rec.time_min - 5, 3) if not np.isnan(rec.time_min) else 10:.0f} and "
        f"{rec.time_min + 5 if not np.isnan(rec.time_min) else 20:.0f} min). Measure the "
        "starting length and the fragment size on the same instrument, then enter the results "
        "under *Your calibration points* (sidebar) to refit."
    )

with tab_cal:
    st.subheader("Fit of λ(t)")
    tbl = cal.summary_table()
    st.dataframe(
        tbl.style.format(
            {"AICc": "{:.1f}", "LOO RMSE (size, %)": "{:.1f}", "max in-sample error (%)": "{:.1f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    pts = cal.points[
        [
            "source",
            "time_min",
            "size_bp",
            "start_length_bp",
            "input_ng",
            "lambda_per_bp",
            "fitted_size_bp",
            "rel_error",
        ]
    ].astype({"start_length_bp": float, "input_ng": float})
    st.dataframe(
        pts.style.format(
            {
                "lambda_per_bp": "{:.2e}",
                "fitted_size_bp": "{:.0f}",
                "rel_error": "{:+.1%}",
                "start_length_bp": "{:,.0f}",
                "input_ng": "{:g}",
            },
            na_rep="reference",
        ),
        hide_index=True,
        use_container_width=True,
    )
    if len(user_df) and use_neb:
        neb_only = get_calibration(load_neb_points(), settings, model_choice)
        before = neb_only.predict([t_min], start, mass_ng, n_samples=1)["size_bp"].iloc[0]
        st.metric(
            f"Prediction at {t_min:g} min after adding your points",
            f"{now['size_bp']:.0f} bp",
            f"{now['size_bp'] - before:+.0f} bp vs NEB-only",
            delta_color="off",
        )

    st.subheader("How much does the size definition matter?")
    comp = []
    for d in ("number", "weight", "mode"):
        c = get_calibration(
            points, ModelSettings(d, gamma_k, alpha, REF_MASS_NG, ref_L0), model_choice
        )
        comp.append(
            {
                "definition": DEF_LABEL[d],
                "predicted size at chosen time (bp)": c.predict(
                    [t_min], start, mass_ng, n_samples=1
                )["size_bp"].iloc[0],
                "time for target (min)": time_for_target_size(
                    c, target_bp, start, mass_ng, n_samples=10
                ).time_min,
            }
        )
    st.dataframe(
        pd.DataFrame(comp).style.format(precision=1), hide_index=True, use_container_width=True
    )
    st.caption(
        "Each row recalibrates to the same data under a different reading of "
        "'average size'. Differences grow as the starting DNA gets shorter."
    )

    st.subheader("Fit cut density and gamma shape from a trace")
    st.caption(
        "Upload an electropherogram export (CSV with a size column, e.g. 'Size [bp]', and one "
        "signal column per sample) of a fragmented sample. The signal is assumed proportional "
        "to mass per unit log-size. The starting DNA in the sidebar is used."
    )
    trace_file = st.file_uploader(
        "Electropherogram CSV", type=["csv", "tsv", "txt"], key="trace_upload"
    )
    if trace_file is not None:
        try:
            trace, cols = read_trace(trace_file)
            col = st.selectbox("Signal column", cols)
            trace_file.seek(0)
            trace, _ = read_trace(trace_file, col)
            r1, r2 = st.columns(2)
            fit_lo = r1.number_input("Fit from (bp)", 10.0, 1e5, 100.0)
            fit_hi = r2.number_input("Fit to (bp)", 20.0, 1e6, 10_000.0)
            tf = fit_trace(trace, start, (fit_lo, fit_hi), resolution=resolution)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=tf.sizes, y=tf.observed, name="Observed", line=dict(color=ORANGE, width=2)
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=tf.sizes,
                    y=tf.fitted,
                    name="Fitted model",
                    line=dict(color=BLUE, width=2, dash="dot"),
                )
            )
            size_axis(fig)
            fig.update_yaxes(title="Normalised signal per ln(length)")
            st.plotly_chart(base_layout(fig, 360), use_container_width=True)
            st.write(
                f"λ = {tf.lam:.2e} cuts/bp, **k = {tf.k:.2f}**, weight average "
                f"{tf.weight_average_bp:.0f} bp (trace mean in range "
                f"{tf.observed_weight_mean_bp:.0f} bp)."
            )

            def _use_k(value: float) -> None:
                st.session_state["gamma_k"] = value

            st.button(
                "Use this k in the model",
                on_click=_use_k,
                args=(float(np.clip(round(tf.k, 1), 1.0, 10.0)),),
            )
        except ValueError as exc:
            st.error(f"Could not fit trace: {exc}")

with tab_about:
    st.markdown(
        """
| Component | Status |
|---|---|
| Size vs time for ~50 ng high-molecular-weight DNA, 5-45 min | **Calibrated** (NEB data, 9 points) |
| Effect of starting length | **Model prediction** (random cleavage); not validated by data |
| Effect of input amount | **Assumption** (α, default 0 = no effect); NEB validates 5-250 ng |
| Shape of the size distribution (gamma k) | **Assumption** (k = 1 Poisson) unless fitted from a trace |
| Meaning of "average size" | **Choice** (default: mass-weighted, as for TapeStation region averages) |
| Times < 5 or > 45 min | **Extrapolation** |

**Model.** Cuts are a random (renewal) process along each molecule with density λ(t) cuts/bp.
For a molecule of length L₀ the number-average fragment length is 1/(λ + 1/L₀); the weight
average for long DNA is (1 + 1/k)/λ. λ(t) is inferred from the NEB time course and fitted with
1-2 parameter curves chosen by AICc, with a residual bootstrap for the bands. See
`docs/model.md` in the repository for details.

**Data source.** NEBNext UltraShear (NEB #M7634) manual: 50 ng NA12878 DNA, 5-45 min at 37 °C
then 15 min at 65 °C, TapeStation D5000 HS. Cite New England Biolabs when reusing these values.
"""
    )

# --------------------------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------------------------
st.divider()
d1, d2, d3 = st.columns(3)
export = curve.assign(
    definition=definition,
    start_kind=start.kind,
    start_median_bp=start.median_bp,
    start_cv=start.cv if start.kind == "lognormal" else np.nan,
    input_ng=mass_ng,
)
d1.download_button(
    "Predictions (CSV)",
    export.to_csv(index=False),
    "ultrashear_predictions.csv",
    "text/csv",
    use_container_width=True,
)
settings_json = {
    "model_settings": settings.to_dict(),
    "lambda_model": cal.model,
    "lambda_formula": cal.form.formula,
    "lambda_params": cal.params.tolist(),
    "start_length": {"kind": start.kind, "median_bp": start.median_bp, "cv": start.cv},
    "input_ng": mass_ng,
    "time_min": t_min,
    "target_bp": target_bp,
    "window_bp": [win_lo, win_hi],
    "instrument_resolution": resolution,
    "calibration_points": cal.points[
        ["source", "time_min", "size_bp", "start_length_bp", "input_ng"]
    ].to_dict(orient="records"),
    "recommendation": {"target": rec.to_dict(), "window": win.to_dict()},
}
d2.download_button(
    "Settings (JSON)",
    json.dumps(settings_json, indent=2, default=float),
    "ultrashear_settings.json",
    "application/json",
    use_container_width=True,
)
html = "".join(
    f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
    for i, f in enumerate(figures.values())
)
d3.download_button(
    "Plots (interactive HTML)",
    f"<html><body>{html}</body></html>",
    "ultrashear_plots.html",
    "text/html",
    use_container_width=True,
)
st.caption("For PNGs use the camera icon on any chart.")
