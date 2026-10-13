"""Load user calibration points and electropherogram traces; fit a trace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from ultrashear.analytic import DEFAULT_RESOLUTION, mass_spectrum, reported_size
from ultrashear.model import StartLength

_POINT_ALIASES = {
    "time_min": ("time_min", "time", "t", "minutes", "min", "time (min)", "fragmentation_time"),
    "size_bp": ("size_bp", "size", "mean_size", "average_size", "avg_size", "size (bp)", "mean"),
    "start_length_bp": ("start_length_bp", "start_length", "l0", "l0_bp", "input_length"),
    "input_ng": ("input_ng", "input", "mass_ng", "mass", "ng", "input (ng)"),
    "label": ("label", "sample", "name", "sample_name"),
}


def _norm(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def read_points(source) -> pd.DataFrame:
    """Read user calibration points from CSV/TSV (path or file-like).

    Required columns: time (min) and measured size (bp). Optional: starting length (bp) and input
    mass (ng); blanks mean "same as the calibration reference". Common header spellings are
    accepted (see ``_POINT_ALIASES``).
    """
    df = pd.read_csv(source, sep=None, engine="python", comment="#")
    rename = {}
    for col in df.columns:
        n = _norm(col)
        for canon, aliases in _POINT_ALIASES.items():
            if n in {_norm(a) for a in aliases} and canon not in rename.values():
                rename[col] = canon
                break
    df = df.rename(columns=rename)
    missing = {"time_min", "size_bp"} - set(df.columns)
    if missing:
        raise ValueError(f"missing required column(s): {sorted(missing)}")
    keep = [c for c in _POINT_ALIASES if c in df.columns]
    df = df[keep].copy()
    for col in ("time_min", "size_bp", "start_length_bp", "input_ng"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_min", "size_bp"])
    if (df["time_min"] <= 0).any() or (df["size_bp"] <= 0).any():
        raise ValueError("time and size must be positive")
    df["source"] = "user"
    return df.reset_index(drop=True)


def read_trace(source, signal_column: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Read an electropherogram export (e.g. TapeStation/Bioanalyzer CSV).

    Uses the first column whose header mentions "size" or "bp" as the size axis and
    ``signal_column`` (default: the first other numeric column) as the signal. Returns
    ``(DataFrame[size_bp, signal], available_signal_columns)``.
    """
    df = pd.read_csv(source, sep=None, engine="python", comment="#")
    size_col = next(
        (c for c in df.columns if "size" in _norm(c) or _norm(c) in ("bp", "size_bp")),
        df.columns[0],
    )
    numeric = [
        c
        for c in df.columns
        if c != size_col and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9
    ]
    if not numeric:
        raise ValueError("no numeric signal column found")
    col = signal_column or numeric[0]
    out = pd.DataFrame(
        {
            "size_bp": pd.to_numeric(df[size_col], errors="coerce"),
            "signal": pd.to_numeric(df[col], errors="coerce"),
        }
    ).dropna()
    out = out[out["size_bp"] > 0].sort_values("size_bp").reset_index(drop=True)
    return out, numeric


@dataclass
class TraceFit:
    lam: float
    k: float
    scale: float
    sizes: np.ndarray
    observed: np.ndarray  # normalised mass per ln(size)
    fitted: np.ndarray
    rmse: float
    weight_average_bp: float
    number_average_bp: float
    observed_weight_mean_bp: float  # mass-weighted mean of the trace inside the fitted range


def fit_trace(
    trace: pd.DataFrame,
    start: StartLength,
    size_range: tuple[float, float] = (100.0, 10_000.0),
    fit_k: bool = True,
    k: float = 1.0,
    signal_per: str = "log",
    resolution: float = DEFAULT_RESOLUTION,
) -> TraceFit:
    """Fit cut density (and optionally gamma shape ``k``) to a measured size distribution.

    ``signal_per="log"`` assumes the exported signal is proportional to mass per unit ln(size),
    which holds approximately when sizing is log-linear in migration time (as for TapeStation and
    Bioanalyzer). Use ``"bp"`` if the signal is already mass per bp.
    """
    sel = trace[(trace["size_bp"] >= size_range[0]) & (trace["size_bp"] <= size_range[1])]
    if len(sel) < 10:
        raise ValueError("too few trace points inside the size range")
    x = sel["size_bp"].to_numpy(float)
    y = np.clip(sel["signal"].to_numpy(float), 0, None)
    if signal_per == "bp":
        y = y * x
    logx = np.log(x)
    area = np.trapezoid(y, logx)
    if area <= 0:
        raise ValueError("trace has no positive signal in range")
    y = y / area
    x_hi = max(start.nodes()[0].max() * 1.5, x.max() * 2)
    edges = np.geomspace(min(15.0, x.min() / 2), x_hi, 800)

    def model(lam, kk):
        centers, dens = mass_spectrum(np.array([lam]), start, kk, edges, resolution)
        return np.interp(logx, np.log(centers), dens[0])

    def loss(u):
        lam = np.exp(u[0])
        kk = 1.0 + np.exp(u[1]) if fit_k else k
        m = model(lam, kk)
        c = np.dot(m, y) / max(np.dot(m, m), 1e-300)
        return float(np.mean((y - c * m) ** 2))

    mean_obs = float(np.trapezoid(y * x, logx) / np.trapezoid(y, logx))
    u0 = np.array([np.log(2.0 / mean_obs), np.log(1.0)])
    best = None
    for du in (0.0, -0.5, 0.5):
        res = optimize.minimize(
            loss,
            u0 + np.array([du, 0.0]),
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-12, "maxiter": 2000},
        )
        if best is None or res.fun < best.fun:
            best = res
    lam = float(np.exp(best.x[0]))
    kk = float(1.0 + np.exp(best.x[1])) if fit_k else float(k)
    m = model(lam, kk)
    c = float(np.dot(m, y) / np.dot(m, m))
    return TraceFit(
        lam=lam,
        k=kk,
        scale=c,
        sizes=x,
        observed=y,
        fitted=c * m,
        rmse=float(np.sqrt(np.mean((y - c * m) ** 2))),
        weight_average_bp=float(reported_size(np.array([lam]), start, kk, "weight")[0]),
        number_average_bp=float(reported_size(np.array([lam]), start, kk, "number")[0]),
        observed_weight_mean_bp=mean_obs,
    )
