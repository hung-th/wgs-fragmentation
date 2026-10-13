"""Solve for the fragmentation time that hits a target."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ultrashear.analytic import fraction_in_window, reported_size
from ultrashear.calibration import Calibration
from ultrashear.model import CALIBRATED_TIME_RANGE_MIN, StartLength

DEFAULT_TIME_GRID = np.arange(1.0, 120.0 + 1e-9, 0.5)


def _in_calibrated(t: float) -> bool:
    lo, hi = CALIBRATED_TIME_RANGE_MIN
    return lo <= t <= hi


def _crossing(t: np.ndarray, size: np.ndarray, target: float) -> float:
    """First time at which a (decreasing) size curve drops to ``target``; NaN if never."""
    below = np.nonzero(size <= target)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    j = below[0]
    # Interpolate in log size, linear in time.
    y0, y1 = np.log(size[j - 1]), np.log(size[j])
    return float(t[j - 1] + (np.log(target) - y0) / (y1 - y0) * (t[j] - t[j - 1]))


@dataclass
class TargetSizeResult:
    target_bp: float
    time_min: float  # NaN when unreachable within the time grid
    time_lo_min: float
    time_hi_min: float
    within_calibrated_range: bool
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def time_for_target_size(
    cal: Calibration,
    target_bp: float,
    start: StartLength,
    mass_ng: float | None = None,
    band: tuple[float, float] = (5.0, 95.0),
    t_grid: np.ndarray = DEFAULT_TIME_GRID,
    n_samples: int | None = 200,
) -> TargetSizeResult:
    """Time at which the predicted reported size equals ``target_bp`` (with a bootstrap range)."""
    s = cal.settings
    mass = s.ref_mass_ng if mass_ng is None else mass_ng
    t_eff = s.effective_time(t_grid, mass)
    size = reported_size(cal.lam(t_eff), start, s.gamma_shape, s.definition)
    t_star = _crossing(t_grid, size, target_bp)
    if np.isnan(t_star):
        if size[0] <= target_bp:
            msg = (
                f"Predicted size is already <= {target_bp:.0f} bp at {t_grid[0]:g} min: the "
                "starting DNA is too short to need much fragmentation; use the shortest practical "
                "time or skip fragmentation."
            )
        else:
            msg = f"Target not reached within {t_grid[-1]:g} min."
        return TargetSizeResult(target_bp, np.nan, np.nan, np.nan, False, msg)

    samples = cal.lam_samples(t_eff, prediction=True, n_samples=n_samples)
    sizes = reported_size(samples, start, s.gamma_shape, s.definition)
    times = np.array([_crossing(t_grid, row, target_bp) for row in sizes])
    times = times[np.isfinite(times)]
    lo, hi = np.percentile(times, band) if times.size else (np.nan, np.nan)
    inside = _in_calibrated(t_star)
    msg = (
        f"{t_star:.1f} min ({lo:.1f}-{hi:.1f} min) "
        + ("is inside" if inside else "is OUTSIDE")
        + f" the calibrated {CALIBRATED_TIME_RANGE_MIN[0]:g}-{CALIBRATED_TIME_RANGE_MIN[1]:g} min "
        "range."
    )
    return TargetSizeResult(target_bp, t_star, float(lo), float(hi), inside, msg)


@dataclass
class WindowResult:
    low_bp: float
    high_bp: float
    weight: str
    time_min: float
    max_fraction: float
    time_range_min: tuple[float, float]  # times giving >= `tolerance` x the maximum
    within_calibrated_range: bool
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def window_fraction_curve(
    cal: Calibration,
    low: float,
    high: float,
    start: StartLength,
    mass_ng: float | None = None,
    t_grid: np.ndarray = DEFAULT_TIME_GRID,
):
    """Predicted (mass, number) fraction in ``[low, high]`` along ``t_grid``."""
    s = cal.settings
    mass = s.ref_mass_ng if mass_ng is None else mass_ng
    lam = cal.lam(s.effective_time(t_grid, mass))
    return fraction_in_window(lam, start, s.gamma_shape, low, high)


def time_maximizing_window(
    cal: Calibration,
    low: float,
    high: float,
    start: StartLength,
    mass_ng: float | None = None,
    weight: str = "mass",
    tolerance: float = 0.9,
    t_grid: np.ndarray = DEFAULT_TIME_GRID,
) -> WindowResult:
    """Time that maximises the fraction of mass (or molecules) within ``[low, high]``."""
    mass_frac, num_frac = window_fraction_curve(cal, low, high, start, mass_ng, t_grid)
    frac = mass_frac if weight == "mass" else num_frac
    j = int(np.argmax(frac))
    good = t_grid[frac >= tolerance * frac[j]]
    t_star = float(t_grid[j])
    edge = j in (0, len(t_grid) - 1)
    inside = _in_calibrated(t_star)
    msg = (
        f"Max {weight} fraction {frac[j]:.0%} at {t_star:.1f} min; "
        f">= {tolerance:.0%} of max between {good.min():.1f} and {good.max():.1f} min. "
        + ("Inside" if inside else "OUTSIDE")
        + " the calibrated time range."
    )
    if edge:
        msg += " The optimum is at the edge of the searched time range."
    return WindowResult(
        low,
        high,
        weight,
        t_star,
        float(frac[j]),
        (float(good.min()), float(good.max())),
        inside,
        msg,
    )
