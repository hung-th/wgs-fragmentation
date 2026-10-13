"""Summary statistics of a set of fragment lengths."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def number_average(fragments: np.ndarray) -> float:
    return float(np.mean(fragments))


def weight_average(fragments: np.ndarray) -> float:
    return float(np.sum(fragments**2) / np.sum(fragments))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, q * cum[-1])])


def n50(fragments: np.ndarray) -> float:
    """Length such that half of the mass is in fragments at least this long (mass median)."""
    return weighted_quantile(fragments, fragments, 0.5)


def log_histogram(fragments: np.ndarray, edges: np.ndarray, weight: str = "mass") -> np.ndarray:
    """Density per unit ``ln(length)``, normalised to the total (``weight`` = mass or number)."""
    w = fragments if weight == "mass" else np.ones_like(fragments)
    hist, _ = np.histogram(fragments, bins=edges, weights=w)
    return hist / w.sum() / np.diff(np.log(edges))


def mode(fragments: np.ndarray, resolution: float = 0.1, n_bins: int = 600) -> float:
    """Peak of the mass-per-log-length histogram blurred by ``resolution`` (SD in ln units)."""
    lo, hi = max(fragments.min(), 1.0), fragments.max() * 1.0001
    edges = np.geomspace(lo, hi, n_bins + 1)
    dens = log_histogram(fragments, edges, "mass")
    dlog = np.log(edges[1] / edges[0])
    if resolution > 0:
        dens = ndimage.gaussian_filter1d(dens, resolution / dlog, mode="constant")
    j = int(np.argmax(dens))
    return float(np.sqrt(edges[j] * edges[j + 1]))


def fraction_in_window(fragments: np.ndarray, low: float, high: float) -> tuple[float, float]:
    """``(mass_fraction, number_fraction)`` of fragments with length in ``[low, high]``."""
    inside = (fragments >= low) & (fragments <= high)
    return float(fragments[inside].sum() / fragments.sum()), float(inside.mean())


def summarize(fragments: np.ndarray, low: float, high: float) -> dict[str, float]:
    mass_frac, num_frac = fraction_in_window(fragments, low, high)
    return {
        "n_fragments": int(fragments.size),
        "number_average_bp": number_average(fragments),
        "weight_average_bp": weight_average(fragments),
        "median_bp": float(np.median(fragments)),
        "n50_bp": n50(fragments),
        "mode_bp": mode(fragments),
        "mass_fraction_in_window": mass_frac,
        "number_fraction_in_window": num_frac,
    }
