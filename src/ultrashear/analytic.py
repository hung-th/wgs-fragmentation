"""Closed-form results for a stationary renewal process of cuts on finite molecules.

Cuts on a molecule of length ``L`` form a stationary renewal process with rate ``lam`` (cuts/bp)
and gamma-distributed spacings ``X ~ Gamma(shape=k, rate=k * lam)`` (mean ``1/lam``; ``k = 1`` is
a Poisson process). Stationarity means the molecule ends fall at random points of the process.
For such a process the expected fragment-length spectrum of one molecule is exact:

* interior fragments (cut at both ends):  count density ``lam * f(x) * (L - x)``, ``x < L``
* the two end fragments:                   count density ``2 * lam * S(x)``,       ``x < L``
* uncut molecules:                         point mass ``P0 = lam * int_L^inf S``  at ``x = L``

where ``f`` and ``S`` are the gamma density and survival function. Everything below follows by
integrating this spectrum, using incomplete-gamma identities so that it vectorises over ``lam``.
See docs/model.md for the derivations.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, optimize, special

from ultrashear.model import SizeDefinition, StartLength

_LAM_BOUNDS = (1e-9, 1.0)

# Sizing resolution of the electrophoresis instrument: SD of a Gaussian blur in ln(length).
# Only used for spectra and the mode; averages are unaffected.
DEFAULT_RESOLUTION = 0.1


def _as_lam(lam) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    if np.any(lam < 0):
        raise ValueError("cut density must be non-negative")
    return lam


class _Gamma:
    """Spacing distribution helpers, broadcasting over ``lam`` and ``x``."""

    def __init__(self, lam: np.ndarray, k: float):
        self.lam = np.maximum(lam, 1e-300)
        self.k = float(k)
        self.beta = self.k * self.lam

    def cdf(self, x):
        return special.gammainc(self.k, self.beta * x)

    def sf(self, x):
        return special.gammaincc(self.k, self.beta * x)

    def pdf(self, x):
        bx = self.beta * x
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = (
                self.k * np.log(self.beta) + (self.k - 1) * np.log(x) - bx - special.gammaln(self.k)
            )
        return np.where(x > 0, np.exp(logp), 0.0)

    def partial_moment(self, m: int, x):
        """``int_0^x a^m f(a) da``."""
        coef = np.exp(special.gammaln(self.k + m) - special.gammaln(self.k)) / self.beta**m
        return coef * special.gammainc(self.k + m, self.beta * x)

    def sf_moment(self, n: int, x):
        """``int_0^x a^n S(a) da`` (integration by parts)."""
        return (x ** (n + 1) * self.sf(x) + self.partial_moment(n + 1, x)) / (n + 1)

    def p_uncut(self, length):
        """Probability that a stationary process puts no cut on ``[0, length]``."""
        bl = self.beta * length
        p0 = special.gammaincc(self.k + 1, bl) - self.lam * length * special.gammaincc(self.k, bl)
        return np.clip(p0, 0.0, 1.0)


def _grid(lam, start: StartLength):
    """Broadcast ``lam`` (any shape) against the starting-length quadrature nodes."""
    lam = _as_lam(lam)
    lengths, w = start.nodes()
    return lam[..., None], lengths, w


def number_average(lam, start: StartLength) -> np.ndarray:
    """Number-average fragment length: total length / expected number of fragments.

    For a single length this is ``1 / (lam + 1/L0)``, independent of the spacing shape.
    """
    lam_, lengths, w = _grid(lam, start)
    return np.sum(w * lengths) / np.sum(w * (1.0 + lam_ * lengths), axis=-1)


def _weight_average_single(g: _Gamma, length) -> np.ndarray:
    """Mass-weighted mean fragment length for molecules of a single ``length``."""
    lam = g.lam
    i1 = g.sf_moment(1, length)
    i2 = g.sf_moment(2, length)
    return length * g.p_uncut(length) + 2 * lam * i1 - lam * i2 / length


def weight_average(lam, start: StartLength, k: float = 1.0) -> np.ndarray:
    """Weight (mass)-average fragment length ``E[sum l_i^2] / E[sum l_i]``.

    Tends to ``(1 + 1/k) / lam`` for long molecules (2/lam for Poisson) and to ``L0`` as lam -> 0.
    """
    lam_, lengths, w = _grid(lam, start)
    g = _Gamma(lam_, k)
    lw = _weight_average_single(g, lengths)
    lw = np.where(lam_ > 0, lw, lengths)
    return np.sum(w * lengths * lw, axis=-1) / np.sum(w * lengths)


def size_bins(start: StartLength, n_bins: int = 600, x_min: float = 20.0):
    """Log-spaced bins covering fragments up to the longest starting molecule."""
    lengths, _ = start.nodes()
    x_max = max(float(lengths.max()) * 1.5, 5 * x_min)
    edges = np.geomspace(x_min, x_max, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers


def mass_spectrum(
    lam,
    start: StartLength,
    k: float = 1.0,
    edges=None,
    resolution: float = DEFAULT_RESOLUTION,
):
    """Expected mass fraction per unit ``ln(length)`` on log bins (what an electropherogram shows).

    ``edges`` must be log-spaced. ``resolution`` is the SD (in ln units) of a Gaussian blur that
    mimics instrument sizing resolution; 0 disables it.

    Returns ``(centers, density)`` with ``density.shape == lam.shape + (n_bins,)``; integrating
    ``density`` over ``ln(x)`` gives the fraction of mass inside the bin range.
    """
    if edges is None:
        edges, centers = size_bins(start)
    else:
        edges = np.asarray(edges, dtype=float)
        centers = np.sqrt(edges[:-1] * edges[1:])
    dlog = np.diff(np.log(edges))
    lam = _as_lam(lam)
    lengths, w = start.nodes()
    lam_ = lam[..., None, None]
    L = lengths[:, None]
    x = centers[None, :]
    g = _Gamma(lam_, k)
    inside = x < L
    count = np.where(inside, lam_ * g.pdf(x) * (L - x) + 2 * lam_ * g.sf(x), 0.0)
    mass_per_bin = (x**2) * count * dlog  # mass per molecule in each bin
    # Uncut molecules: point mass at L.
    p0 = g.p_uncut(L)[..., 0]  # shape lam + (n_nodes,)
    idx = np.clip(np.searchsorted(edges, lengths) - 1, 0, len(centers) - 1)
    uncut = np.zeros(lam.shape + (len(lengths), len(centers)))
    np.put_along_axis(
        uncut,
        np.broadcast_to(idx[:, None], uncut.shape[:-1] + (1,)),
        (lengths * p0)[..., None],
        axis=-1,
    )
    total = np.sum(w[:, None] * (mass_per_bin + uncut), axis=-2)
    density = total / np.sum(w * lengths) / dlog
    if resolution > 0:
        density = ndimage.gaussian_filter1d(density, resolution / dlog[0], axis=-1, mode="constant")
    return centers, density


def mode(
    lam, start: StartLength, k: float = 1.0, resolution: float = DEFAULT_RESOLUTION
) -> np.ndarray:
    """Peak of the blurred mass-per-log-length spectrum (the visual peak of an electropherogram)."""
    lam = _as_lam(lam)
    edges, centers = size_bins(start)
    out = np.empty(lam.shape)
    flat = lam.reshape(-1)
    res = out.reshape(-1)
    # Chunk to bound memory: n_lam x n_nodes x n_bins.
    for i in range(0, flat.size, 32):
        _, dens = mass_spectrum(flat[i : i + 32], start, k, edges, resolution)
        j = np.argmax(dens, axis=-1)
        res[i : i + 32] = _parabolic_peak(np.log(centers), dens, j)
    return out


def _parabolic_peak(logx: np.ndarray, y: np.ndarray, j: np.ndarray) -> np.ndarray:
    j = np.clip(j, 1, len(logx) - 2)
    rows = np.arange(y.shape[0])
    y0, y1, y2 = y[rows, j - 1], y[rows, j], y[rows, j + 1]
    denom = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(denom < 0, 0.5 * (y0 - y2) / denom, 0.0)
    h = logx[1] - logx[0]
    return np.exp(logx[j] + np.clip(shift, -1, 1) * h)


def fraction_in_window(lam, start: StartLength, k: float, low: float, high: float):
    """Expected fraction of mass and of molecules (fragments) with length in ``[low, high]``.

    Returns ``(mass_fraction, number_fraction)``.
    """
    lam_, lengths, w = _grid(lam, start)
    g = _Gamma(lam_, k)
    L = lengths
    a = np.minimum(low, L)
    b = np.minimum(high, L)
    cnt = lam_ * (L * (g.cdf(b) - g.cdf(a)) - (g.partial_moment(1, b) - g.partial_moment(1, a)))
    cnt += 2 * lam_ * (g.sf_moment(0, b) - g.sf_moment(0, a))
    mass = lam_ * (
        L * (g.partial_moment(1, b) - g.partial_moment(1, a))
        - (g.partial_moment(2, b) - g.partial_moment(2, a))
    )
    mass += 2 * lam_ * (g.sf_moment(1, b) - g.sf_moment(1, a))
    in_win = (L >= low) & (L <= high)
    p0 = g.p_uncut(L)
    cnt = cnt + np.where(in_win, p0, 0.0)
    mass = mass + np.where(in_win, L * p0, 0.0)
    mass_frac = np.sum(w * mass, axis=-1) / np.sum(w * L)
    num_frac = np.sum(w * cnt, axis=-1) / np.sum(w * (1 + lam_ * L), axis=-1)
    return mass_frac, num_frac


def reported_size(
    lam, start: StartLength, k: float = 1.0, definition: SizeDefinition = "weight"
) -> np.ndarray:
    """The statistic that is compared with a measured "average size"."""
    if definition == "number":
        return number_average(lam, start)
    if definition == "weight":
        return weight_average(lam, start, k)
    if definition == "mode":
        return mode(lam, start, k)
    raise ValueError(f"unknown definition {definition!r}")


def max_reported_size(start: StartLength, k: float = 1.0, definition: SizeDefinition = "weight"):
    """Size reported for (essentially) uncut DNA: no cut density can give a larger value."""
    return float(reported_size(np.array([_LAM_BOUNDS[0]]), start, k, definition)[0])


def invert_lambda(
    size_bp: float, start: StartLength, k: float = 1.0, definition: SizeDefinition = "weight"
) -> float:
    """Cut density that makes the forward model report ``size_bp``."""

    def f(loglam: float) -> float:
        return float(np.log(reported_size(np.array([np.exp(loglam)]), start, k, definition)[0]))

    target = np.log(size_bp)
    lo, hi = np.log(_LAM_BOUNDS[0]), np.log(_LAM_BOUNDS[1])
    if f(lo) <= target:
        raise ValueError(
            f"size {size_bp:.0f} bp is not below the size of uncut DNA for this starting length; "
            "cannot infer a cut density"
        )
    if f(hi) >= target:
        raise ValueError(f"size {size_bp:.0f} bp is too small to invert")
    return float(np.exp(optimize.brentq(lambda u: f(u) - target, lo, hi, xtol=1e-10)))
