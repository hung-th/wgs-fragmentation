"""Monte Carlo fragmentation engine.

Molecules are laid end to end on one line and a single stationary renewal process of cuts
(gamma spacings with shape ``k`` and mean ``1/lam``) is run along it; molecule boundaries are
added as extra cut points. Each molecule therefore sees a stationary renewal process, which is
exactly the model in :mod:`ultrashear.analytic`, and the whole run is a few numpy calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ultrashear.model import StartLength

MAX_EXPECTED_FRAGMENTS = 20_000_000


@dataclass
class SimulationResult:
    fragments: np.ndarray  # fragment lengths (bp)
    start_lengths: np.ndarray  # simulated starting-molecule lengths (bp)
    lam: float
    k: float


def _renewal_points(total: float, lam: float, k: float, rng: np.random.Generator) -> np.ndarray:
    """Points of a stationary gamma renewal process on ``[0, total]``."""
    if lam <= 0:
        return np.empty(0)
    scale = 1.0 / (k * lam)
    # First point: forward recurrence time = U * (length-biased spacing ~ Gamma(k + 1)).
    first = rng.uniform() * rng.gamma(k + 1, scale)
    chunks = [np.array([first])]
    pos = first
    mean_n = lam * total
    while pos < total:
        n = int(max(mean_n - lam * pos, 0) + 6 * np.sqrt(mean_n / k + 1) + 16)
        gaps = rng.gamma(k, scale, size=n)
        pts = pos + np.cumsum(gaps)
        chunks.append(pts)
        pos = pts[-1]
    pts = np.concatenate(chunks)
    return pts[pts < total]


def simulate(
    lam: float,
    start: StartLength,
    k: float = 1.0,
    n_molecules: int = 20_000,
    seed: int | None = 0,
) -> SimulationResult:
    """Fragment ``n_molecules`` starting molecules at cut density ``lam`` (cuts/bp)."""
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if k < 1:
        raise ValueError("k must be >= 1")
    rng = np.random.default_rng(seed)
    lengths = start.sample(n_molecules, rng)
    expected = float(np.sum(1 + lam * lengths))
    if expected > MAX_EXPECTED_FRAGMENTS:
        n_molecules = max(100, int(n_molecules * MAX_EXPECTED_FRAGMENTS / expected))
        lengths = lengths[:n_molecules]
    bounds = np.concatenate([[0.0], np.cumsum(lengths)])
    cuts = _renewal_points(bounds[-1], lam, k, rng)
    points = np.union1d(cuts, bounds)  # sorted, boundaries included
    fragments = np.diff(points)
    return SimulationResult(fragments[fragments > 0], lengths, float(lam), float(k))
