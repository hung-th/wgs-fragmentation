"""Shared model settings, constants and the starting-length distribution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

SizeDefinition = Literal["number", "weight", "mode"]
SIZE_DEFINITIONS: tuple[SizeDefinition, ...] = ("number", "weight", "mode")

# Reference conditions of the NEB calibration data.
REF_MASS_NG = 50.0
REF_START_LENGTH_BP = 50_000.0
CALIBRATED_TIME_RANGE_MIN = (5.0, 45.0)

# NEB #M7634 manual: "Starting Material: 5-250 ng purified, genomic DNA", 26 ul DNA in 1X TE,
# 44 ul total reaction, 5-45 min at 37 C then 15 min at 65 C. The manual makes no statement that
# input amount changes fragment size, hence the default alpha = 0.
VALIDATED_MASS_RANGE_NG = (5.0, 250.0)
REACTION_VOLUME_UL = 44.0

# Starting lengths below this are considered "far below HMW" and trigger a warning, because the
# effect of starting length is a model prediction that the NEB data do not test.
HMW_WARNING_LENGTH_BP = 10_000.0


@dataclass(frozen=True)
class StartLength:
    """Starting-molecule length distribution.

    ``kind="fixed"``: every molecule has length ``median_bp``.

    ``kind="lognormal"``: ``median_bp`` is the *mass-weighted* median (what a gel or a TapeStation
    genomic-DNA trace shows, since signal is proportional to mass) and ``cv`` is the coefficient of
    variation of the length distribution. Internally the number-weighted lognormal is used, whose
    median is ``median_bp * exp(-sigma^2)``.
    """

    median_bp: float = REF_START_LENGTH_BP
    kind: Literal["fixed", "lognormal"] = "fixed"
    cv: float = 0.5

    def __post_init__(self) -> None:
        if self.median_bp <= 0:
            raise ValueError("median_bp must be positive")
        if self.kind not in ("fixed", "lognormal"):
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.kind == "lognormal" and self.cv <= 0:
            raise ValueError("cv must be positive for a lognormal distribution")

    @property
    def sigma(self) -> float:
        return float(np.sqrt(np.log1p(self.cv**2))) if self.kind == "lognormal" else 0.0

    @property
    def number_median_bp(self) -> float:
        return self.median_bp * float(np.exp(-(self.sigma**2)))

    def nodes(self, n: int = 61, width_sd: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
        """Quadrature nodes ``(lengths, number_weights)``; weights sum to 1."""
        if self.kind == "fixed":
            return np.array([self.median_bp]), np.array([1.0])
        z = np.linspace(-width_sd, width_sd, n)
        lengths = self.number_median_bp * np.exp(self.sigma * z)
        w = np.exp(-0.5 * z**2)
        return lengths, w / w.sum()

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` molecule lengths (number-weighted)."""
        if self.kind == "fixed":
            return np.full(n, float(self.median_bp))
        return self.number_median_bp * np.exp(self.sigma * rng.standard_normal(n))


@dataclass(frozen=True)
class ModelSettings:
    """Model options that are not calibrated by the NEB data.

    definition: which statistic the reported "average size" corresponds to (see docs/model.md).
    gamma_shape: shape ``k`` of the gamma-distributed spacing between cuts (1 = Poisson).
    alpha: input-mass exponent in ``t_eff = t * (ref_mass_ng / mass_ng) ** alpha`` (0 = no effect).
    ref_start_length_bp: assumed starting length of the calibration DNA.
    """

    definition: SizeDefinition = "weight"
    gamma_shape: float = 1.0
    alpha: float = 0.0
    ref_mass_ng: float = REF_MASS_NG
    ref_start_length_bp: float = REF_START_LENGTH_BP

    def __post_init__(self) -> None:
        if self.definition not in SIZE_DEFINITIONS:
            raise ValueError(f"definition must be one of {SIZE_DEFINITIONS}")
        if self.gamma_shape < 1:
            raise ValueError("gamma_shape must be >= 1")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")

    def effective_time(self, t_min, mass_ng) -> np.ndarray:
        """Input-mass scaling of fragmentation time (an assumption, identity when alpha = 0)."""
        t = np.asarray(t_min, dtype=float)
        m = np.asarray(mass_ng, dtype=float)
        return t * (self.ref_mass_ng / m) ** self.alpha

    def to_dict(self) -> dict:
        return asdict(self)
