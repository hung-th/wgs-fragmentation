"""Calibrate the cut-density curve lambda(t) against measured (time, size) points.

Stage 1: invert the forward model for each point to get the cut density lambda_i that reproduces
its measured size (given that point's starting length and input mass).
Stage 2: fit a 1-2 parameter curve lambda(t) through the lambda_i in log space, choose among the
candidate forms by AICc, report leave-one-out (LOO) error, and bootstrap the residuals for
uncertainty bands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from ultrashear.analytic import invert_lambda, reported_size
from ultrashear.model import ModelSettings, StartLength

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "neb_calibration.csv"


@dataclass(frozen=True)
class LambdaForm:
    name: str
    formula: str
    n_params: int
    x0: tuple[float, ...]
    bounds: tuple[tuple[float, ...], tuple[float, ...]]

    def __call__(self, t, p) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if self.name == "linear":
            out = p[0] * t
        elif self.name == "affine":
            out = p[0] + p[1] * t
        elif self.name == "power":
            out = p[0] * t ** p[1]
        elif self.name == "saturating":
            out = p[0] * (1 - np.exp(-t / p[1]))
        else:  # pragma: no cover
            raise ValueError(self.name)
        return np.maximum(out, 1e-12)


# Parameters are in cuts/bp and minutes. Starting points/bounds cover the NEB scale (~1e-3/bp).
LAMBDA_FORMS: dict[str, LambdaForm] = {
    f.name: f
    for f in (
        LambdaForm("linear", "lambda = a*t", 1, (1e-4,), ((1e-9,), (1.0,))),
        LambdaForm("affine", "lambda = a + b*t", 2, (5e-4, 1e-4), ((-1e-1, 1e-9), (1e-1, 1.0))),
        LambdaForm("power", "lambda = a*t^b", 2, (3e-4, 0.8), ((1e-9, 0.05), (1.0, 3.0))),
        LambdaForm(
            "saturating",
            "lambda = lmax*(1 - exp(-t/tau))",
            2,
            (1e-2, 100.0),
            ((1e-6, 0.1), (10.0, 1e5)),
        ),
    )
}


def load_neb_points(path: str | Path | None = None) -> pd.DataFrame:
    """The NEB reference time course (50 ng HMW human DNA)."""
    if path is None:
        path = DATA_FILE if DATA_FILE.exists() else resources.files("ultrashear") / "neb.csv"
    df = pd.read_csv(path, comment="#")
    df["source"] = "NEB"
    return df


def _fit_form(form: LambdaForm, t: np.ndarray, loglam: np.ndarray) -> np.ndarray:
    def resid(p):
        return np.log(form(t, p)) - loglam

    best = None
    # A few starts guard against the saturating/affine forms stalling at a bound.
    for scale in (1.0, 0.3, 3.0):
        x0 = np.clip(np.array(form.x0) * scale, form.bounds[0], form.bounds[1])
        res = optimize.least_squares(resid, x0, bounds=form.bounds, x_scale="jac")
        if best is None or res.cost < best.cost:
            best = res
    return best.x


@dataclass
class FormResult:
    name: str
    formula: str
    params: np.ndarray
    rss: float  # residual sum of squares in ln(lambda)
    aicc: float
    loo_rmse_log: float  # LOO RMSE in ln(lambda)
    loo_size_rel_errors: np.ndarray  # LOO relative size errors per point
    max_abs_rel_error: float  # in-sample max |relative size error|


@dataclass
class Calibration:
    settings: ModelSettings
    points: pd.DataFrame
    forms: dict[str, FormResult]
    model: str
    boot_params: np.ndarray
    boot_resid: np.ndarray
    notes: list[str] = field(default_factory=list)

    @property
    def form(self) -> LambdaForm:
        return LAMBDA_FORMS[self.model]

    @property
    def params(self) -> np.ndarray:
        return self.forms[self.model].params

    def lam(self, t_eff) -> np.ndarray:
        return self.form(t_eff, self.params)

    def lam_samples(
        self, t_eff, prediction: bool = True, n_samples: int | None = None
    ) -> np.ndarray:
        """Bootstrap draws of lambda(t_eff), shape ``(n_boot, len(t_eff))``.

        With ``prediction=True`` a resampled residual is added (prediction band); otherwise only
        parameter uncertainty is included (confidence band).
        """
        t_eff = np.atleast_1d(np.asarray(t_eff, dtype=float))
        n = len(self.boot_params) if n_samples is None else min(n_samples, len(self.boot_params))
        out = np.stack([self.form(t_eff, p) for p in self.boot_params[:n]])
        if prediction:
            out = out * np.exp(self.boot_resid[:n])[:, None]
        return out

    def predict(
        self,
        t_min,
        start: StartLength,
        mass_ng: float | None = None,
        band: tuple[float, float] = (5.0, 95.0),
        prediction: bool = True,
        n_samples: int | None = None,
    ) -> pd.DataFrame:
        """Predicted reported size vs time with a bootstrap band."""
        s = self.settings
        t = np.atleast_1d(np.asarray(t_min, dtype=float))
        t_eff = s.effective_time(t, s.ref_mass_ng if mass_ng is None else mass_ng)
        lam = self.lam(t_eff)
        size = reported_size(lam, start, s.gamma_shape, s.definition)
        samples = self.lam_samples(t_eff, prediction, n_samples)
        sizes = reported_size(samples, start, s.gamma_shape, s.definition)
        lo, hi = np.percentile(sizes, band, axis=0)
        return pd.DataFrame(
            {
                "time_min": t,
                "t_eff_min": t_eff,
                "lambda_per_bp": lam,
                "size_bp": size,
                "size_lo_bp": lo,
                "size_hi_bp": hi,
            }
        )

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for r in self.forms.values():
            rows.append(
                {
                    "form": r.name,
                    "formula": r.formula,
                    "params": ", ".join(f"{v:.4g}" for v in r.params),
                    "AICc": r.aicc,
                    "LOO RMSE (size, %)": 100 * float(np.sqrt(np.mean(r.loo_size_rel_errors**2))),
                    "max in-sample error (%)": 100 * r.max_abs_rel_error,
                    "selected": r.name == self.model,
                }
            )
        return pd.DataFrame(rows)


def _point_start(row, settings: ModelSettings) -> StartLength:
    L = row.get("start_length_bp")
    if L is None or pd.isna(L):
        L = settings.ref_start_length_bp
    return StartLength(float(L))


def _prepare_points(points: pd.DataFrame, settings: ModelSettings) -> pd.DataFrame:
    df = points.copy().reset_index(drop=True)
    for col in ("start_length_bp", "input_ng"):
        if col not in df:
            df[col] = np.nan
    df["source"] = df["source"].fillna("user") if "source" in df else "user"
    mass = df["input_ng"].fillna(settings.ref_mass_ng).to_numpy(float)
    df["t_eff_min"] = settings.effective_time(df["time_min"].to_numpy(float), mass)
    lam, notes = [], []
    for _, row in df.iterrows():
        try:
            lam.append(
                invert_lambda(
                    float(row["size_bp"]),
                    _point_start(row, settings),
                    settings.gamma_shape,
                    settings.definition,
                )
            )
        except ValueError as exc:
            lam.append(np.nan)
            notes.append(f"Point t={row['time_min']} min, {row['size_bp']} bp dropped: {exc}")
    df["lambda_per_bp"] = lam
    df.attrs["notes"] = notes
    return df


def _fitted_sizes(df: pd.DataFrame, form: LambdaForm, p, settings: ModelSettings) -> np.ndarray:
    out = []
    for _, row in df.iterrows():
        lam = form(np.array([row["t_eff_min"]]), p)
        out.append(
            reported_size(
                lam, _point_start(row, settings), settings.gamma_shape, settings.definition
            )[0]
        )
    return np.array(out)


def calibrate(
    points: pd.DataFrame | None = None,
    settings: ModelSettings | None = None,
    model: str = "auto",
    n_boot: int = 300,
    seed: int = 0,
) -> Calibration:
    """Fit lambda(t). ``points`` needs ``time_min`` and ``size_bp`` columns and may have
    ``start_length_bp`` and ``input_ng`` (blank = calibration reference). Defaults to NEB data."""
    settings = settings or ModelSettings()
    if points is None:
        points = load_neb_points()
    df = _prepare_points(points, settings)
    notes = list(df.attrs.get("notes", []))
    df = df[np.isfinite(df["lambda_per_bp"])].reset_index(drop=True)
    n = len(df)
    if n < 2:
        raise ValueError("need at least two usable calibration points")
    t = df["t_eff_min"].to_numpy(float)
    y = np.log(df["lambda_per_bp"].to_numpy(float))
    sizes = df["size_bp"].to_numpy(float)

    forms: dict[str, FormResult] = {}
    for form in LAMBDA_FORMS.values():
        if form.n_params >= n:
            continue
        p = _fit_form(form, t, y)
        rss = float(np.sum((np.log(form(t, p)) - y) ** 2))
        k = form.n_params
        aicc = n * np.log(max(rss, 1e-300) / n) + 2 * k
        if n - k - 1 > 0:
            aicc += 2 * k * (k + 1) / (n - k - 1)
        loo_log, loo_rel = [], []
        if n > k + 1:
            for i in range(n):
                keep = np.arange(n) != i
                pi = _fit_form(form, t[keep], y[keep])
                loo_log.append(np.log(form(t[i : i + 1], pi))[0] - y[i])
                fitted = _fitted_sizes(df.iloc[[i]], form, pi, settings)[0]
                loo_rel.append(fitted / sizes[i] - 1)
        fitted_all = _fitted_sizes(df, form, p, settings)
        forms[form.name] = FormResult(
            name=form.name,
            formula=form.formula,
            params=p,
            rss=rss,
            aicc=float(aicc),
            loo_rmse_log=float(np.sqrt(np.mean(np.square(loo_log)))) if loo_log else np.nan,
            loo_size_rel_errors=np.array(loo_rel),
            max_abs_rel_error=float(np.max(np.abs(fitted_all / sizes - 1))),
        )

    chosen = min(forms, key=lambda name: forms[name].aicc) if model == "auto" else model
    if chosen not in forms:
        raise ValueError(f"model {chosen!r} not available (have {list(forms)})")
    form = LAMBDA_FORMS[chosen]
    p = forms[chosen].params
    df["fitted_size_bp"] = _fitted_sizes(df, form, p, settings)
    df["rel_error"] = df["fitted_size_bp"] / df["size_bp"] - 1

    # Residual bootstrap in ln(lambda), residuals inflated for the fitted degrees of freedom.
    rng = np.random.default_rng(seed)
    fit_log = np.log(form(t, p))
    resid = (y - fit_log) * np.sqrt(n / max(n - form.n_params, 1))
    boot_params = np.empty((n_boot, form.n_params))
    for b in range(n_boot):
        y_star = fit_log + rng.choice(resid, size=n, replace=True)
        boot_params[b] = _fit_form(form, t, y_star)
    boot_resid = rng.choice(resid, size=n_boot, replace=True)
    return Calibration(settings, df, forms, chosen, boot_params, boot_resid, notes)
