# Model: equations, calibration, assumptions and limitations

## 1. Cleavage model

Each starting molecule of length `L0` is cut at points that form a **stationary renewal
process** with density `λ(t)` cuts per bp. The spacing between cuts is
`X ~ Gamma(shape = k, rate = kλ)`, so its mean is `1/λ` for every `k`.

* `k = 1` gives a Poisson process: cuts are independent and uniformly random (the default).
* `k > 1` makes spacings more regular and suppresses very short fragments. This is a
  phenomenological stand-in for constraints such as needing two nearby nicks to make a
  double-strand break. `k` is **not** identified by the NEB average-size data. Set it by hand, or
  fit it from an electropherogram (§5).

"Stationary" means the molecule ends fall at random positions in the cut process. With
`f`, `S` the gamma density and survival function, the expected fragment-length spectrum of one
molecule is exact:

| fragments | count density at length `x < L0` |
|---|---|
| interior (cut at both ends) | `λ f(x) (L0 − x)` |
| the two ends of the molecule | `2 λ S(x)` |
| uncut molecule (point mass at `L0`) | `P0 = λ ∫_{L0}^∞ S(a) da` |

Integrating this spectrum gives every statistic in closed form, using incomplete-gamma
functions (`src/ultrashear/analytic.py`):

* **Number average** (total length / number of fragments). The expected number of cuts is `λL0`,
  so `1/L_n = λ + 1/L0`, for any `k`. For HMW DNA the `1/L0` term is negligible. For DNA of only
  1-3 kb it dominates at long times. This is how starting length changes the outcome.
* **Weight (mass) average** `E[Σ lᵢ²] / L0` is the mean length of the fragment containing a
  random base:
  `L_w = L0·P0 + 2λ·I1 − λ·I2 / L0`, with `I_n = ∫_0^{L0} aⁿ S(a) da`.
  For Poisson cuts this reduces to `L_w = (2/λ)·[1 − (1 − e^{−λL0})/(λL0)]`. It tends to
  `(1 + 1/k)/λ` for long DNA (so weight/number = 2 for Poisson) and to `L0` as `λ → 0`.
* **Peak (mode)**: the maximum of mass per unit `ln(length)`, which is what an electropherogram
  with a log-size axis shows. It is blurred by a Gaussian of SD 0.1 in `ln(length)` to mimic
  instrument sizing resolution, so that the spike of uncut molecules does not produce a spurious
  peak. For long DNA the peak is at `(1 + 1/k)/λ`.
* **Fraction in a size window**, by mass and by molecule count.

**Starting-length distribution.** Either fixed, or lognormal with a given *mass-weighted*
median (roughly the peak of a gel or genomic-DNA trace) and coefficient of variation `CV`. With
`σ² = ln(1 + CV²)`, the number-weighted median is `median·e^{−σ²}`. Population statistics are
integrated over the length distribution by quadrature.

**Input amount (assumption).** `t_eff = t·(50 ng / m)^α`. The default `α = 0` means no effect.
The NEB M7634 manual validates 5-250 ng (26 µl DNA in 1X TE, 44 µl reaction) and does not say that
input amount changes fragment size. If you suspect an enzyme-to-DNA ratio effect, set `α > 0`
(lower input fragments faster) and confirm it with your own runs.

## 2. What the measured "average size" means

The NEB sizes are TapeStation region averages. Fluorescence is proportional to DNA mass, so the
region average is a **mass-weighted mean**, and that is the default (`definition="weight"`).
Alternatives are `number` and `mode`. The model is recalibrated under whichever definition you
choose, so all three fit the NEB data equally well. They diverge only when they are extrapolated
to short starting DNA:

| definition | predicted size at 10 / 30 min, L0 = 1.5 kb | L0 = 3 kb | L0 = 10 kb |
|---|---|---|---|
| number | 543 / 294 bp | 663 / 326 bp | 784 / 353 bp |
| weight | 613 / 321 bp | 725 / 343 bp | 808 / 358 bp |
| mode   | 672 / 326 bp | 742 / 344 bp | 809 / 358 bp |

For HMW DNA (50 kb) all three give the NEB curve. The app shows this comparison for your inputs.

## 3. Calibration

Data: 50 ng NA12878 HMW DNA, 5-45 min at 37 °C + 15 min at 65 °C, TapeStation D5000 HS
(`data/neb_calibration.csv`).

1. **Invert** the forward model for each point to get `λᵢ`. The forward model uses the point's
   own starting length (default: the assumed calibration length `L0_ref` = 50 kb, editable) and
   input mass.
2. **Fit** `λ(t)` through the `λᵢ` by least squares in `ln λ`. Four forms with at most two
   parameters are compared:

| form | formula | AICc | LOO RMSE (size) | max in-sample error |
|---|---|---|---|---|
| linear | `λ = a·t` | −27.6 | 24.1 % | 53.3 % |
| **affine (selected)** | `λ = a + b·t` | **−54.7** | **4.4 %** | **5.9 %** |
| power | `λ = a·t^b` | −53.5 | 5.8 % | 6.6 % |
| saturating | `λ = λmax·(1 − e^{−t/τ})` | −40.8 | 11.8 % | 16.7 % |

   (Weight-average definition, `k = 1`, `L0_ref = 50 kb`. Selected parameters:
   `a = 8.17e-4 /bp`, `b = 1.56e-4 /bp/min`.)

   **Why affine.** It has the lowest AICc and the lowest leave-one-out error. Plotting `1/size`
   against time gives a nearly straight line with a positive intercept. The intercept
   (≈ one cut per 1.2 kb at t = 0) can be read as cleavage during setup and the 65 °C
   inactivation step, plus any pre-existing breaks. The power law is almost as good, and its
   predictions differ mainly when extrapolated beyond 45 min. You can select it in the app to see
   the difference. The 9 points do not support more than two parameters.

3. **Uncertainty.** Residual bootstrap in `ln λ` (300 resamples, residuals inflated by
   `√(n/(n−p))`). Each bootstrap fit is paired with a resampled residual to give a 90 %
   **prediction** band, which is what one new measurement would scatter within. The band covers
   model-fit uncertainty only. It does not cover the unvalidated assumptions (starting length,
   input, `k`), lot-to-lot enzyme variation, or differences between instruments.

**Achieved fit:** every NEB point is reproduced within **5.9 %** (residuals −5.4 % to +5.9 %),
and the leave-one-out RMSE is **4.4 %**.

### Adding your own points

Points can carry their own starting length and input. They are inverted with those values and
fitted together with the NEB data, or alone if you untick the NEB data. A point whose size
exceeds what uncut DNA of the stated length would show cannot be inverted, so it is dropped with
a note.

## 4. Monte Carlo engine

Molecules are laid end to end. One stationary gamma renewal process runs along the whole line,
and the molecule boundaries are added as extra cuts. Each molecule therefore sees exactly the
process of §1. The first point is drawn from the equilibrium (forward-recurrence) distribution
`U·Gamma(k+1, kλ)`. A run of 20,000 × 50 kb molecules (~4 M fragments) takes under 1 s. The
tests check that the MC number and weight averages, fraction in window and peak agree with the
closed forms within Monte Carlo error.

## 5. Fitting a trace

`ultrashear.io.fit_trace` fits `λ` and `k` (and a free amplitude) to an exported
electropherogram. It assumes the exported signal is proportional to mass per unit `ln(size)`,
which holds approximately for log-linear sizing (TapeStation, Bioanalyzer). Use
`signal_per="bp"` otherwise. Restrict the fit range to exclude the marker peaks.

## 6. Limitations

* Calibration is one DNA (human, HMW), one input (50 ng), one instrument and one enzyme lot.
* Starting-length and input effects are model extrapolations, and the app labels them as such.
  For degraded or FFPE DNA, NEB recommends shorter times (5-15 min) with optimisation.
* Bead clean-ups remove short fragments (< ~100-150 bp). The model predicts the reaction
  products before any clean-up.
* Fragmentation kinetics may depend on buffer (EDTA, salts) and temperature accuracy. Neither is
  modelled.
* Times outside 5-45 min are extrapolated, and `λ(t)` forms diverge there.

**Use the tool to choose a narrow time range, then run a small calibration:** an HMW control
plus a couple of your own samples at two times that bracket the recommendation.
