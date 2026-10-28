"""Body-weight growth curve (Gompertz) fitted to the study birds."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from . import data

#: Gompertz parameters (A [g], k [1/d], t0 [d]) used in the model, as fitted by
#: :func:`fit_gompertz`. They are stored rather than refitted at run time because
#: the optimiser result differs in the 8th digit between SciPy versions; the
#: tests check that the fit reproduces them.
GOMPERTZ_PARAMETERS = (3722.192916417706, 0.07460860042001156, 20.675636975640955)

#: Body weight is clamped to this range (g). The lower bound is about hatch
#: weight; the fitted curve stays above it, so the clamp is a safety net only.
_MIN_WEIGHT_G = 40.0
_MAX_WEIGHT_G = 6000.0


def gompertz(day: np.ndarray, asymptote: float, rate: float, inflection: float) -> np.ndarray:
    """Gompertz curve W(t) = A * exp(-exp(-k * (t - t0)))."""
    return asymptote * np.exp(-np.exp(-rate * (day - inflection)))


def fit_gompertz() -> tuple[float, float, float]:
    """Fit (A, k, t0) to the median body weight per day of all study birds.

    The fit minimises squared error in log weight, so that early (light) and late
    (heavy) weights count equally; a fit on raw grams is dominated by the
    slaughter weights and underestimates early growth.
    """
    weights = data.body_weights().dropna(subset=["day", "body_weight_g"])
    median = weights.groupby("day")["body_weight_g"].median()
    days = median.index.to_numpy(dtype=float)
    grams = median.to_numpy(dtype=float)

    initial = [max(1.2 * grams.max(), 1500.0), 0.07, float(np.median(days))]
    params, _ = curve_fit(
        lambda t, a, k, t0: np.log(gompertz(t, a, k, t0)),
        days,
        np.log(grams),
        p0=initial,
        maxfev=200_000,
        bounds=([500.0, 0.01, 0.0], [8000.0, 1.0, 45.0]),
    )
    return float(params[0]), float(params[1]), float(params[2])


def body_weight_g(day: float) -> float:
    """Predicted body weight (g) on a given study day."""
    weight = float(gompertz(np.array([float(day)]), *GOMPERTZ_PARAMETERS)[0])
    return min(max(weight, _MIN_WEIGHT_G), _MAX_WEIGHT_G)
