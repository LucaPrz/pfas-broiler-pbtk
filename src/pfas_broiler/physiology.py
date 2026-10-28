"""Time-dependent broiler physiology: compartment volumes and plasma flows.

Volumes are in L (tissue density taken as 1 kg/L), flows in L/d. Everything
scales with the body weight predicted by the growth curve (:mod:`growth`).

Sources
-------
- Wang et al. (2021), physiological parameters for PBPK models in chicken:
  hematocrit, cardiac output, blood volume, stomach and intestine volume
  fractions, blood-flow fractions.
- In-house organ weights of the study birds (``data/raw/organ_weights.csv``):
  allometric liver and kidney weights, see :func:`fit_organ_allometry`.
- Aviagen (2022), Ross 308 performance objectives: deboned meat yield of males
  (``data/literature/ross308_male_meat_yield.csv``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import data
from .growth import body_weight_g

HEMATOCRIT = 0.307
#: Cardiac output, L/(h kg) blood.
CARDIAC_OUTPUT_BLOOD = 10.2
#: Blood volume as a fraction of body weight.
BLOOD_VOLUME_FRACTION = 0.0483
STOMACH_VOLUME_FRACTION = 0.035

#: Allometric organ weights, organ_g = a * body_weight_g ** b, fitted by ordinary
#: least squares in log-log space to all 50 study birds with organ weights.
#: The values are reproduced by :func:`fit_organ_allometry` (see tests).
LIVER_ALLOMETRY = (0.105729, 0.78505)
KIDNEY_ALLOMETRY = (0.011535, 0.91784)

#: The intestine volume is a fixed multiple of the liver volume, so that the
#: intestine:liver partition ratio in the lumped gut-liver pool stays constant
#: as the bird grows (see :mod:`model`). The multiple reproduces the Wang et al.
#: (2021) intestine fraction of 3.5 % body weight at the day-35 liver fraction of
#: 1.94 % body weight.
INTESTINE_TO_LIVER_VOLUME = 1.80

#: Fractions of cardiac plasma output; "rest" receives the remainder.
FLOW_FRACTION = {
    "stomach": 0.0196,
    "intestine": 0.1161,
    "hepatic": 0.2526,
    "kidney": 0.2012,
    "muscle": 0.0764,
}


@dataclass(frozen=True)
class Physiology:
    """Physiology of an average bird on one study day."""

    body_weight_kg: float
    volume: dict[str, float]  # L, keys: plasma, liver, stomach, intestine, kidney, muscle, rest
    flow: dict[str, float]  # L/d plasma, keys: stomach, intestine, hepatic, kidney, muscle, rest


def _allometric_fraction(coefficients: tuple[float, float], body_weight_kg: float) -> float:
    """Organ weight as a fraction of body weight from organ_g = a * bw_g ** b."""
    a, b = coefficients
    bw_g = body_weight_kg * 1000.0
    return a * bw_g**b / bw_g


def muscle_fraction(body_weight_kg: float) -> float:
    """Deboned breast and leg meat as a fraction of live weight.

    Linear interpolation of the male Ross 308 yield table; weights outside the
    table are clamped to its end points.
    """
    table = data.ross308_meat_yield()
    percent = np.interp(
        body_weight_kg,
        table["live_weight_kg"].to_numpy(dtype=float),
        table["total_meat_percent"].to_numpy(dtype=float),
    )
    return float(percent) / 100.0


@lru_cache(maxsize=256)
def physiology(day: float) -> Physiology:
    """Return volumes and flows for an average bird on ``day``."""
    bw = body_weight_g(day) / 1000.0
    plasma_flow = bw * CARDIAC_OUTPUT_BLOOD * (1.0 - HEMATOCRIT) * 24.0
    liver = _allometric_fraction(LIVER_ALLOMETRY, bw) * bw

    volume = {
        # Plasma = blood volume x (1 - hematocrit); serum was measured.
        "plasma": BLOOD_VOLUME_FRACTION * bw * (1.0 - HEMATOCRIT),
        "liver": liver,
        "stomach": STOMACH_VOLUME_FRACTION * bw,
        "intestine": INTESTINE_TO_LIVER_VOLUME * liver,
        "kidney": _allometric_fraction(KIDNEY_ALLOMETRY, bw) * bw,
        "muscle": muscle_fraction(bw) * bw,
    }
    volume["rest"] = bw - sum(
        volume[k] for k in ("intestine", "liver", "kidney", "muscle", "plasma", "stomach")
    )

    flow = {name: fraction * plasma_flow for name, fraction in FLOW_FRACTION.items()}
    flow["rest"] = plasma_flow - sum(
        flow[k] for k in ("hepatic", "stomach", "intestine", "kidney", "muscle")
    )
    return Physiology(body_weight_kg=bw, volume=volume, flow=flow)


def fit_organ_allometry(organ: str) -> tuple[float, float]:
    """Fit organ_g = a * body_weight_g ** b to the measured organ weights.

    ``organ`` is "liver" or "kidney". Returns (a, b).
    """
    weights = data.organ_weights().dropna(subset=["body_weight_g", f"{organ}_g"])
    weights = weights[(weights["body_weight_g"] > 0) & (weights[f"{organ}_g"] > 0)]
    b, log_a = np.polyfit(np.log(weights["body_weight_g"]), np.log(weights[f"{organ}_g"]), 1)
    return float(np.exp(log_a)), float(b)
