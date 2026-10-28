"""Daily PFAS intake (µg per bird per day).

Intake vectors are indexed by study day: element ``i`` is the intake during day
``i + 1``, i.e. during the time interval [i, i + 1) of the simulation.
"""

from __future__ import annotations

import numpy as np

from . import data
from .config import EXPOSURE_STOP_DAY, SLAUGHTER_DAY


def study_intake(
    group: str, compound: str, isomer: str = "Linear", n_days: int = SLAUGHTER_DAY
) -> np.ndarray:
    """Intake of one study group: measured feed intake x diet concentration.

    The intake is zero after the group's exposure stop day (depuration).
    """
    feed = data.feed_intake()
    feed = feed[feed["group"] == group].set_index("day")["feed_kg_per_bird"]
    concentration = data.feed_concentration(compound, isomer)
    days = np.arange(1, n_days + 1)
    intake = np.array([feed.get(d, 0.0) * concentration for d in days])
    intake[days > EXPOSURE_STOP_DAY[group]] = 0.0
    return intake


def mean_feed_intake(n_days: int = SLAUGHTER_DAY) -> np.ndarray:
    """Daily feed intake (kg per bird) averaged over all study groups."""
    feed = data.feed_intake().groupby("day")["feed_kg_per_bird"].mean()
    days = np.arange(1, n_days + 1)
    missing = set(days) - set(feed.index)
    if missing:
        raise ValueError(f"No feed intake recorded for days {sorted(missing)}")
    return feed.loc[days].to_numpy(dtype=float)


def scenario_intake(
    feed_concentration_ug_per_kg: float,
    exposure_days: int = SLAUGHTER_DAY,
    n_days: int = SLAUGHTER_DAY,
) -> np.ndarray:
    """Intake for a hypothetical diet concentration fed from day 1 to ``exposure_days``.

    Uses the mean feed intake of the study birds.
    """
    intake = mean_feed_intake(n_days) * feed_concentration_ug_per_kg
    intake[np.arange(1, n_days + 1) > exposure_days] = 0.0
    return intake
