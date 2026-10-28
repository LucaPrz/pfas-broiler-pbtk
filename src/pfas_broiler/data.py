"""Loaders for the input data in ``data/``.

All loaders are cached and return a fresh copy, so callers may modify the result.
The data directory defaults to ``<repository>/data`` and can be overridden with
the ``PFAS_BROILER_DATA`` environment variable.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("PFAS_BROILER_DATA", Path(__file__).resolve().parents[2] / "data"))


@cache
def _read_csv(relative_path: str) -> pd.DataFrame:
    path = DATA_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def tissue_concentrations() -> pd.DataFrame:
    """Measured PFAS concentrations, one row per animal, matrix and analyte.

    Columns: animal, group, day, matrix (serum/liver/kidney/muscle), compound,
    isomer, concentration (µg/kg wet weight; µg/L for serum). Values below the
    LOQ are recorded as 0.
    """
    return _read_csv("raw/tissue_concentrations.csv").copy()


def feed_intake() -> pd.DataFrame:
    """Daily feed intake per bird.

    Columns: group, day (1 = day of placement), feed_kg_per_bird.
    """
    df = _read_csv("raw/feed_intake.csv").copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["group", "date"])
    df["day"] = df.groupby("group").cumcount() + 1
    df["feed_kg_per_bird"] = df["feed_intake_g"] / df["n_birds"] / 1000.0
    return df[["group", "day", "feed_kg_per_bird"]].reset_index(drop=True)


def feed_concentration(compound: str, isomer: str = "Linear") -> float:
    """PFAS concentration in the contaminated diet (µg/kg feed, 88 % dry matter)."""
    df = _read_csv("raw/feed_concentration.csv")
    row = df[(df["compound"] == compound) & (df["isomer"] == isomer)]
    if row.empty:
        raise KeyError(f"No feed concentration for {compound} {isomer}")
    return float(row["concentration_ug_per_kg"].iloc[0])


def body_weights() -> pd.DataFrame:
    """Individual body weights. Columns: group, animal, date, day, body_weight_g."""
    return _read_csv("raw/body_weight.csv").copy()


def organ_weights() -> pd.DataFrame:
    """Body, liver and kidney weights (g) of birds weighed at slaughter."""
    return _read_csv("raw/organ_weights.csv").copy()


def ross308_meat_yield() -> pd.DataFrame:
    """Deboned meat yield of male Ross 308 broilers (% of live weight).

    Columns: live_weight_kg, total_meat_percent.
    """
    return _read_csv("literature/ross308_male_meat_yield.csv").copy()
