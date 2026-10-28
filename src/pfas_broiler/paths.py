"""Locations of the input data and of the results.

Both default to folders in the repository and can be overridden with the
environment variables ``PFAS_BROILER_DATA`` and ``PFAS_BROILER_RESULTS``.
"""

from __future__ import annotations

import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PFAS_BROILER_DATA", REPOSITORY / "data"))
RESULTS_DIR = Path(os.environ.get("PFAS_BROILER_RESULTS", REPOSITORY / "results"))

POINT_ESTIMATES = RESULTS_DIR / "point_estimates.csv"
POSTERIOR_DIR = RESULTS_DIR / "posterior"


def posterior_samples_file(compound: str) -> Path:
    return POSTERIOR_DIR / f"samples_{compound}.csv"
