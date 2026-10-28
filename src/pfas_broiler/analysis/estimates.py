"""Loading the fitted parameters produced by workflow steps 1 and 2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..model import ModelParameters

PARAMETERS = ("k_elim", "P_il")


def point_estimate(compound: str, results_dir: Path | None = None) -> dict[str, float]:
    """Point estimate {k_elim, P_il} of one compound."""
    path = (results_dir or paths.RESULTS_DIR) / paths.POINT_ESTIMATES.name
    row = pd.read_csv(path).set_index("compound").loc[compound]
    return {p: float(row[p]) for p in PARAMETERS}


def point_parameters(compound: str, results_dir: Path | None = None) -> ModelParameters:
    """Model parameters at the point estimate."""
    return ModelParameters.for_compound(compound, **point_estimate(compound, results_dir))


def posterior_samples(compound: str, results_dir: Path | None = None) -> np.ndarray:
    """Posterior samples as an array (n, 2) of (k_elim, P_il).

    Parameters that were not fitted (P_il for PFHxS) are set to their fixed value.
    """
    directory = (results_dir or paths.RESULTS_DIR) / paths.POSTERIOR_DIR.name
    samples = pd.read_csv(directory / f"samples_{compound}.csv")
    fixed = point_estimate(compound, results_dir)
    columns = [
        samples[p].to_numpy(dtype=float) if p in samples else np.full(len(samples), fixed[p])
        for p in PARAMETERS
    ]
    return np.stack(columns, axis=1)
