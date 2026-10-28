"""Observation model with left-censoring of values below the LOQ.

Each animal i belongs to a prediction cell c(i) = (group, day, tissue), and
mu_c = log10 of the model concentration in that cell. With sigma the residual
standard deviation in log10 units:

    detected:    log10(y_i) ~ Normal(mu_c(i), sigma)
    below LOQ:   contributes  log Phi((log10(LOQ) - mu_c(i)) / sigma)

A below-LOQ value is thus used as the information "the true value is below the
LOQ", rather than being set to zero or dropped. Cells in which every animal is
below the LOQ still constrain the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

from .. import data
from ..config import LOQ, TISSUES

LOG10_LOQ = float(np.log10(LOQ))

#: Range of sigma (log10 units) for profiling and for the posterior grid.
SIGMA_BOUNDS: tuple[float, float] = (0.02, 1.2)

_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))
#: Measured matrix -> model compartment (serum is modelled as plasma).
_MATRIX_TO_COMPARTMENT = {
    "serum": "plasma",
    "liver": "liver",
    "kidney": "kidney",
    "muscle": "muscle",
}


@dataclass(frozen=True)
class Observations:
    """Measurements of one analyte, mapped onto prediction cells.

    cells          : unique (group, day, compartment) cells, in prediction order.
    detected_cell  : cell index of each detected value.
    detected_log10 : log10 concentration of each detected value.
    censored_cell  : cell index of each value below the LOQ.
    """

    cells: list[tuple[str, int, str]]
    detected_cell: np.ndarray
    detected_log10: np.ndarray
    censored_cell: np.ndarray

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    @property
    def n_detected(self) -> int:
        return int(self.detected_cell.size)

    @property
    def n_censored(self) -> int:
        return int(self.censored_cell.size)

    @property
    def groups(self) -> list[str]:
        return sorted({group for group, _, _ in self.cells})


def observations(compound: str, isomer: str = "Linear") -> Observations:
    """Observations of one analyte in serum (as plasma), liver, kidney and muscle."""
    df = data.tissue_concentrations()
    df = df[(df["compound"] == compound) & (df["isomer"] == isomer)].copy()
    df["compartment"] = df["matrix"].map(_MATRIX_TO_COMPARTMENT)
    df = df[df["compartment"].isin(TISSUES)].dropna(subset=["concentration"])

    cells = sorted(set(zip(df["group"], df["day"].astype(int), df["compartment"], strict=True)))
    index = {cell: i for i, cell in enumerate(cells)}
    cell_of_row = np.array(
        [
            index[(g, int(d), c)]
            for g, d, c in zip(df["group"], df["day"], df["compartment"], strict=True)
        ],
        dtype=int,
    )
    detected = (df["concentration"] > 0).to_numpy()
    return Observations(
        cells=cells,
        detected_cell=cell_of_row[detected],
        detected_log10=np.log10(df.loc[detected, "concentration"].to_numpy(dtype=float)),
        censored_cell=cell_of_row[~detected],
    )


def log_likelihood(mu: np.ndarray, obs: Observations, sigma: float) -> float:
    """Log-likelihood for predicted log10 concentrations ``mu`` (one per cell)."""
    z = (obs.detected_log10 - mu[obs.detected_cell]) / sigma
    ll = -0.5 * float(np.sum(z * z)) - obs.n_detected * (np.log(sigma) + _LOG_SQRT_2PI)
    if obs.n_censored:
        ll += float(np.sum(log_ndtr((LOG10_LOQ - mu[obs.censored_cell]) / sigma)))
    return ll


def profile_negative_log_likelihood(mu: np.ndarray, obs: Observations) -> float:
    """Negative log-likelihood minimised over sigma (objective of the point fit).

    Under censoring sigma has no closed form; it is optimised over log10(sigma).
    """
    result = minimize_scalar(
        lambda log_sigma: -log_likelihood(mu, obs, 10.0**log_sigma),
        bounds=(log10(SIGMA_BOUNDS[0]), log10(SIGMA_BOUNDS[1])),
        method="bounded",
    )
    return float(result.fun)


def log_likelihood_grid(mu: np.ndarray, obs: Observations, sigma: np.ndarray) -> np.ndarray:
    """Log-likelihood for a batch of predictions and a grid of sigma values.

    mu has shape (B, n_cells); the result has shape (B, len(sigma)).
    """
    residual = obs.detected_log10[None, :] - mu[:, obs.detected_cell]
    ssr = np.sum(residual * residual, axis=1)
    out = np.empty((mu.shape[0], sigma.size))
    for k, s in enumerate(sigma):
        ll = -0.5 * ssr / (s * s) - obs.n_detected * (np.log(s) + _LOG_SQRT_2PI)
        if obs.n_censored:
            ll = ll + np.sum(log_ndtr((LOG10_LOQ - mu[:, obs.censored_cell]) / s), axis=1)
        out[:, k] = ll
    return out
