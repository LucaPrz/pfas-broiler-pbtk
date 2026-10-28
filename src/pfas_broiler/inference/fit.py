"""Point estimate of the fitted parameters (k_elim, P_il) by maximum likelihood."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution

from ..exposure import study_intake
from ..model import ModelParameters, simulate
from .likelihood import Observations, observations, profile_negative_log_likelihood

#: Search range of each parameter, log10 scale. P_il spans a negligible gut
#: reservoir (carboxylates, ~0.01) to a large one (sulfonates, > 1).
LOG10_BOUNDS = {"k_elim": (-3.0, 2.0), "P_il": (-2.0, 3.0)}

#: Settings of the global optimiser (differential evolution, then L-BFGS-B polish).
OPTIMIZER_SETTINGS = {
    "strategy": "best1bin",
    "popsize": 20,
    "maxiter": 600,
    "tol": 0.01,
    "mutation": (0.5, 1.0),
    "recombination": 0.7,
    "polish": True,
    "updating": "deferred",
    "seed": 42,
}

#: Returned for parameter values at which the model gives no valid prediction.
_INVALID = 1e10
_LOG_EPS = 1e-10


def predicted_log10(compound: str, params: ModelParameters, obs: Observations) -> np.ndarray:
    """log10 model concentration in every observation cell."""
    concentrations = {}
    for group in obs.groups:
        sim = simulate(params, study_intake(group, compound))
        concentrations[group] = {c: sim.concentration(c) for c in {c for _, _, c in obs.cells}}
    pred = np.array([concentrations[g][c][day] for g, day, c in obs.cells])
    return np.log10(pred + _LOG_EPS)


@dataclass
class Objective:
    """Negative profile log-likelihood as a function of the free parameters.

    ``fixed`` holds parameters that are not fitted (e.g. P_il for PFHxS).
    """

    compound: str
    free: list[str]
    fixed: dict[str, float]
    obs: Observations

    @classmethod
    def for_compound(cls, compound: str, fixed: dict[str, float] | None = None) -> Objective:
        fixed = dict(fixed or {})
        free = [p for p in ("k_elim", "P_il") if p not in fixed]
        return cls(compound=compound, free=free, fixed=fixed, obs=observations(compound))

    def parameters(self, values: np.ndarray) -> dict[str, float]:
        return {**dict(zip(self.free, map(float, values), strict=True)), **self.fixed}

    def __call__(self, values: np.ndarray) -> float:
        p = self.parameters(values)
        try:
            mu = predicted_log10(
                self.compound, ModelParameters.for_compound(self.compound, **p), self.obs
            )
        except np.linalg.LinAlgError:
            return _INVALID
        if not np.all(np.isfinite(mu)):
            return _INVALID
        return profile_negative_log_likelihood(mu, self.obs)


@dataclass(frozen=True)
class FitResult:
    compound: str
    parameters: dict[str, float]
    neg_log_likelihood: float
    n_evaluations: int


def fit(compound: str, fixed: dict[str, float] | None = None) -> FitResult:
    """Maximum-likelihood estimate of k_elim and P_il (or of k_elim only, if
    P_il is given in ``fixed``)."""
    objective = Objective.for_compound(compound, fixed)
    result = differential_evolution(
        lambda log_values: objective(10.0**log_values),
        bounds=[LOG10_BOUNDS[p] for p in objective.free],
        **OPTIMIZER_SETTINGS,
    )
    return FitResult(
        compound=compound,
        parameters=objective.parameters(10.0**result.x),
        neg_log_likelihood=float(result.fun),
        n_evaluations=int(result.nfev),
    )
