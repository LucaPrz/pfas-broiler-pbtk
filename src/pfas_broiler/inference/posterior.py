"""Exact posterior of the fitted parameters on a dense grid.

With one or two fitted parameters, the posterior can be computed exactly by
evaluating the likelihood on a dense grid instead of sampling it with MCMC.
Priors are uniform in log10 of each parameter and uniform in sigma; sigma is
integrated out numerically. Posterior samples are drawn from the grid cells
(with uniform jitter within the cell). The forward model is evaluated in batches
with JAX (:mod:`pfas_broiler.batched`).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..batched import BatchedModel
from ..model import INDEX
from .likelihood import SIGMA_BOUNDS, Observations, log_likelihood_grid, observations

PARAMETERS = ("k_elim", "P_il")
_LOG_EPS = 1e-10


class CellPredictor:
    """log10 predictions in the observation cells for a batch of parameter sets."""

    def __init__(self, model: BatchedModel, obs: Observations):
        group_index = {g: i for i, g in enumerate(model.scenarios)}
        g = jnp.asarray([group_index[group] for group, _, _ in obs.cells])
        d = jnp.asarray([day for _, day, _ in obs.cells])
        c = jnp.asarray([INDEX[comp] for _, _, comp in obs.cells])

        def predict(theta):
            conc = model.concentrations(theta)[g, d, c]
            return jnp.log10(jnp.maximum(conc, 0.0) + _LOG_EPS)

        self._batch = jax.jit(jax.vmap(predict))

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        """theta (B, 2) -> log10 predictions (B, n_cells)."""
        return np.asarray(self._batch(jnp.asarray(theta)))


@dataclass(frozen=True)
class GridPosterior:
    """Posterior on a regular grid in log10 parameter space.

    Each grid point is the centre of a cell of width ``step`` (log10), and the
    posterior mass of a cell is spread uniformly across it (in log10). Quantiles,
    means and samples all follow this convention.

    names     : fitted parameters (grid axes).
    log_grids : grid points per parameter, log10 scale.
    weights   : posterior probability per grid cell, shape (npts,) * len(names).
    """

    names: tuple[str, ...]
    log_grids: tuple[np.ndarray, ...]
    weights: np.ndarray

    def step(self, name: str) -> float:
        """Cell width of one parameter's grid (log10)."""
        grid = self.log_grids[self.names.index(name)]
        return float(grid[1] - grid[0])

    def marginal(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(values at the cell centres, probabilities) of one parameter's marginal."""
        j = self.names.index(name)
        others = tuple(k for k in range(len(self.names)) if k != j)
        return 10.0 ** self.log_grids[j], self.weights.sum(axis=others)

    def quantiles(self, name: str, probs) -> np.ndarray:
        """Marginal quantiles; the cumulative distribution is linear within a cell."""
        _, probabilities = self.marginal(name)
        grid, half = self.log_grids[self.names.index(name)], 0.5 * self.step(name)
        edges = np.append(grid - half, grid[-1] + half)
        cumulative = np.concatenate([[0.0], np.cumsum(probabilities)])
        return 10.0 ** np.interp(probs, cumulative / cumulative[-1], edges)

    def mean(self, name: str) -> float:
        """Posterior mean (exact for mass spread uniformly in log10 within a cell)."""
        values, probabilities = self.marginal(name)
        half = 0.5 * self.step(name) * np.log(10.0)
        within_cell = np.sinh(half) / half  # mean of 10**u, u uniform on the cell
        return float(np.sum(values * probabilities) * within_cell)

    def save(self, path) -> None:
        """Save the grid to a NumPy .npz file."""
        np.savez_compressed(
            path,
            names=np.array(self.names),
            weights=self.weights,
            **{f"log_grid_{n}": g for n, g in zip(self.names, self.log_grids, strict=True)},
        )

    @classmethod
    def load(cls, path) -> GridPosterior:
        with np.load(path) as f:
            names = tuple(str(n) for n in f["names"])
            return cls(names, tuple(f[f"log_grid_{n}"] for n in names), f["weights"])

    def sample(self, n: int = 4000, seed: int = 0) -> dict[str, np.ndarray]:
        """Draw grid cells by their probability, with uniform jitter within a cell."""
        rng = np.random.default_rng(seed)
        flat = rng.choice(self.weights.size, size=n, p=self.weights.ravel())
        cell = np.unravel_index(flat, self.weights.shape)
        samples = {}
        for j, name in enumerate(self.names):
            grid = self.log_grids[j]
            jitter = rng.uniform(-0.5, 0.5, n) * (grid[1] - grid[0])
            samples[name] = 10.0 ** (grid[cell[j]] + jitter)
        return samples


def grid_posterior(
    compound: str,
    center: dict[str, float],
    fixed: dict[str, float] | None = None,
    npts: int = 240,
    decades: float = 2.0,
    n_sigma: int = 80,
    batch_size: int = 1024,
) -> GridPosterior:
    """Compute the grid posterior of the fitted parameters of one compound.

    The grid spans ``center`` +/- ``decades`` (log10) for every parameter that is
    not in ``fixed``; ``center`` is normally the point estimate.
    """
    fixed = dict(fixed or {})
    names = tuple(p for p in PARAMETERS if p not in fixed)
    log_grids = tuple(
        np.linspace(np.log10(center[p]) - decades, np.log10(center[p]) + decades, npts)
        for p in names
    )
    mesh = np.meshgrid(*log_grids, indexing="ij")
    points = {p: 10.0 ** m.ravel() for p, m in zip(names, mesh, strict=True)}
    n_points = mesh[0].size
    theta = np.stack(
        [points[p] if p in points else np.full(n_points, fixed[p]) for p in PARAMETERS], axis=1
    )

    obs = observations(compound)
    predict = CellPredictor(BatchedModel.for_study_groups(compound, groups=tuple(obs.groups)), obs)
    sigma = np.linspace(SIGMA_BOUNDS[0], SIGMA_BOUNDS[1], n_sigma)
    ll = np.empty((n_points, n_sigma))
    for start in range(0, n_points, batch_size):
        mu = predict(theta[start : start + batch_size])
        ll[start : start + batch_size] = log_likelihood_grid(mu, obs, sigma)

    # Integrate sigma out (uniform prior), then normalise.
    peak = ll.max()
    with np.errstate(divide="ignore"):  # far tails integrate to 0 -> weight 0
        log_w = np.log(np.trapezoid(np.exp(ll - peak), sigma, axis=1)) + peak
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()
    return GridPosterior(names, log_grids, weights.reshape((npts,) * len(names)))
