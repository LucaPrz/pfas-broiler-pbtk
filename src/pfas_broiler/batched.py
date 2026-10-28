"""The PBTK model evaluated for many parameter sets at once (JAX).

This is not a second implementation of the model. The rate matrix is affine in
k_elim and in the liver fraction g = 1 / (1 + P_il V_int / V_liver), so its
constant part and both slopes are taken from
:func:`pfas_broiler.model.rate_matrix_from_fraction`; the code here only combines
them and solves the linear ODE on daily steps exactly as
:func:`pfas_broiler.model.simulate` does. The tests check that both agree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm

from .config import GROUPS, SLAUGHTER_DAY
from .exposure import study_intake
from .model import COMPARTMENTS, INDEX, rate_matrix_from_fraction
from .partitions import partition_coefficients
from .physiology import physiology

jax.config.update("jax_enable_x64", True)


class BatchedModel:
    """Concentrations for a set of intake scenarios and many parameter sets.

    ``intakes`` maps a scenario name to a daily intake vector (µg/d, see
    :mod:`pfas_broiler.exposure`); all vectors must have the same length n_days.
    ``concentrations(theta)`` takes theta = (k_elim, P_il) and returns an array
    (scenario, day, compartment) of concentrations (µg/L) for days 0..n_days.
    """

    def __init__(self, compound: str, intakes: dict[str, np.ndarray], isomer: str = "Linear"):
        self.scenarios = tuple(intakes)
        n_days = len(next(iter(intakes.values())))
        self.n_days = n_days
        pc = partition_coefficients(compound, isomer)
        phys = [physiology(float(day)) for day in range(n_days + 1)]

        # Affine decomposition per day: T = T0 + k_elim * Dk + g * Dg.
        t0 = np.stack([rate_matrix_from_fraction(pc, p, 0.0, 0.0) for p in phys[:-1]])
        dk = np.stack([rate_matrix_from_fraction(pc, p, 1.0, 0.0) for p in phys[:-1]]) - t0
        dg = np.stack([rate_matrix_from_fraction(pc, p, 0.0, 1.0) for p in phys[:-1]]) - t0
        self._t0, self._dk, self._dg = jnp.asarray(t0), jnp.asarray(dk), jnp.asarray(dg)
        self._intestine_to_liver = jnp.asarray(
            [p.volume["intestine"] / p.volume["liver"] for p in phys[:-1]]
        )

        u = np.zeros((len(intakes), n_days, len(COMPARTMENTS)))
        for i, intake in enumerate(intakes.values()):
            u[i, :, INDEX["stomach"]] = intake
        self._intake = jnp.asarray(u)
        self._volumes = jnp.asarray([[p.volume[c] for c in COMPARTMENTS] for p in phys])

        self.concentrations = jax.jit(self._concentrations)
        self._batch = jax.jit(jax.vmap(self._concentrations))

    @classmethod
    def for_study_groups(
        cls,
        compound: str,
        groups: tuple[str, ...] = GROUPS,
        n_days: int = SLAUGHTER_DAY,
        isomer: str = "Linear",
    ) -> BatchedModel:
        """Scenarios = the study groups, each with its measured intake."""
        intakes = {g: study_intake(g, compound, isomer, n_days) for g in groups}
        return cls(compound, intakes, isomer)

    def _concentrations(self, theta: jnp.ndarray) -> jnp.ndarray:
        k_elim, p_il = theta[0], theta[1]
        g = 1.0 / (1.0 + self._intestine_to_liver * p_il)  # (day,)
        rate = self._t0 + k_elim * self._dk + g[:, None, None] * self._dg

        def run_scenario(intake):
            def step(amount, inputs):
                T, u = inputs
                steady = -jnp.linalg.solve(T, u)
                new = steady + expm(T) @ (amount - steady)
                return new, new

            start = jnp.zeros(len(COMPARTMENTS))
            _, amounts = jax.lax.scan(step, start, (rate, intake))
            return jnp.concatenate([start[None, :], amounts])

        amounts = jax.vmap(run_scenario)(self._intake)  # (scenario, day, compartment)
        return amounts / self._volumes[None, :, :]

    def concentrations_batch(self, theta: np.ndarray, batch_size: int = 500) -> np.ndarray:
        """theta (B, 2) -> concentrations (B, scenario, day, compartment)."""
        out = [
            np.asarray(self._batch(jnp.asarray(theta[i : i + batch_size])))
            for i in range(0, len(theta), batch_size)
        ]
        return np.concatenate(out)
