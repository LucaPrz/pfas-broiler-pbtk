"""PBTK model for PFAS in broiler chickens.

Compartments (state = amount in µg): stomach (including crop), liver, plasma,
kidney, muscle and rest of body. The model is linear, dA/dt = T(t) A + u(t),
with a time-dependent rate matrix T(t) because the bird grows.

Gut-liver pool
--------------
The intestine is not a separate state. It is assumed to be in instantaneous
partition equilibrium with the liver (enterohepatic circulation),
C_intestine = P_il * C_liver, so the intestine holds r = P_il * V_int / V_liver
times the liver amount. Of any amount entering the gut-liver pool, the fraction
g = 1 / (1 + r) is in the liver. Fluxes into the pool and the liver's outflow
term are therefore scaled by g. Because V_int / V_liver is constant, g is
constant and the pool conserves mass as the bird grows.

Elimination is first order from plasma (k_elim). The intake enters the stomach,
which empties into the gut-liver pool with rate k_sto.

Numerical solution
------------------
The equations are solved exactly on daily steps: T is evaluated at the start
of each day and the intake is held constant over the day, so each step is
A(t + 1) = A_ss + exp(T) (A(t) - A_ss) with the step's steady state A_ss = -T^-1 u.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .partitions import PartitionCoefficients, partition_coefficients
from .physiology import Physiology, physiology

COMPARTMENTS: tuple[str, ...] = ("stomach", "liver", "plasma", "kidney", "muscle", "rest")
INDEX: dict[str, int] = {name: i for i, name in enumerate(COMPARTMENTS)}

#: Stomach (+ crop) emptying rate, 1/d, from the gastrointestinal mean retention
#: time of 9.25 h in broilers (Wang et al., 2021): 24 / 9.25.
K_STO = 2.6


@dataclass(frozen=True)
class ModelParameters:
    """Compound-specific model parameters.

    k_elim : first-order elimination rate from plasma, 1/d (fitted).
    P_il   : intestine:liver partition coefficient (fitted).
    """

    k_elim: float
    P_il: float
    partition: PartitionCoefficients
    k_sto: float = K_STO

    @classmethod
    def for_compound(
        cls, compound: str, k_elim: float, P_il: float, isomer: str = "Linear"
    ) -> ModelParameters:
        """Parameters with the compound's partition coefficients."""
        return cls(k_elim=k_elim, P_il=P_il, partition=partition_coefficients(compound, isomer))

    def liver_fraction(self, phys: Physiology) -> float:
        """Fraction g of the gut-liver pool that is in the liver."""
        r = self.P_il * phys.volume["intestine"] / phys.volume["liver"]
        return 1.0 / (1.0 + r)


def rate_matrix(params: ModelParameters, phys: Physiology) -> np.ndarray:
    """Rate matrix T (1/d) with dA/dt = T A + u; column = source, row = target."""
    i = INDEX
    v, q, pc = phys.volume, phys.flow, params.partition
    g = params.liver_fraction(phys)
    T = np.zeros((len(COMPARTMENTS), len(COMPARTMENTS)))

    # Stomach empties into the gut-liver pool; the fraction g reaches the liver.
    T[i["stomach"], i["stomach"]] = -params.k_sto
    T[i["liver"], i["stomach"]] = params.k_sto * g

    # Liver <-> plasma. Outflow depends on the liver amount itself (not scaled);
    # the pool's loss and the return flow from plasma are scaled by g.
    hepatic_clearance = q["hepatic"] / (pc.liver * v["liver"])
    T[i["liver"], i["liver"]] = -hepatic_clearance * g
    T[i["plasma"], i["liver"]] = hepatic_clearance
    T[i["liver"], i["plasma"]] = q["hepatic"] / v["plasma"] * g

    # Plasma: outflow to all organs plus elimination.
    total_flow = q["hepatic"] + q["kidney"] + q["muscle"] + q["rest"]
    T[i["plasma"], i["plasma"]] = -total_flow / v["plasma"] - params.k_elim

    # Flow-limited exchange between plasma and kidney, muscle and rest.
    for organ, p in (("muscle", pc.muscle), ("kidney", pc.kidney), ("rest", pc.rest)):
        T[i[organ], i[organ]] = -q[organ] / (p * v[organ])
        T[i[organ], i["plasma"]] = q[organ] / v["plasma"]
        T[i["plasma"], i[organ]] = q[organ] / (p * v[organ])
    return T


@dataclass(frozen=True)
class Simulation:
    """Result of :func:`simulate`.

    days       : (n + 1,) study days 0..n.
    amounts    : (n + 1, 6) amount per compartment (µg), columns as COMPARTMENTS.
    eliminated : (n + 1,) cumulative amount eliminated (µg).
    """

    params: ModelParameters
    days: np.ndarray
    amounts: np.ndarray
    eliminated: np.ndarray

    def amount(self, compartment: str) -> np.ndarray:
        """Amount (µg) in one compartment on each day."""
        return self.amounts[:, INDEX[compartment]]

    def concentration(self, compartment: str) -> np.ndarray:
        """Concentration (µg/L, approx. µg/kg) in one compartment on each day."""
        volumes = np.array([physiology(float(d)).volume[compartment] for d in self.days])
        return self.amount(compartment) / volumes

    def intestine_amount(self) -> np.ndarray:
        """Amount (µg) held in the intestine part of the gut-liver pool."""
        r = np.array(
            [
                self.params.P_il
                * physiology(float(d)).volume["intestine"]
                / physiology(float(d)).volume["liver"]
                for d in self.days
            ]
        )
        return r * self.amount("liver")

    def body_burden(self) -> np.ndarray:
        """Total amount in the bird (µg), including the intestine."""
        return self.amounts.sum(axis=1) + self.intestine_amount()


def simulate(params: ModelParameters, daily_intake: np.ndarray) -> Simulation:
    """Simulate from day 0 (no PFAS in the bird) with the given daily intake.

    ``daily_intake[i]`` is the intake (µg/d) during day i + 1; see :mod:`exposure`.
    """
    n_comp = len(COMPARTMENTS)
    n_days = len(daily_intake)
    amounts = np.zeros((n_days + 1, n_comp))
    eliminated = np.zeros(n_days + 1)
    identity = np.eye(n_comp)

    for day in range(n_days):
        T = rate_matrix(params, physiology(float(day)))
        u = np.zeros(n_comp)
        u[INDEX["stomach"]] = daily_intake[day]
        step = expm(T)
        steady = -np.linalg.solve(T, u)
        amounts[day + 1] = steady + step @ (amounts[day] - steady)

        # Time integral of the plasma amount over the day gives the eliminated mass.
        integral = steady + (step - identity) @ np.linalg.solve(T, amounts[day] - steady)
        eliminated[day + 1] = eliminated[day] + params.k_elim * integral[INDEX["plasma"]]

    return Simulation(
        params=params, days=np.arange(n_days + 1), amounts=amounts, eliminated=eliminated
    )
