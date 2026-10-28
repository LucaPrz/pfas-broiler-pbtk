"""Physical consistency of the model."""

import numpy as np
import pytest

from pfas_broiler import ModelParameters, physiology, simulate, study_intake
from pfas_broiler.config import GROUPS
from pfas_broiler.growth import GOMPERTZ_PARAMETERS, fit_gompertz
from pfas_broiler.model import INDEX, rate_matrix
from pfas_broiler.physiology import KIDNEY_ALLOMETRY, LIVER_ALLOMETRY, fit_organ_allometry

PARAMS = ModelParameters.for_compound("PFOS", k_elim=0.144, P_il=2.47)


@pytest.mark.parametrize("day", [0, 10, 35])
def test_rate_matrix_conserves_mass_except_elimination(day):
    phys = physiology(float(day))
    T = rate_matrix(PARAMS, phys)
    # Total mass is w @ A: the liver state stands for the whole gut-liver pool,
    # which holds 1/g times the liver amount.
    w = np.ones(T.shape[0])
    w[INDEX["liver"]] = 1.0 / PARAMS.liver_fraction(phys)
    expected = np.zeros(T.shape[0])
    expected[INDEX["plasma"]] = -PARAMS.k_elim
    np.testing.assert_allclose(w @ T, expected, atol=1e-10)


@pytest.mark.parametrize("group", GROUPS)
def test_mass_balance(group):
    intake = study_intake(group, "PFOS")
    sim = simulate(PARAMS, intake)
    total_in = intake.sum()
    accounted = sim.body_burden()[-1] + sim.eliminated[-1]
    assert accounted == pytest.approx(total_in, rel=1e-10)


def test_no_intake_no_pfas():
    sim = simulate(PARAMS, np.zeros(35))
    assert np.all(sim.amounts == 0.0)


def test_volumes_and_flows_are_positive():
    for day in range(36):
        phys = physiology(float(day))
        assert all(v > 0 for v in phys.volume.values())
        assert all(q > 0 for q in phys.flow.values())
        assert sum(phys.volume.values()) == pytest.approx(phys.body_weight_kg)


def test_growth_curve_reproduces_from_data():
    np.testing.assert_allclose(fit_gompertz(), GOMPERTZ_PARAMETERS, rtol=1e-6)


@pytest.mark.parametrize(
    "organ, published", [("liver", LIVER_ALLOMETRY), ("kidney", KIDNEY_ALLOMETRY)]
)
def test_organ_allometry_reproduces_from_data(organ, published):
    a, b = fit_organ_allometry(organ)
    assert a == pytest.approx(published[0], abs=5e-7)
    assert b == pytest.approx(published[1], abs=5e-6)
