"""The package reproduces the results of the original analysis code.

The files in ``tests/reference/`` were produced by the code used for the
manuscript, before it was restructured into this package.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfas_broiler import (
    ModelParameters,
    partition_coefficients,
    physiology,
    scenario_intake,
    simulate,
    study_intake,
)
from pfas_broiler.config import COMPOUNDS, GROUPS, SLAUGHTER_DAY
from pfas_broiler.model import COMPARTMENTS

REFERENCE = Path(__file__).parent / "reference"
RTOL = 1e-9


@pytest.fixture(scope="module")
def fitted():
    df = pd.read_csv(REFERENCE / "fitted_point_nll.csv").set_index("compound")
    return {c: (df.loc[c, "k_elim"], df.loc[c, "P_il"]) for c in df.index}


def test_physiology():
    ref = pd.read_csv(REFERENCE / "physiology.csv")
    for _, row in ref.iterrows():
        phys = physiology(float(row["day"]))
        assert phys.body_weight_kg == pytest.approx(row["body_weight_kg"], rel=RTOL)
        for name, value in phys.volume.items():
            assert value == pytest.approx(row[f"V_{name}"], rel=RTOL), name
        for name, value in phys.flow.items():
            key = "Q_hepatic" if name == "hepatic" else f"Q_{name}"
            assert value == pytest.approx(row[key], rel=RTOL), name


@pytest.mark.parametrize("compound", COMPOUNDS)
def test_partition_coefficients(compound):
    ref = pd.read_csv(REFERENCE / "partition_coefficients.csv").set_index("compound")
    pc = partition_coefficients(compound)
    for tissue in ("liver", "kidney", "muscle", "rest"):
        assert getattr(pc, tissue) == pytest.approx(ref.loc[compound, tissue], rel=RTOL)


@pytest.mark.parametrize("compound", COMPOUNDS)
def test_study_group_trajectories(compound, fitted):
    ref = pd.read_csv(REFERENCE / "trajectories_study_groups.csv")
    params = ModelParameters.for_compound(compound, *fitted[compound])
    for group in GROUPS:
        sim = simulate(params, study_intake(group, compound))
        expected = (
            ref[(ref["compound"] == compound) & (ref["group"] == group)]
            .pivot(index="day", columns="compartment", values="amount_ug")[list(COMPARTMENTS)]
            .to_numpy()
        )
        np.testing.assert_allclose(sim.amounts, expected, rtol=RTOL, atol=1e-14)


@pytest.mark.parametrize("compound", COMPOUNDS)
def test_slaughter_concentration_per_unit_feed(compound, fitted):
    """Basis of the tolerable feed levels: tissue concentration at slaughter
    after feeding 1 µg/kg from day 1 to slaughter."""
    ref = pd.read_csv(REFERENCE / "slaughter_conc_per_unit_feed.csv")
    ref = ref[ref["compound"] == compound].set_index("compartment")["conc_per_unit_feed"]
    sim = simulate(ModelParameters.for_compound(compound, *fitted[compound]), scenario_intake(1.0))
    for tissue, expected in ref.items():
        assert sim.concentration(tissue)[SLAUGHTER_DAY] == pytest.approx(expected, rel=RTOL)
