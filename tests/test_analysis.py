"""Reported model outputs reproduce the original analysis code.

Inputs are the production point estimates and posterior samples
(``tests/reference/production/``); the expected values were computed from them
with the original code.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jax")

from pfas_broiler.analysis import feed_limits, goodness_of_fit, kinetics, predictive  # noqa: E402
from pfas_broiler.analysis.estimates import point_parameters, posterior_samples  # noqa: E402

PRODUCTION = Path(__file__).parent / "reference" / "production"
RTOL = 1e-8


@pytest.fixture(scope="module")
def predictions():
    return {
        c: predictive.study_group_predictions(c, posterior_samples(c, PRODUCTION))
        for c in ("PFOS", "PFNA", "PFOA", "PFDA", "PFHxS")
    }


@pytest.mark.parametrize("compound", ["PFOS", "PFNA", "PFOA", "PFHxS"])
def test_tolerable_feed_levels(compound):
    ref = pd.read_csv(PRODUCTION / "feed_limits.csv")
    ref = ref[ref["compound"] == compound].set_index("tissue")
    table = feed_limits.tolerable_feed_levels(
        compound, point_parameters(compound, PRODUCTION), posterior_samples(compound, PRODUCTION)
    ).set_index("tissue")
    for column, ref_column in [
        ("conc_per_unit_feed", "conc_per_unit_feed"),
        ("tfl", "tfl_point"),
        ("tfl_median", "tfl_median"),
        ("tfl_ci_lower", "tfl_ci_lower"),
        ("tfl_ci_upper", "tfl_ci_upper"),
    ]:
        np.testing.assert_allclose(table[column], ref.loc[table.index, ref_column], rtol=RTOL)


@pytest.mark.parametrize("compound", ["PFOS", "PFNA"])
def test_transfer_factors(compound):
    ref = pd.read_csv(PRODUCTION / "transfer_factors.csv")
    ref = ref[ref["compound"] == compound].set_index("tissue")
    result = kinetics.transfer_factors(point_parameters(compound, PRODUCTION))
    for tissue, values in result.items():
        assert values["tf"] == pytest.approx(ref.loc[tissue, "tf"], rel=1e-9)
        assert values["btf"] == pytest.approx(ref.loc[tissue, "btf"], rel=1e-9)
        assert values["tr_percent"] == pytest.approx(ref.loc[tissue, "tr_pct"], rel=1e-9)


@pytest.mark.parametrize("compound", ["PFOS", "PFNA"])
def test_half_life_and_mean_residence_time(compound):
    ref = pd.read_csv(PRODUCTION / "kinetics.csv").set_index("compound").loc[compound]
    params = point_parameters(compound, PRODUCTION)
    assert kinetics.whole_body_half_life(params) == pytest.approx(ref["t_half"], rel=1e-9)
    mrt = kinetics.mean_residence_times(params)
    for name in ("muscle", "liver", "kidney", "whole_body"):
        assert mrt[name] == pytest.approx(ref[f"mrt_{name}"], rel=1e-9)


@pytest.mark.parametrize("compound", ["PFOS", "PFNA", "PFOA", "PFDA", "PFHxS"])
def test_posterior_predictive(compound, predictions):
    ref = pd.read_csv(PRODUCTION / "predictive_study_groups.csv")
    ref = ref[ref["Compound"] == compound].rename(
        columns={"Group": "group", "Compartment": "compartment", "Time": "day"}
    )
    ref["day"] = ref["day"].astype(int)
    merged = ref.merge(predictions[compound], on=["group", "compartment", "day"])
    assert len(merged) == len(ref)
    for column, ref_column in [
        ("median", "Pred_Median"),
        ("ci_lower", "CI_Lower"),
        ("ci_upper", "CI_Upper"),
    ]:
        np.testing.assert_allclose(merged[column], merged[ref_column], rtol=RTOL, atol=1e-12)


@pytest.mark.parametrize("compound", ["PFOS", "PFNA", "PFOA", "PFDA", "PFHxS"])
def test_goodness_of_fit(compound, predictions):
    ref = pd.read_csv(PRODUCTION / "goodness_of_fit.csv").set_index("Compound").loc[compound]
    m = goodness_of_fit.metrics(
        goodness_of_fit.observed_vs_predicted(compound, predictions[compound])
    )
    assert m["r2"] == pytest.approx(ref["R2"], rel=RTOL)
    assert m["gmfe"] == pytest.approx(ref["GMFE"], rel=RTOL)
    assert m["bias_log10"] == pytest.approx(ref["Bias_log10"], rel=1e-6)
    assert m["n_detected"] == ref["N_detected"]
    assert m["n_censored_cells"] == ref["N_censored"]
