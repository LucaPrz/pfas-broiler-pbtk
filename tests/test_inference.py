"""Parameter estimation reproduces the original analysis code."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfas_broiler import ModelParameters
from pfas_broiler.config import P_IL_READ_ACROSS
from pfas_broiler.inference import Objective, observations
from pfas_broiler.inference.fit import predicted_log10

REFERENCE = Path(__file__).parent / "reference"
FITTED = pd.read_csv(REFERENCE / "fitted_point_nll.csv").set_index("compound")


def _fixed(compound):
    return {"P_il": FITTED.loc[compound, "P_il"]} if compound in P_IL_READ_ACROSS else {}


@pytest.mark.parametrize("compound", FITTED.index)
def test_negative_log_likelihood_at_fitted_point(compound):
    objective = Objective.for_compound(compound, _fixed(compound))
    values = np.array([FITTED.loc[compound, p] for p in objective.free])
    assert objective(values) == pytest.approx(FITTED.loc[compound, "neg_log_lik"], rel=1e-9)


def test_observations_pfos():
    obs = observations("PFOS")
    # 5 groups x 2 sampling days x 4 tissues, except E5 (sampled once, on day 35)
    assert obs.n_cells == 36
    assert obs.n_detected + obs.n_censored == 192


jax = pytest.importorskip("jax")
from pfas_broiler.inference.posterior import (  # noqa: E402
    BatchedModel,
    CellPredictor,
    grid_posterior,
)


@pytest.mark.parametrize("compound", ["PFOS", "PFNA", "PFHxS"])
def test_batched_model_matches_numpy_model(compound):
    obs = observations(compound)
    predict = CellPredictor(BatchedModel(compound, groups=tuple(obs.groups)), obs)
    for scale in (0.5, 1.0, 2.0):
        k_elim, p_il = FITTED.loc[compound, "k_elim"] * scale, FITTED.loc[compound, "P_il"]
        numpy = predicted_log10(compound, ModelParameters.for_compound(compound, k_elim, p_il), obs)
        batched = predict(np.array([[k_elim, p_il]]))[0]
        np.testing.assert_allclose(batched, numpy, atol=1e-9)


@pytest.mark.parametrize("compound", ["PFOS", "PFNA", "PFHxS"])
def test_grid_posterior_matches_original(compound):
    """Coarse grid (25 points, +/- 0.25 decades) against the original code."""
    ref = pd.read_csv(REFERENCE / "posterior_grid_npts25_decades025.csv")
    ref = ref[ref["compound"] == compound].set_index("parameter")
    center = {p: FITTED.loc[compound, p] for p in ("k_elim", "P_il")}
    post = grid_posterior(compound, center, _fixed(compound), npts=25, decades=0.25)
    for name in post.names:
        expected = ref.loc[name, ["q025", "q25", "q50", "q75", "q975"]].to_numpy(dtype=float)
        q = post.quantiles(name, [0.025, 0.25, 0.5, 0.75, 0.975])
        np.testing.assert_allclose(q, expected, rtol=1e-8)
        assert post.mean(name) == pytest.approx(ref.loc[name, "mean"], rel=1e-8)

    samples = post.sample(500, seed=0)
    expected = pd.read_csv(REFERENCE / f"posterior_samples_npts25_{compound}.csv")
    for name in post.names:
        np.testing.assert_allclose(samples[name], expected[name], rtol=1e-8)
