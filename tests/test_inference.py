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


def _coarse_posterior(compound):
    """Coarse grid (25 points, +/- 0.25 decades) around the fitted point."""
    center = {p: FITTED.loc[compound, p] for p in ("k_elim", "P_il")}
    return grid_posterior(compound, center, _fixed(compound), npts=25, decades=0.25)


@pytest.mark.parametrize("compound", ["PFOS", "PFNA", "PFHxS"])
def test_grid_posterior_samples_match_original(compound):
    post = _coarse_posterior(compound)
    samples = post.sample(500, seed=0)
    expected = pd.read_csv(REFERENCE / f"posterior_samples_npts25_{compound}.csv")
    for name in post.names:
        np.testing.assert_allclose(samples[name], expected[name], rtol=1e-8)


@pytest.mark.parametrize("compound", ["PFOS", "PFNA"])
def test_grid_quantiles_and_mean_agree_with_samples(compound):
    post = _coarse_posterior(compound)
    samples = post.sample(400_000, seed=1)
    probs = [0.025, 0.25, 0.5, 0.75, 0.975]
    for name in post.names:
        np.testing.assert_allclose(
            post.quantiles(name, probs), np.quantile(samples[name], probs), rtol=3e-3
        )
        assert post.mean(name) == pytest.approx(samples[name].mean(), rel=1e-3)
