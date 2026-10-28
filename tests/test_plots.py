"""Every figure can be drawn from the shipped results (smoke tests)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from pfas_broiler import plots  # noqa: E402
from pfas_broiler.analysis import goodness_of_fit, predictive  # noqa: E402
from pfas_broiler.analysis.estimates import posterior_samples  # noqa: E402
from pfas_broiler.inference.posterior import GridPosterior, grid_posterior  # noqa: E402

PRODUCTION = Path(__file__).parent / "reference" / "production"


@pytest.fixture(scope="module")
def predictions():
    return predictive.study_group_predictions("PFOS", posterior_samples("PFOS", PRODUCTION)[:200])


def test_growth_curve():
    plots.growth_curve()
    assert plots.growth_r2() == pytest.approx(0.9547, abs=5e-4)


def test_observed_vs_predicted_and_concentration_time(predictions):
    table = goodness_of_fit.observed_vs_predicted("PFOS", predictions)
    plots.observed_vs_predicted({"PFOS": table}, {"PFOS": goodness_of_fit.metrics(table)})
    plots.concentration_time("PFOS", predictions)


@pytest.mark.parametrize("compound, fixed", [("PFOS", None), ("PFHxS", {"P_il": 2.47})])
def test_posterior_figure_and_grid_roundtrip(compound, fixed, tmp_path):
    post = grid_posterior(compound, {"k_elim": 0.2, "P_il": 2.47}, fixed, npts=15, decades=0.3)
    post.save(tmp_path / "grid.npz")
    loaded = GridPosterior.load(tmp_path / "grid.npz")
    assert loaded.names == post.names
    np.testing.assert_array_equal(loaded.weights, post.weights)
    plots.posterior(loaded)
