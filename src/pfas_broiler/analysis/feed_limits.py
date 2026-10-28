"""Tolerable feed levels (TFL).

The model is linear in the intake, so the tissue concentration at slaughter is
proportional to the feed concentration. With C1 the tissue concentration at
slaughter after feeding 1 µg/kg from day 1 to slaughter (mean feed intake of the
study birds), the tolerable feed level of a tissue is

    TFL = ML / C1   (µg/kg feed)

where ML is the EU maximum level for that tissue. The 95 % credible interval is
obtained by computing the TFL for every posterior sample.

TFLs are derived for tissues with an EU maximum level whose partition coefficient
was measured in this study. Tissues with a read-across partition coefficient are
reported as indicative only; tissues with a literature value get no TFL (see
:func:`tfl_tissues`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..batched import BatchedModel
from ..config import EU_MAXIMUM_LEVEL, SLAUGHTER_DAY
from ..exposure import scenario_intake
from ..model import INDEX, simulate
from ..partitions import partition_basis


def tfl_tissues(compound: str) -> dict[str, str]:
    """Tissues that get a TFL -> basis of their partition coefficient
    ("measured" or "read-across"; the latter is indicative only)."""
    return {
        tissue: partition_basis(compound, tissue)
        for tissue in EU_MAXIMUM_LEVEL.get(compound, {})
        if partition_basis(compound, tissue) != "literature"
    }


def concentration_per_unit_feed(params, tissue: str, day: int = SLAUGHTER_DAY) -> float:
    """Tissue concentration (µg/kg) at ``day`` per µg/kg feed, for one parameter set."""
    return float(simulate(params, scenario_intake(1.0, n_days=day)).concentration(tissue)[day])


def tolerable_feed_levels(
    compound: str, point, theta: np.ndarray, day: int = SLAUGHTER_DAY
) -> pd.DataFrame:
    """TFL per tissue (see :func:`tfl_tissues`).

    point: ModelParameters at the point estimate; theta: posterior samples (n, 2).
    Columns: compound, tissue, partition_basis, ml, conc_per_unit_feed, tfl (point
    estimate), tfl_median, tfl_ci_lower, tfl_ci_upper, binding (lowest TFL among the
    tissues with measured partition coefficients).
    """
    tissues = tfl_tissues(compound)
    model = BatchedModel(compound, {"unit_feed": scenario_intake(1.0, n_days=day)})
    conc = model.concentrations_batch(theta)[:, 0, day, :]  # (sample, compartment)

    rows = []
    for tissue, basis in tissues.items():
        ml = EU_MAXIMUM_LEVEL[compound][tissue]
        c = conc[:, INDEX[tissue]]
        tfl = ml / c[np.isfinite(c) & (c > 0)]
        lower, median, upper = np.percentile(tfl, [2.5, 50, 97.5])
        c_point = concentration_per_unit_feed(point, tissue, day)
        rows.append(
            {
                "compound": compound,
                "tissue": tissue,
                "partition_basis": basis,
                "ml": ml,
                "conc_per_unit_feed": c_point,
                "tfl": ml / c_point,
                "tfl_median": median,
                "tfl_ci_lower": lower,
                "tfl_ci_upper": upper,
            }
        )
    table = pd.DataFrame(rows)
    measured = table["partition_basis"] == "measured"
    table["binding"] = measured & (table["tfl"] == table.loc[measured, "tfl"].min())
    return table
