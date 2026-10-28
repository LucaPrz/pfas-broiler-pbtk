"""Posterior predictive concentrations in the study groups.

Every posterior sample is run through the model for each group's own intake;
the median and the 95 % credible interval of the predicted concentration are
reported per group, compartment and day. (This is the predictive median, not
the prediction at the median parameters.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..batched import BatchedModel
from ..config import GROUPS
from ..model import COMPARTMENTS


def study_group_predictions(compound: str, theta: np.ndarray) -> pd.DataFrame:
    """Median and 95 % interval per (group, compartment, day).

    theta: posterior samples, shape (n, 2) of (k_elim, P_il).
    Columns: group, compartment, day, median, ci_lower, ci_upper (µg/L).
    """
    model = BatchedModel.for_study_groups(compound, GROUPS)
    conc = model.concentrations_batch(theta)  # (sample, group, day, compartment)
    median, lower, upper = np.percentile(conc, [50, 2.5, 97.5], axis=0)

    rows = []
    for gi, group in enumerate(model.scenarios):
        for ci, compartment in enumerate(COMPARTMENTS):
            for day in range(model.n_days + 1):
                rows.append(
                    {
                        "group": group,
                        "compartment": compartment,
                        "day": day,
                        "median": median[gi, day, ci],
                        "ci_lower": lower[gi, day, ci],
                        "ci_upper": upper[gi, day, ci],
                    }
                )
    return pd.DataFrame(rows)
