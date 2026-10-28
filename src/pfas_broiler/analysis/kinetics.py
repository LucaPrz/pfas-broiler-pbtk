"""Transfer factors and elimination kinetics at slaughter (point estimates).

Under continuous exposure from day 1 to slaughter at the mean feed intake:

    TF   transfer factor        = C_tissue / C_feed                       (kg/kg)
    BTF  biotransfer factor     = C_tissue / daily feed-borne intake       (d/kg)
                                = TF / feed intake on the slaughter day
    TR   transfer rate          = amount in tissue / total amount ingested (%)

Elimination descriptors from the rate matrix T at slaughter age:

    t1/2 whole-body half-life   = ln 2 / |lambda_min|, lambda_min the slowest
                                  eigenvalue of T (one value per compound)
    MRT  mean residence time    = stomach column of (-T)^-1: mean time a molecule
                                  entering with the feed spends in a compartment
"""

from __future__ import annotations

import numpy as np

from ..config import SLAUGHTER_DAY
from ..exposure import mean_feed_intake
from ..model import COMPARTMENTS, INDEX, ModelParameters, rate_matrix
from ..physiology import physiology
from .feed_limits import concentration_per_unit_feed

TISSUES = ("muscle", "liver", "kidney")


def transfer_factors(params: ModelParameters, day: int = SLAUGHTER_DAY) -> dict[str, dict]:
    """TF, BTF and TR (%) per edible tissue."""
    feed = mean_feed_intake(day)  # kg per bird and day, days 1..day
    out = {}
    for tissue in TISSUES:
        tf = concentration_per_unit_feed(params, tissue, day)
        out[tissue] = {
            "tf": tf,
            "btf": tf / feed[-1],
            "tr_percent": 100.0 * tf * physiology(float(day)).volume[tissue] / feed.sum(),
        }
    return out


def whole_body_half_life(params: ModelParameters, day: int = SLAUGHTER_DAY) -> float:
    """Whole-body elimination half-life (d) from the slowest eigenvalue of T."""
    eigenvalues = np.linalg.eigvals(rate_matrix(params, physiology(float(day))))
    decay = eigenvalues[(np.abs(eigenvalues.imag) <= 1e-8) & (eigenvalues.real < 0)].real
    return float(np.log(2.0) / np.min(-decay))


def mean_residence_times(params: ModelParameters, day: int = SLAUGHTER_DAY) -> dict[str, float]:
    """Mean residence time (d) per compartment and for the whole body."""
    fundamental = -np.linalg.inv(rate_matrix(params, physiology(float(day))))
    from_feed = fundamental[:, INDEX["stomach"]]
    mrt = {c: float(from_feed[INDEX[c]]) for c in COMPARTMENTS}
    mrt["whole_body"] = float(from_feed.sum())
    return mrt
