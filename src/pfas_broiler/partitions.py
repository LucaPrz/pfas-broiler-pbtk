"""Tissue:plasma partition coefficients.

Partition coefficients are derived from the study data: for every animal with a
same-day serum and tissue measurement above the LOQ, the ratio tissue/serum is
computed, and the median over animals is used per tissue and analyte. The median
is used because the ratios are right-skewed; it reproduces the partition
coefficients reported in the experimental part of the study.

Some cells are not informative because the tissue is below the LOQ at the low
dietary dose. These are filled by read-across or from the literature, see
:data:`READ_ACROSS` and :data:`LITERATURE_VALUES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import data

#: Tissue partition coefficients borrowed from a homologous compound, used where
#: the compound's own tissue data are below the LOQ:
#: (compound, tissue) -> donor compound.
READ_ACROSS: dict[tuple[str, str], str] = {
    ("PFOA", "muscle"): "PFNA",  # C8 <- C9 carboxylate
    ("PFDA", "muscle"): "PFNA",  # C10 <- C9 carboxylate
}

#: Literature-based partition coefficients for PFHxS, whose tissues are all below
#: the LOQ. Liver and kidney: Aarab et al. (2026), laying hens, environmental
#: exposure (tissue / serum). Muscle: PFOS muscle partition (0.130) scaled by the
#: geometric mean of the PFHxS/PFOS ratios in liver and kidney (0.382).
LITERATURE_VALUES: dict[tuple[str, str], float] = {
    ("PFHxS", "liver"): 0.207,  # 0.42 µg/kg / 2.03 µg/L
    ("PFHxS", "kidney"): 0.438,  # 0.89 µg/kg / 2.03 µg/L
    ("PFHxS", "muscle"): 0.050,  # 0.130 x 0.382
}

_TISSUES = ("liver", "kidney", "muscle")


@dataclass(frozen=True)
class PartitionCoefficients:
    """Tissue:plasma partition coefficients (dimensionless)."""

    liver: float
    kidney: float
    muscle: float
    rest: float


def tissue_serum_ratios() -> pd.DataFrame:
    """Per-animal tissue:serum ratios for all analytes.

    Only pairs with both values above the LOQ (recorded > 0) give a ratio; below-LOQ
    values would bias the median downwards in proportion to the censoring rate.
    """
    df = data.tissue_concentrations()
    keys = ["animal", "day", "compound", "isomer"]
    serum = df[df["matrix"] == "serum"][keys + ["concentration"]]
    tissue = df[df["matrix"] != "serum"]
    merged = tissue.merge(serum.rename(columns={"concentration": "serum"}), on=keys, how="left")
    detected = (merged["concentration"] > 0) & (merged["serum"] > 0)
    merged["ratio"] = np.where(detected, merged["concentration"] / merged["serum"], np.nan)
    return merged


@lru_cache(maxsize=1)
def median_ratios() -> pd.DataFrame:
    """Median tissue:serum ratio per tissue, compound and isomer."""
    ratios = tissue_serum_ratios()
    return ratios.groupby(["matrix", "compound", "isomer"])["ratio"].median().reset_index()


def _median_ratio(compound: str, isomer: str, tissue: str) -> float:
    table = median_ratios()
    row = table[
        (table["compound"] == compound) & (table["isomer"] == isomer) & (table["matrix"] == tissue)
    ]
    value = float(row["ratio"].iloc[0]) if not row.empty else np.nan
    if not np.isfinite(value):
        raise ValueError(f"No partition coefficient for {compound} {isomer} {tissue}")
    return value


def partition_coefficients(compound: str, isomer: str = "Linear") -> PartitionCoefficients:
    """Partition coefficients used in the model for one analyte.

    Precedence: literature value > read-across > the compound's own data. The
    rest-of-body compartment uses the muscle value: it consists mainly of lean
    carcass tissue, and PFAS bind to proteins rather than to fat.
    """
    values = {}
    for tissue in _TISSUES:
        if (compound, tissue) in LITERATURE_VALUES:
            values[tissue] = LITERATURE_VALUES[(compound, tissue)]
        elif (compound, tissue) in READ_ACROSS:
            values[tissue] = _median_ratio(READ_ACROSS[(compound, tissue)], isomer, tissue)
        else:
            values[tissue] = _median_ratio(compound, isomer, tissue)
    return PartitionCoefficients(**values, rest=values["muscle"])
