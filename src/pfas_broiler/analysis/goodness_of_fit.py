"""Goodness of fit: observed vs posterior-predictive median concentrations.

Every animal with a value above the LOQ is compared with the predictive median
of its group, sampling day and tissue. Cells in which every animal is below the
LOQ are kept as one "censored" row each (for the figures); they do not enter the
metrics. Metrics on the log10 scale:

    R2    1 - SS_res / SS_tot of log10(obs) vs log10(pred)
    GMFE  10 ** mean |log10(pred / obs)|      geometric mean fold error
    bias  mean log10(pred / obs); bias_fold = 10 ** |bias|
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import data
from ..config import MATRIX_TO_COMPARTMENT, TISSUES

_EPS = 1e-10


def observed_vs_predicted(compound: str, predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-animal observed and predicted concentrations.

    predictions: output of :func:`pfas_broiler.analysis.predictive.study_group_predictions`.
    Columns: group, day, compartment, observed (NaN if censored), predicted, censored.
    """
    df = data.tissue_concentrations()
    df = df[(df["compound"] == compound) & (df["isomer"] == "Linear")].copy()
    df["compartment"] = df["matrix"].map(MATRIX_TO_COMPARTMENT)
    df = df[df["compartment"].isin(TISSUES)].dropna(subset=["concentration"])
    predicted = predictions.set_index(["group", "day", "compartment"])["median"]

    detected = df[df["concentration"] > 0]
    rows = [
        {
            "group": r.group,
            "day": int(r.day),
            "compartment": r.compartment,
            "observed": float(r.concentration),
            "predicted": float(predicted[(r.group, int(r.day), r.compartment)]),
            "censored": False,
        }
        for r in detected.itertuples()
    ]
    cell_max = df.groupby(["group", "day", "compartment"], as_index=False)["concentration"].max()
    for r in cell_max[cell_max["concentration"] <= 0].itertuples():
        rows.append(
            {
                "group": r.group,
                "day": int(r.day),
                "compartment": r.compartment,
                "observed": np.nan,
                "predicted": float(predicted[(r.group, int(r.day), r.compartment)]),
                "censored": True,
            }
        )
    return pd.DataFrame(rows)


def metrics(table: pd.DataFrame) -> dict[str, float]:
    """R2, GMFE and bias of the detected (non-censored) rows."""
    d = table[~table["censored"] & (table["observed"] > 0) & (table["predicted"] >= 0)]
    log_obs = np.log10(d["observed"].to_numpy() + _EPS)
    log_pred = np.log10(d["predicted"].to_numpy() + _EPS)
    log_ratio = np.log10((d["predicted"].to_numpy() + _EPS) / (d["observed"].to_numpy() + _EPS))
    ss_res = np.sum((log_obs - log_pred) ** 2)
    ss_tot = np.sum((log_obs - log_obs.mean()) ** 2)
    bias = float(np.mean(log_ratio))
    return {
        "r2": float(1.0 - ss_res / ss_tot),
        "gmfe": float(10.0 ** np.mean(np.abs(log_ratio))),
        "bias_log10": bias,
        "bias_fold": float(10.0 ** abs(bias)),
        "n_detected": len(d),
        "n_censored_cells": int(table["censored"].sum()),
    }
