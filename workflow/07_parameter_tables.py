"""Step 7: parameter tables (growth, physiology, partition coefficients, fitted parameters).

Input:  results/posterior/samples_<compound>.csv, results/tables/goodness_of_fit.csv
Output: results/tables/growth_parameters.csv
        results/tables/physiology.csv              (volumes in % body weight, days 1/15/35)
        results/tables/partition_coefficients.csv  (median, IQR, n; value used in the model)
        results/tables/fitted_parameters.csv       (posterior median and 95 % interval)
Runtime: seconds.

    python workflow/07_parameter_tables.py
"""

import numpy as np
import pandas as pd

from pfas_broiler.analysis.estimates import posterior_samples
from pfas_broiler.config import COMPOUNDS, P_IL_READ_ACROSS
from pfas_broiler.growth import GOMPERTZ_PARAMETERS
from pfas_broiler.partitions import (
    LITERATURE_VALUES,
    READ_ACROSS,
    partition_coefficients,
    tissue_serum_ratios,
)
from pfas_broiler.paths import TABLES_DIR
from pfas_broiler.physiology import physiology
from pfas_broiler.plots import growth_r2

TISSUES = ("liver", "kidney", "muscle")


def growth_table() -> pd.DataFrame:
    a, k, t0 = GOMPERTZ_PARAMETERS
    return pd.DataFrame(
        [
            {"parameter": "asymptotic body weight A", "value": a, "unit": "g"},
            {"parameter": "growth-rate constant C", "value": k, "unit": "1/d"},
            {"parameter": "inflection time t0", "value": t0, "unit": "d"},
            {"parameter": "R2 (individual body weights)", "value": growth_r2(), "unit": "-"},
        ]
    )


def physiology_table(days=(1, 15, 35)) -> pd.DataFrame:
    rows = []
    for day in days:
        phys = physiology(float(day))
        rows.append(
            {
                "day": day,
                "body_weight_g": 1000.0 * phys.body_weight_kg,
                **{
                    f"{c}_percent_bw": 100.0 * v / phys.body_weight_kg
                    for c, v in phys.volume.items()
                },
            }
        )
    return pd.DataFrame(rows)


def partition_table() -> pd.DataFrame:
    ratios = tissue_serum_ratios()
    ratios = ratios[ratios["isomer"] == "Linear"].dropna(subset=["ratio"])
    rows = []
    for compound in COMPOUNDS:
        used = partition_coefficients(compound)
        for tissue in TISSUES:
            r = ratios[(ratios["compound"] == compound) & (ratios["matrix"] == tissue)]["ratio"]
            if (compound, tissue) in LITERATURE_VALUES:
                source = "literature"
            elif (compound, tissue) in READ_ACROSS:
                source = f"read-across from {READ_ACROSS[(compound, tissue)]}"
            else:
                source = "this study"
            rows.append(
                {
                    "compound": compound,
                    "tissue": tissue,
                    "value_used": getattr(used, tissue),
                    "source": source,
                    "median": r.median() if len(r) else np.nan,
                    "q25": r.quantile(0.25) if len(r) else np.nan,
                    "q75": r.quantile(0.75) if len(r) else np.nan,
                    "n": len(r),
                }
            )
    return pd.DataFrame(rows)


def fitted_parameter_table() -> pd.DataFrame:
    fit = pd.read_csv(TABLES_DIR / "goodness_of_fit.csv").set_index("compound")
    rows = []
    for compound in COMPOUNDS:
        theta = posterior_samples(compound)
        row = {"compound": compound}
        for j, name in enumerate(("k_elim", "P_il")):
            lo, med, hi = np.percentile(theta[:, j], [2.5, 50, 97.5])
            row |= {f"{name}_median": med, f"{name}_ci_lower": lo, f"{name}_ci_upper": hi}
        row["P_il_from"] = P_IL_READ_ACROSS.get(compound, "")
        row |= fit.loc[compound, ["r2", "gmfe", "n_detected", "n_censored_cells"]].to_dict()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in [
        ("growth_parameters", growth_table()),
        ("physiology", physiology_table()),
        ("partition_coefficients", partition_table()),
        ("fitted_parameters", fitted_parameter_table()),
    ]:
        table.to_csv(TABLES_DIR / f"{name}.csv", index=False, float_format="%.6g")
        print(f"\n{name}\n{table.to_string(index=False, float_format='%.4g')}")


if __name__ == "__main__":
    main()
