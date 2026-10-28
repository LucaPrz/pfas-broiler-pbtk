"""Step 5: transfer factors, half-lives and mean residence times at slaughter.

Input:  results/point_estimates.csv
Output: results/tables/transfer_and_kinetics.csv
Runtime: seconds.

    python workflow/05_transfer_and_kinetics.py
"""

import numpy as np
import pandas as pd

from pfas_broiler.analysis.estimates import point_parameters
from pfas_broiler.analysis.kinetics import (
    mean_residence_times,
    transfer_factors,
    whole_body_half_life,
)
from pfas_broiler.paths import TABLES_DIR

#: Compounds with all partition coefficients measured in this study.
COMPOUNDS = ("PFOS", "PFNA")


def main() -> None:
    rows = []
    for compound in COMPOUNDS:
        params = point_parameters(compound)
        half_life = whole_body_half_life(params)
        mrt = mean_residence_times(params)
        for tissue, values in transfer_factors(params).items():
            rows.append(
                {
                    "compound": compound,
                    "tissue": tissue,
                    **values,
                    "t_half_d": half_life,
                    "t_half_plasma_only_d": np.log(2.0) / params.k_elim,
                    "mrt_d": mrt[tissue],
                    "mrt_whole_body_d": mrt["whole_body"],
                }
            )
    table = pd.DataFrame(rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "transfer_and_kinetics.csv"
    table.to_csv(out, index=False, float_format="%.6g")
    print(table.to_string(index=False, float_format="%.4g"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
