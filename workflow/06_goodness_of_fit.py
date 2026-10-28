"""Step 6: goodness of fit of the posterior-predictive median to the measurements.

Input:  results/predictions/study_groups_<compound>.csv (step 3)
Output: results/tables/goodness_of_fit.csv
        results/predictions/observed_vs_predicted_<compound>.csv
Runtime: seconds.

    python workflow/06_goodness_of_fit.py
"""

import pandas as pd

from pfas_broiler.analysis.goodness_of_fit import metrics, observed_vs_predicted
from pfas_broiler.config import COMPOUNDS
from pfas_broiler.paths import PREDICTIONS_DIR, TABLES_DIR, predictions_file


def main() -> None:
    rows = []
    for compound in COMPOUNDS:
        table = observed_vs_predicted(compound, pd.read_csv(predictions_file(compound)))
        table.to_csv(
            PREDICTIONS_DIR / f"observed_vs_predicted_{compound}.csv",
            index=False,
            float_format="%.6g",
        )
        rows.append({"compound": compound, **metrics(table)})
    summary = pd.DataFrame(rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "goodness_of_fit.csv"
    summary.to_csv(out, index=False, float_format="%.6g")
    print(summary.to_string(index=False, float_format="%.4g"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
