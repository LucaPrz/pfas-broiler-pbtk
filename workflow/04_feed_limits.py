"""Step 4: tolerable feed levels (TFL) per edible tissue with 95 % credible intervals.

Input:  results/point_estimates.csv, results/posterior/samples_<compound>.csv
Output: results/tables/tolerable_feed_levels.csv
Runtime: under a minute.

    python workflow/04_feed_limits.py
"""

import pandas as pd

from pfas_broiler.analysis.estimates import point_parameters, posterior_samples
from pfas_broiler.analysis.feed_limits import tolerable_feed_levels
from pfas_broiler.config import EU_MAXIMUM_LEVEL
from pfas_broiler.paths import TABLES_DIR


def main() -> None:
    tables = [
        tolerable_feed_levels(c, point_parameters(c), posterior_samples(c))
        for c in EU_MAXIMUM_LEVEL
    ]
    table = pd.concat(tables, ignore_index=True)
    # The binding TFL of a compound is the lowest across its tissues.
    table["binding"] = table["tfl"] == table.groupby("compound")["tfl"].transform("min")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "tolerable_feed_levels.csv"
    table.to_csv(out, index=False, float_format="%.6g")
    print(table.to_string(index=False, float_format="%.4g"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
