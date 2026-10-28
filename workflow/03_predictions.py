"""Step 3: posterior predictive concentrations in the study groups.

Input:  results/posterior/samples_<compound>.csv
Output: results/predictions/study_groups_<compound>.csv
        (median and 95 % credible interval per group, compartment and day, µg/L)
Runtime: under a minute.

    python workflow/03_predictions.py
"""

from pfas_broiler.analysis.estimates import posterior_samples
from pfas_broiler.analysis.predictive import study_group_predictions
from pfas_broiler.config import COMPOUNDS
from pfas_broiler.paths import PREDICTIONS_DIR, predictions_file


def main() -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for compound in COMPOUNDS:
        table = study_group_predictions(compound, posterior_samples(compound))
        table.to_csv(predictions_file(compound), index=False, float_format="%.6g")
        print(f"wrote {predictions_file(compound)}")


if __name__ == "__main__":
    main()
