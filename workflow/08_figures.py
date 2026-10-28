"""Step 8: figures, as PNG (600 dpi) and PDF (vector, fonts embedded).

Input:  results/predictions/ (steps 3 and 6), results/posterior/grid_<compound>.npz (step 2)
Output: results/figures/
Runtime: under a minute.

    python workflow/08_figures.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from pfas_broiler import plots
from pfas_broiler.inference.posterior import GridPosterior
from pfas_broiler.paths import (
    FIGURES_DIR,
    PREDICTIONS_DIR,
    TABLES_DIR,
    posterior_grid_file,
    predictions_file,
)

#: Compounds shown in the figures (all tissue partition coefficients measured).
COMPOUNDS = ("PFOS", "PFNA")


def save(fig, name: str) -> None:
    for suffix, options in ((".png", {"dpi": 600}), (".pdf", {})):
        fig.savefig(FIGURES_DIR / f"{name}{suffix}", bbox_inches="tight", **options)
    plt.close(fig)
    print(f"wrote {FIGURES_DIR / name}.png/.pdf")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save(plots.growth_curve(), "growth_curve")

    fit = pd.read_csv(TABLES_DIR / "goodness_of_fit.csv").set_index("compound")
    tables = {c: pd.read_csv(PREDICTIONS_DIR / f"observed_vs_predicted_{c}.csv") for c in COMPOUNDS}
    save(
        plots.observed_vs_predicted(tables, {c: fit.loc[c] for c in COMPOUNDS}),
        "observed_vs_predicted",
    )

    for compound in COMPOUNDS:
        predictions = pd.read_csv(predictions_file(compound))
        save(plots.concentration_time(compound, predictions), f"concentration_time_{compound}")
        grid = posterior_grid_file(compound)
        if grid.exists():
            save(plots.posterior(GridPosterior.load(grid)), f"posterior_{compound}")
        else:
            print(f"skipped posterior_{compound}: {grid} missing (run step 2)")


if __name__ == "__main__":
    main()
