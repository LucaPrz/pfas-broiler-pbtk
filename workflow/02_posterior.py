"""Step 2: exact grid posterior of the fitted parameters and posterior samples.

Input:  results/point_estimates.csv (step 1; the grid is centred on it)
Output: results/posterior/samples_<compound>.csv  (4000 samples)
        results/posterior/grid_<compound>.npz      (posterior grid, for the figures)
        results/posterior/summary.csv              (median, 95 % credible interval)
Runtime: several minutes per compound (240 x 240 grid); requires JAX.

    python workflow/02_posterior.py                 # all compounds
    python workflow/02_posterior.py --compound PFOS
"""

import argparse

import pandas as pd

from pfas_broiler.config import COMPOUNDS
from pfas_broiler.inference.posterior import grid_posterior
from pfas_broiler.paths import (
    POINT_ESTIMATES,
    POSTERIOR_DIR,
    posterior_grid_file,
    posterior_samples_file,
)

N_SAMPLES = 4000
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compound", choices=COMPOUNDS, help="only this compound")
    args = parser.parse_args()

    estimates = pd.read_csv(POINT_ESTIMATES).set_index("compound")
    POSTERIOR_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = POSTERIOR_DIR / "summary.csv"
    summary = pd.read_csv(summary_file) if summary_file.exists() else pd.DataFrame()

    for compound in [args.compound] if args.compound else COMPOUNDS:
        row = estimates.loc[compound]
        fitted = row["fitted"].split(",")
        center = {p: float(row[p]) for p in ("k_elim", "P_il")}
        fixed = {p: v for p, v in center.items() if p not in fitted}

        post = grid_posterior(compound, center, fixed)
        post.save(posterior_grid_file(compound))
        pd.DataFrame(post.sample(N_SAMPLES, seed=SEED)).to_csv(
            posterior_samples_file(compound), index=False, float_format="%.10g"
        )
        rows = []
        for name in post.names:
            lo, median, hi = post.quantiles(name, [0.025, 0.5, 0.975])
            rows.append(
                dict(
                    compound=compound,
                    parameter=name,
                    median=median,
                    ci_lower=lo,
                    ci_upper=hi,
                    mean=post.mean(name),
                )
            )
            print(f"{compound:6} {name:7} median={median:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        if not summary.empty:
            summary = summary[summary["compound"] != compound]
        summary = pd.concat([summary, pd.DataFrame(rows)], ignore_index=True)

    summary.to_csv(summary_file, index=False, float_format="%.6g")
    print(f"wrote {POSTERIOR_DIR}")


if __name__ == "__main__":
    main()
