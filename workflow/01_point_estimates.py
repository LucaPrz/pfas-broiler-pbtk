"""Step 1: maximum-likelihood point estimates of k_elim and P_il for all compounds.

Output: results/point_estimates.csv
Runtime: a few minutes per compound.

    python workflow/01_point_estimates.py
"""

import pandas as pd

from pfas_broiler.config import COMPOUNDS, P_IL_READ_ACROSS
from pfas_broiler.inference import fit
from pfas_broiler.paths import POINT_ESTIMATES


def main() -> None:
    # Fit donor compounds first, so that read-across compounds can use their P_il.
    order = sorted(COMPOUNDS, key=lambda c: c in P_IL_READ_ACROSS)
    estimates: dict[str, dict] = {}
    for compound in order:
        donor = P_IL_READ_ACROSS.get(compound)
        fixed = {"P_il": estimates[donor]["P_il"]} if donor else None
        result = fit(compound, fixed)
        estimates[compound] = {
            "compound": compound,
            **result.parameters,
            "fitted": ",".join(p for p in ("k_elim", "P_il") if p not in (fixed or {})),
            "neg_log_likelihood": result.neg_log_likelihood,
        }
        print(f"{compound:6} " + "  ".join(f"{k}={v:.4f}" for k, v in result.parameters.items()))

    POINT_ESTIMATES.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([estimates[c] for c in COMPOUNDS])
    table.to_csv(POINT_ESTIMATES, index=False, float_format="%.10g")
    print(f"wrote {POINT_ESTIMATES}")


if __name__ == "__main__":
    main()
