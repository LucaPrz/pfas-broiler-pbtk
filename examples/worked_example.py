"""Worked example: PFOS in broilers fed a contaminated diet.

Uses the fitted model (point estimate and posterior samples in results/) to answer
three questions for a given PFOS concentration in the feed:

1. Which concentrations are reached in muscle, liver and kidney at slaughter
   (day 35) if the diet is fed from day 1 until slaughter?
2. How much lower are they if the bird is switched to clean feed earlier?
3. Which feed concentration keeps each tissue at its EU maximum level
   (tolerable feed level, TFL), with its 95 % credible interval?

Requires only the base installation (no JAX). Runtime about 30 s.

    python examples/worked_example.py
    python examples/worked_example.py --feed 2.0 --stop-day 25
"""

import argparse

import numpy as np

from pfas_broiler import ModelParameters, scenario_intake, simulate
from pfas_broiler.analysis.estimates import point_estimate, posterior_samples
from pfas_broiler.config import EU_MAXIMUM_LEVEL, SLAUGHTER_DAY

COMPOUND = "PFOS"
TISSUES = ("muscle", "liver", "kidney")


def main(feed: float = 1.0, stop_day: int = 25, n_samples: int | None = None) -> None:
    fitted = point_estimate(COMPOUND)
    params = ModelParameters.for_compound(COMPOUND, **fitted)
    limits = EU_MAXIMUM_LEVEL[COMPOUND]
    print(f"{COMPOUND}: k_elim = {fitted['k_elim']:.3f} 1/d, P_il = {fitted['P_il']:.2f}")
    print(f"Feed concentration: {feed:g} µg/kg (88 % dry matter)\n")

    # 1 + 2: concentrations at slaughter, continuous exposure vs early switch.
    continuous = simulate(params, scenario_intake(feed, exposure_days=SLAUGHTER_DAY))
    switched = simulate(params, scenario_intake(feed, exposure_days=stop_day))
    print(f"{'tissue':8} {'ML':>6} {'until day 35':>14} {f'until day {stop_day}':>14}")
    for tissue in TISSUES:
        c_cont = continuous.concentration(tissue)[SLAUGHTER_DAY]
        c_stop = switched.concentration(tissue)[SLAUGHTER_DAY]
        flag = "  > ML" if c_cont > limits[tissue] else ""
        print(f"{tissue:8} {limits[tissue]:6.2f} {c_cont:14.3f} {c_stop:14.3f}{flag}")
    print("(µg/kg wet weight at slaughter)\n")

    # 3: the model is linear in the intake, so TFL = ML / (concentration per µg/kg feed).
    samples = posterior_samples(COMPOUND)
    if n_samples is not None:
        samples = samples[:n_samples]
    per_unit_feed = np.array(
        [
            [
                simulate(
                    ModelParameters.for_compound(COMPOUND, k, p), scenario_intake(1.0)
                ).concentration(tissue)[SLAUGHTER_DAY]
                for tissue in TISSUES
            ]
            for k, p in samples
        ]
    )
    print(f"Tolerable feed levels ({len(samples)} posterior samples)")
    for j, tissue in enumerate(TISSUES):
        point = limits[tissue] / continuous.concentration(tissue)[SLAUGHTER_DAY] * feed
        lower, upper = np.percentile(limits[tissue] / per_unit_feed[:, j], [2.5, 97.5])
        print(f"  {tissue:8} TFL = {point:.3f} µg/kg feed  (95 % CI {lower:.3f}–{upper:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--feed", type=float, default=1.0, help="feed concentration, µg/kg")
    parser.add_argument("--stop-day", type=int, default=25, help="last day on contaminated feed")
    args = parser.parse_args()
    main(args.feed, args.stop_day)
