"""Worked example: PFAS in broilers fed a contaminated diet.

Uses the fitted model (point estimate and posterior samples in results/) to answer
three questions for a given PFAS concentration in the feed:

1. Which concentrations are reached in muscle, liver and kidney at slaughter
   (day 35) if the diet is fed from day 1 until slaughter?
2. How much lower are they if the bird is switched to clean feed earlier?
3. Which feed concentration keeps each tissue at its EU maximum level
   (tolerable feed level, TFL), with its 95 % credible interval?

Requires only the base installation (no JAX). Runtime about 10 s.

    python examples/worked_example.py                       # PFOS, 1 µg/kg feed
    python examples/worked_example.py --compound PFNA --feed 0.5 --stop-day 25
"""

import argparse

import numpy as np

from pfas_broiler import ModelParameters, scenario_intake, simulate
from pfas_broiler.analysis.estimates import point_estimate, posterior_samples
from pfas_broiler.analysis.feed_limits import tfl_tissues
from pfas_broiler.config import COMPOUNDS, EU_MAXIMUM_LEVEL, SLAUGHTER_DAY
from pfas_broiler.partitions import LITERATURE_VALUES, READ_ACROSS

TISSUES = ("muscle", "liver", "kidney")


def main(
    compound: str = "PFOS",
    feed: float = 1.0,
    stop_day: int = 25,
    n_samples: int | None = None,
) -> None:
    fitted = point_estimate(compound)
    params = ModelParameters.for_compound(compound, **fitted)
    limits = EU_MAXIMUM_LEVEL.get(compound, {})
    print(f"{compound}: k_elim = {fitted['k_elim']:.3f} 1/d, P_il = {fitted['P_il']:.2f}")
    print(f"Feed concentration: {feed:g} µg/kg (88 % dry matter)\n")

    # 1 + 2: concentrations at slaughter, continuous exposure vs early switch.
    continuous = simulate(params, scenario_intake(feed, exposure_days=SLAUGHTER_DAY))
    switched = simulate(params, scenario_intake(feed, exposure_days=stop_day))
    print(f"{'tissue':8} {'ML':>6} {'until day 35':>14} {f'until day {stop_day}':>14}")
    for tissue in TISSUES:
        c_cont = continuous.concentration(tissue)[SLAUGHTER_DAY]
        c_stop = switched.concentration(tissue)[SLAUGHTER_DAY]
        ml = f"{limits[tissue]:6.2f}" if tissue in limits else f"{'-':>6}"
        flag = "  > ML" if tissue in limits and c_cont > limits[tissue] else ""
        print(f"{tissue:8} {ml} {c_cont:14.3f} {c_stop:14.3f}{flag}")
    print("(µg/kg wet weight at slaughter; ML = EU maximum level)\n")
    literature = [t for t in TISSUES if (compound, t) in LITERATURE_VALUES]
    if literature:
        print(
            f"Note: partition coefficients of {', '.join(literature)} from the literature "
            f"(all {compound} tissue values below LOQ in this study)."
        )
    for tissue in TISSUES:
        if (compound, tissue) in READ_ACROSS:
            print(
                f"Note: {tissue} partition coefficient read across from "
                f"{READ_ACROSS[(compound, tissue)]} ({tissue} below LOQ in this study)."
            )

    tissues = tfl_tissues(compound)
    if not limits:
        print(f"\nNo EU maximum level for {compound}; no tolerable feed level.")
        return
    if not tissues:
        print(
            f"\nNo tolerable feed level for {compound}: TFLs are derived only for tissues "
            "whose partition coefficient was measured in this study."
        )
        return

    # 3: the model is linear in the intake, so TFL = ML / (concentration per µg/kg feed).
    samples = posterior_samples(compound)
    if n_samples is not None:
        samples = samples[:n_samples]
    per_unit_feed = np.empty((len(samples), len(TISSUES)))
    for i, (k_elim, p_il) in enumerate(samples):
        sim = simulate(ModelParameters.for_compound(compound, k_elim, p_il), scenario_intake(1.0))
        per_unit_feed[i] = [sim.concentration(t)[SLAUGHTER_DAY] for t in TISSUES]

    print(f"\nTolerable feed levels ({len(samples)} posterior samples)")
    for j, tissue in enumerate(TISSUES):
        if tissue not in tissues:
            continue
        point = limits[tissue] / continuous.concentration(tissue)[SLAUGHTER_DAY] * feed
        lower, upper = np.percentile(limits[tissue] / per_unit_feed[:, j], [2.5, 97.5])
        note = "  (indicative: read-across)" if tissues[tissue] == "read-across" else ""
        print(f"  {tissue:8} TFL = {point:.3f} µg/kg feed  (95 % CI {lower:.3f}–{upper:.3f}){note}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compound", choices=COMPOUNDS, default="PFOS")
    parser.add_argument("--feed", type=float, default=1.0, help="feed concentration, µg/kg")
    parser.add_argument("--stop-day", type=int, default=25, help="last day on contaminated feed")
    args = parser.parse_args()
    main(args.compound, args.feed, args.stop_day)
