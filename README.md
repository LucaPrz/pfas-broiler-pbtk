# pfas-broiler-pbtk

Physiologically based toxicokinetic (PBTK) model for the transfer of per- and
polyfluoroalkyl substances (PFAS) from feed into broiler chicken tissues, and the
derivation of tolerable feed levels from the EU maximum levels in meat and offal.

This repository contains the model, the data of the feeding study, and the complete
workflow that produces every model result reported in the article
[TODO: reference]. It is archived on Zenodo [TODO: DOI].

## The model in brief

Six compartments: stomach (with crop), liver, plasma, kidney, muscle and rest of body. The
intestine is in instantaneous partition equilibrium with the liver (enterohepatic
reservoir, intestine:liver partition coefficient P_il). Exchange between plasma and the
organs is flow-limited and governed by tissue:plasma partition coefficients measured in
the study. PFAS are eliminated from plasma with a first-order rate k_elim. Organ volumes
and blood flows change with body weight, which follows a Gompertz growth curve fitted to
the study birds.

Two parameters per compound, k_elim and P_il, were estimated from the tissue
concentrations of five exposure groups with staggered ends of exposure. Values below the
limit of quantification are treated as left-censored. Parameter uncertainty comes from
the exact Bayesian posterior on a grid, and is propagated to all outputs with 4000
posterior samples. Details: article, Section [TODO], and the docstrings of
`src/pfas_broiler/model.py` and `src/pfas_broiler/inference/`.

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[bayes,dev]"      # bayes: JAX, needed for steps 2–8 of the workflow
```

To reproduce `results/` exactly, install the package versions that produced it:

```bash
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

## Worked example

```bash
python examples/worked_example.py               # PFOS, 1 µg/kg feed
python examples/worked_example.py --feed 2.0 --stop-day 25
```

For a given PFOS concentration in the feed, the example computes the concentrations in
muscle, liver and kidney at slaughter, both for exposure until slaughter and after an
earlier switch to clean feed. It also computes the tolerable feed level of each tissue
with its 95 % credible interval. It needs only the base installation (no JAX).

The model can also be used directly:

```python
from pfas_broiler import ModelParameters, scenario_intake, simulate

params = ModelParameters.for_compound("PFOS", k_elim=0.144, P_il=2.47)
sim = simulate(params, scenario_intake(1.0))  # 1 µg/kg feed from day 1 to day 35
print(sim.concentration("muscle")[35])  # µg/kg in muscle at slaughter
```

## Repository structure

```
data/            input data (study measurements and literature values), see data/README.md
src/pfas_broiler/
  config.py      study design and constants (exposure groups, LOQ, EU maximum levels)
  data.py        loading the input data
  growth.py      body-weight growth curve
  physiology.py  compartment volumes and blood flows
  partitions.py  tissue:plasma partition coefficients
  exposure.py    daily PFAS intake
  model.py       PBTK model and solver
  batched.py     the model for many parameter sets at once (JAX)
  inference/     censored likelihood, point fit, grid posterior
  analysis/      feed limits, transfer factors and kinetics, predictions, goodness of fit
  plots.py       figures
  paths.py       locations of data and results
workflow/        numbered scripts that produce everything in results/
examples/        worked example
results/         outputs of the workflow
tests/           unit and regression tests
```

## Reproducing the results

Run the steps in order from the repository root:

| Step | Script | Output | Runtime |
|---|---|---|---|
| 1 | `workflow/01_point_estimates.py` | `results/point_estimates.csv` | ~5 min |
| 2 | `workflow/02_posterior.py` | `results/posterior/` | ~10–30 min |
| 3 | `workflow/03_predictions.py` | `results/predictions/study_groups_*.csv` | < 1 min |
| 4 | `workflow/04_feed_limits.py` | `results/tables/tolerable_feed_levels.csv` | < 1 min |
| 5 | `workflow/05_transfer_and_kinetics.py` | `results/tables/transfer_and_kinetics.csv` | seconds |
| 6 | `workflow/06_goodness_of_fit.py` | `results/tables/goodness_of_fit.csv` | seconds |
| 7 | `workflow/07_parameter_tables.py` | `results/tables/` (parameter tables) | seconds |
| 8 | `workflow/08_figures.py` | `results/figures/` (PNG 600 dpi and PDF) | < 1 min |

The results of steps 1 and 2 are part of the repository, so steps 3–8 can be run on their
own in about a minute.

### Where each result of the article comes from

| Article | File in `results/` | Step |
|---|---|---|
| Fig. 2 growth curve | `figures/growth_curve` | 8 |
| Fig. 4 predicted vs observed | `figures/observed_vs_predicted` | 8 |
| Figs. 5, 6 concentration–time (PFOS, PFNA) | `figures/concentration_time_PFOS`, `_PFNA` | 8 |
| Table 3 TF, BTF, TR, t½, MRT | `tables/transfer_and_kinetics.csv` | 5 |
| Table 4 tolerable feed levels | `tables/tolerable_feed_levels.csv` | 4 |
| Table S15 growth model | `tables/growth_parameters.csv` | 7 |
| Table S16 physiology | `tables/physiology.csv` | 7 |
| Table S17 partition coefficients | `tables/partition_coefficients.csv` | 7 |
| Table S18 fitted parameters, goodness of fit | `tables/fitted_parameters.csv` | 7 |
| Fig. S4 joint posterior (PFOS, PFNA) | `figures/posterior_PFOS`, `_PFNA` | 8 |

Fig. 1 (model structure) is a drawing and not produced by code.
[TODO: update the numbering to the final article.]

`results/` was produced with the package versions in `requirements-lock.txt`. With other
versions of SciPy the point estimates can differ in the 5th significant digit (the optimum
is flat in P_il). The effect on all reported results is far below their Monte Carlo error.

## Tests

```bash
pytest
```

The regression tests check that the package reproduces the results of the original
analysis code used during the study. Model trajectories, likelihood values, transfer
factors and kinetics agree to a relative tolerance of 1e-9. Posterior predictions, feed
limits and goodness of fit agree to 1e-8, computed from the same posterior samples.

## Citation

Please cite the article and this software; see `CITATION.cff`. [TODO]

## Licence

[TODO: licence of the code and of the data (see data/README.md).]
