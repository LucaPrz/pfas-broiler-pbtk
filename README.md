# pfas-broiler-pbtk

Physiologically based toxicokinetic (PBTK) model for the transfer of per- and
polyfluoroalkyl substances (PFAS) from feed into broiler chicken tissues, and the
derivation of tolerable feed levels from the EU maximum levels in meat and offal.

> **Status:** work in progress. The model core is complete and tested; the parameter
> estimation and the analysis scripts for the manuscript are being added.

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick start

```python
from pfas_broiler import ModelParameters, scenario_intake, simulate

# PFOS with the fitted parameters (point estimate)
params = ModelParameters.for_compound("PFOS", k_elim=0.144, P_il=2.47)

# Diet with 1 µg PFOS per kg feed from day 1 to slaughter (day 35)
sim = simulate(params, scenario_intake(1.0))
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
  paths.py       locations of data and results
  batched.py     the model for many parameter sets at once (JAX)
  inference/     censored likelihood, point fit, grid posterior
  analysis/      feed limits, transfer factors and kinetics, predictions, goodness of fit
  plots.py       figures
workflow/        numbered scripts that produce everything in results/
results/         outputs of the workflow
tests/           unit and regression tests
```

## Reproducing the results

Install with the Bayesian extra (`pip install -e ".[dev,bayes]"`), then run the steps in
order from the repository root:

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
own in about two minutes.

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

Fig. 1 (model structure) is a drawing and not produced by code. Numbering refers to the
submitted manuscript. [TODO: update after final numbering.]

`results/` was produced with the exact package versions in `requirements-lock.txt`.
With other versions of SciPy the point estimates can differ in the 5th significant digit
(the optimum is flat in P_il); the effect on all reported results is far below their
Monte Carlo error.

## Tests

```bash
pytest
```

The regression tests check that the package reproduces the results of the original
analysis code: model trajectories and likelihood values to a relative tolerance of 1e-9,
and the grid posterior (on a coarse test grid) to 1e-8.

## Citation

[TODO: article reference and Zenodo DOI]

## Licence

[TODO: licence to be decided]
