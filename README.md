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
  inference/     censored likelihood, point fit, grid posterior (JAX)
workflow/        numbered scripts that produce everything in results/
results/         outputs of the workflow (posterior samples are shipped, see below)
tests/           unit and regression tests
```

## Reproducing the results

Install with the Bayesian extra (`pip install -e ".[dev,bayes]"`), then run the steps in
order from the repository root:

| Step | Script | Output | Runtime |
|---|---|---|---|
| 1 | `python workflow/01_point_estimates.py` | `results/point_estimates.csv` | ~5 min |
| 2 | `python workflow/02_posterior.py` | `results/posterior/` | ~10–30 min |

[TODO: remaining steps and the mapping of figures and tables in the article to scripts.]

The posterior samples are part of the repository, so later steps can be run without
repeating steps 1 and 2.

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
