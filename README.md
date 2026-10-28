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
print(sim.concentration("muscle")[35])   # µg/kg in muscle at slaughter
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
tests/           unit and regression tests
```

## Tests

```bash
pytest
```

The regression tests check that the package reproduces the results of the original
analysis code to a relative tolerance of 1e-9.

## Citation

[TODO: article reference and Zenodo DOI]

## Licence

[TODO: licence to be decided]
