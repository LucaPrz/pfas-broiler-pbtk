# Input data

All files are CSV (comma-separated, UTF-8, `.` as decimal separator).

## Study data (`raw/`)

Feeding study with male broilers [TODO: strain, Ross 308 or Ross 500 — check against the
methods section]. Day 1 is the day of placement (8 October 2024), slaughter on day 35. Five
groups (E1–E5) received a diet containing PFAS-contaminated fishmeal until day 16, 23, 28, 32
and 35, respectively, and clean feed afterwards.

| File | Content | Units |
|---|---|---|
| `tissue_concentrations.csv` | PFAS concentration per animal in serum, liver, kidney and muscle. Each group was sampled at the end of its exposure and on day 35 (5 birds per group and day; 4 for E4 on day 35, 9 for E5 on day 35). Values below the limit of quantification (LOQ = 0.5) are recorded as `0`. [TODO: which muscle was sampled] | µg/kg wet weight; serum µg/L |
| `feed_intake.csv` | Daily feed consumption per pen (`feed_intake_g`) and number of birds in the pen (`n_birds`). | g/pen/day |
| `feed_concentration.csv` | PFAS concentration in the contaminated diet used in the model (see below). | µg/kg feed, 88 % dry matter |
| `body_weight.csv` | Individual body weights during the study. | g |
| `organ_weights.csv` | Body, liver and kidney weights of birds weighed at slaughter, including control birds (`C`). Group labels as in the slaughter protocol. | g |

**Analysis.** Tissues and serum were analysed at Fraunhofer IME, feed and fishmeal by the
NRL POP (laboratory reports in the original data). [TODO: analytical method, reference and
validation, from the experimental section of the paper.]

**Diet concentrations.** PFOS, PFNA and PFOA were quantified in the finished diet
(sample "Futter E"). PFDA and PFHxS were below the LOQ in the diet; their concentration was
calculated from the fishmeal analysis multiplied by the fishmeal inclusion rate of 10 %.

> **Open:** the values used for PFOS, PFNA and PFOA differ by up to 0.13 % from the
> laboratory value × 0.88 (e.g. PFOS 3.44 × 0.88 = 3.027 vs 3.0274 used). The stored values are
> those used for the published results.

**Derivation of the files from the laboratory reports.** The files were exported from the
laboratory spreadsheets without changes to the values. Columns were renamed to English, dates
converted to ISO format, the matrix "Plasma" renamed to "serum" (what was measured), and the
organ weights taken from the slaughter protocol sheet.

## Literature data (`literature/`)

| File | Source | Used for |
|---|---|---|
| `ross308_male_meat_yield.csv` | Aviagen (2022). *Ross 308 / Ross 308 FF Broiler Performance Objectives*, carcass yield of males, "total meat %" (deboned breast and leg meat without skin, % of live weight). | Volume of the muscle compartment |

Further literature values are defined in the code, each with its source:
physiological parameters from Wang et al. (2021) in `src/pfas_broiler/physiology.py`, and
PFHxS partition coefficients from Aarab et al. (2026) in `src/pfas_broiler/partitions.py`.

## Licence

[TODO: licence for the data, e.g. CC BY 4.0, after approval by the data owners.]
