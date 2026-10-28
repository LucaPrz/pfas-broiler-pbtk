"""Study design and fixed constants shared across the package."""

from __future__ import annotations

#: Compounds with a fitted model (linear isomers only).
COMPOUNDS: tuple[str, ...] = ("PFOS", "PFNA", "PFOA", "PFDA", "PFHxS")
ISOMER = "Linear"

#: Compounds whose data cannot identify P_il (all PFHxS tissues are below the
#: LOQ); P_il is taken from the fit of a structurally similar compound and only
#: k_elim is fitted. compound -> donor compound.
P_IL_READ_ACROSS: dict[str, str] = {"PFHxS": "PFOS"}

#: Exposure groups. All birds were placed on day 1; group Ek received the
#: contaminated diet up to and including ``EXPOSURE_STOP_DAY[Ek]`` and clean feed
#: afterwards (depuration). E5 was exposed until slaughter.
GROUPS: tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5")
EXPOSURE_STOP_DAY: dict[str, int] = {"E1": 16, "E2": 23, "E3": 28, "E4": 32, "E5": 35}

#: Day of slaughter (fasted morning slaughter; no additional depuration).
SLAUGHTER_DAY = 35

#: Tissues measured and predicted. Serum was measured; it is modelled as plasma.
TISSUES: tuple[str, ...] = ("plasma", "liver", "kidney", "muscle")

#: Measured matrix -> model compartment.
MATRIX_TO_COMPARTMENT: dict[str, str] = {
    "serum": "plasma",
    "liver": "liver",
    "kidney": "kidney",
    "muscle": "muscle",
}

#: Limit of quantification, uniform for all analytes and matrices
#: (µg/kg tissue, µg/L serum).
LOQ = 0.5

#: EU maximum levels, Commission Regulation (EU) 2023/915, µg/kg wet weight.
#: Muscle = "meat of poultry"; liver and kidney = "offal of poultry".
EU_MAXIMUM_LEVEL: dict[str, dict[str, float]] = {
    "PFOS": {"muscle": 0.30, "liver": 6.0, "kidney": 6.0},
    "PFNA": {"muscle": 0.20, "liver": 0.4, "kidney": 0.4},
    "PFOA": {"muscle": 0.80, "liver": 0.70, "kidney": 0.70},
    "PFHxS": {"muscle": 0.20},
}
