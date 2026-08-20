import numpy as np
import pandas as pd
import os
import tempfile
from scipy.stats import qmc
from scipy.optimize import brentq

# ============================================================
# SETTINGS
# ============================================================

N_SAMPLES = 2000

# Fractional variation about the local UH-60A baseline twist
# 0.20 = +/- 20% of the local baseline twist magnitude
VARIATION_FRACTION = 0.20

RNG_SEED = 42

# UH-60A BEMT inputs
RPM = 258.0
NB = 4
RMAX = 8.18
RMIN = 1.06
N_RADIAL = 100

# Target CT
CT_TARGET = 0.005958

# Input files
GEOM_FILE = "blade_geometry.csv"
POLAR_FILE = "airfoil_polar.csv"

# Output database
OUTPUT_FILE = "UH60A_BEMT_DATABASE_2000.csv"


# ============================================================
# IMPORT EXISTING BEMT SOLVER
# ============================================================

from BEMT import BEMTsingle


# ============================================================
# READ BASELINE UH-60A GEOMETRY
# ============================================================

geom = pd.read_csv(GEOM_FILE)

rR = geom["r/R"].to_numpy(dtype=float)
chord = geom["Chord"].to_numpy(dtype=float)
baseline_twist = geom["Twist"].to_numpy(dtype=float)
airfoil = geom["Airfoil"].astype(str).str.strip().to_numpy()


# ============================================================
# LHS DESIGN VARIABLES
# ============================================================

# Keep the first/root station fixed
design_indices = np.arange(1, len(rR))

design_rR = rR[design_indices]
design_baseline_twist = baseline_twist[design_indices]

n_variables = len(design_indices)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print()
print("==============================================")
print(" UH-60A BEMT DATABASE GENERATION")
print("==============================================")
print(f"Samples             : {N_SAMPLES}")
print(f"Radial stations     : {len(rR)}")
print(f"LHS variables       : {n_variables}")
print(f"Variation fraction  : +/- {VARIATION_FRACTION * 100:.1f}%")
print()
print("UH-60A baseline twist:")
print("----------------------------------------------")

for r, twist in zip(design_rR, design_baseline_twist):
    print(
        f"r/R = {r:.2f}   "
        f"Baseline = {twist:8.3f} deg"
    )

print()


# ============================================================
# GENERATE NONLINEAR UH-60A-BASED LHS BOUNDS
# ============================================================

# Local variation is based on the magnitude of the
# actual UH-60A baseline twist at each radial station.

variation = (
    VARIATION_FRACTION
    * np.abs(design_baseline_twist)
)

lower_bounds = (
    design_baseline_twist
    - variation
)

upper_bounds = (
    design_baseline_twist
    + variation
)


# ============================================================
# DISPLAY LHS BOUNDS
# ============================================================

print("LHS twist bounds:")
print("----------------------------------------------")

for r, base, low, high in zip(
    design_rR,
    design_baseline_twist,
    lower_bounds,
    upper_bounds
):

    print(
        f"r/R = {r:.2f}   "
        f"Baseline = {base:8.3f}°   "
        f"Range = [{low:8.3f}, {high:8.3f}]°"
    )

print()


# ============================================================
# GENERATE LATIN HYPERCUBE
# ============================================================

sampler = qmc.LatinHypercube(
    d=n_variables,
    seed=RNG_SEED
)

lhs_normalized = sampler.random(
    n=N_SAMPLES
)


# Scale normalized LHS points to the
# individual nonlinear twist bounds

lhs_twist = qmc.scale(
    lhs_normalized,
    lower_bounds,
    upper_bounds
)


# ============================================================
# BEMT SETTINGS
# ============================================================

BChar = {
    "Nb": NB,
    "Rmax": RMAX,
    "Rmin": RMIN
}


# ============================================================
# RESULT STORAGE
# ============================================================

results = []


# ============================================================
# RUN ALL 2000 DESIGNS
# ============================================================

for sample_number in range(N_SAMPLES):

    print(
        f"Running design "
        f"{sample_number + 1}/{N_SAMPLES}",
        end="\r"
    )

    # --------------------------------------------------------
    # Create modified geometry
    # --------------------------------------------------------

    modified_geom = geom.copy()

    # Root remains unchanged
    modified_geom.loc[0, "Twist"] = baseline_twist[0]

    # Apply LHS-generated nonlinear twist distribution
    modified_geom.loc[
        design_indices,
        "Twist"
    ] = lhs_twist[sample_number, :]

    # --------------------------------------------------------
    # Temporary geometry file
    # --------------------------------------------------------

    temp_file = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            newline=""
        ) as f:

            temp_file = f.name

            modified_geom.to_csv(
                temp_file,
                index=False
            )

        # ----------------------------------------------------
        # Collective trim
        # ----------------------------------------------------

        def residual(collective):

            ret, _ = BEMTsingle(
                collective,
                RPM,
                BChar,
                N_RADIAL,
                temp_file,
                POLAR_FILE
            )

            return ret[0] - CT_TARGET

        # ----------------------------------------------------
        # Find collective bracket
        # ----------------------------------------------------

        lo = 0.0
        hi = 30.0

        r_lo = residual(lo)
        r_hi = residual(hi)

        # ----------------------------------------------------
        # Check bracket
        # ----------------------------------------------------

        if r_lo * r_hi > 0:

            results.append(
                [
                    sample_number + 1,
                    *lhs_twist[sample_number, :],
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan
                ]
            )

            continue

        # ----------------------------------------------------
        # Brent trim
        # ----------------------------------------------------

        collective_trim = brentq(
            residual,
            lo,
            hi,
            xtol=1e-4
        )

        # ----------------------------------------------------
        # Final BEMT calculation
        # ----------------------------------------------------

        ret, vect = BEMTsingle(
            collective_trim,
            RPM,
            BChar,
            N_RADIAL,
            temp_file,
            POLAR_FILE
        )

        # ----------------------------------------------------
        # Extract BEMT outputs
        # ----------------------------------------------------

        CT = ret[0]
        CP = ret[1]
        FoM = ret[2]
        Thrust = ret[3]
        Power = ret[4]

        sigma_mean = ret[5]

        # ret[8] = mean tip-loss factor
        Fmean = ret[8]

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            [
                sample_number + 1,
                *lhs_twist[sample_number, :],
                collective_trim,
                CT,
                CP,
                FoM,
                Thrust,
                Power,
                sigma_mean,
                Fmean
            ]
        )

    except Exception as e:

        print(
            f"\nDesign {sample_number + 1} failed: {e}"
        )

        results.append(
            [
                sample_number + 1,
                *lhs_twist[sample_number, :],
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan
            ]
        )

    finally:

        if temp_file is not None:

            try:
                os.remove(temp_file)

            except:
                pass


# ============================================================
# DATABASE COLUMNS
# ============================================================

columns = [
    "Sample",

    "Twist_0.19R",
    "Twist_0.47R",
    "Twist_0.50R",
    "Twist_0.74R",
    "Twist_0.82R",
    "Twist_0.85R",
    "Twist_0.86R",
    "Twist_0.93R",
    "Twist_1.00R",

    "Trim_Collective",

    "CT",
    "CP",
    "FoM",
    "Thrust_N",
    "Power_W",

    "Mean_Solidity",
    "Mean_Tip_Loss"
]


# ============================================================
# CREATE FINAL DATABASE
# ============================================================

database = pd.DataFrame(
    results,
    columns=columns
)


# ============================================================
# SAVE FINAL DATABASE
# ============================================================

database.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

successful = database["FoM"].notna().sum()

failed = (
    N_SAMPLES
    - successful
)


print()
print()
print("==============================================")
print(" DATABASE GENERATION COMPLETE")
print("==============================================")
print(f"Total designs     : {N_SAMPLES}")
print(f"Successful runs   : {successful}")
print(f"Failed runs       : {failed}")
print()
print("Output file:")
print(OUTPUT_FILE)
print("==============================================")