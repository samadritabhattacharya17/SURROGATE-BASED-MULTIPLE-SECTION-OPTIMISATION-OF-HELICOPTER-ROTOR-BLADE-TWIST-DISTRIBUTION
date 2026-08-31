import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def linear_interp_extrap(x_data, y_data, x_query):

    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    x_query = np.asarray(x_query, dtype=float)

    y_query = np.interp(x_query, x_data, y_data)

    left = x_query < x_data[0]

    if np.any(left):
        slope_left = (
            (y_data[1] - y_data[0])
            / (x_data[1] - x_data[0])
        )

        y_query[left] = (
            y_data[0]
            + slope_left * (x_query[left] - x_data[0])
        )

    right = x_query > x_data[-1]

    if np.any(right):
        slope_right = (
            (y_data[-1] - y_data[-2])
            / (x_data[-1] - x_data[-2])
        )

        y_query[right] = (
            y_data[-1]
            + slope_right * (x_query[right] - x_data[-1])
        )

    return y_query


def BEMTsingle(
    alpha_collective,
    rpm,
    BChar,
    n,
    bladeGeomFile,
    airfoilPolarFile
):

    rho = 1.225

    Nb = BChar["Nb"]
    Rmax = BChar["Rmax"]
    Rmin = BChar["Rmin"]

    Tgeom = pd.read_csv(bladeGeomFile)

    if "r/R" in Tgeom.columns:
        rR_csv = Tgeom["r/R"].to_numpy(dtype=float)

    elif "r_R" in Tgeom.columns:
        rR_csv = Tgeom["r_R"].to_numpy(dtype=float)

    else:
        rR_csv = Tgeom.iloc[:, 0].to_numpy(dtype=float)

    chord_csv = Tgeom["Chord"].to_numpy(dtype=float)
    twist_csv = Tgeom["Twist"].to_numpy(dtype=float)
    airfoil_csv = Tgeom["Airfoil"].astype(str).str.strip().to_numpy()

    Tpolar = pd.read_csv(airfoilPolarFile)

    Tpolar["airfoil"] = Tpolar["airfoil"].astype(str).str.strip()

    polar_data = {}

    for airfoil_name in Tpolar["airfoil"].unique():

        data = Tpolar[
            Tpolar["airfoil"] == airfoil_name
        ].sort_values("alpha")

        polar_data[airfoil_name] = {
            "alpha": data["alpha"].to_numpy(dtype=float),
            "Cl": data["Cl"].to_numpy(dtype=float),
            "Cd": data["Cd"].to_numpy(dtype=float)
        }

    missing_airfoils = sorted(
        set(airfoil_csv) - set(polar_data.keys())
    )

    if missing_airfoils:
        raise ValueError(
            f"Airfoil(s) in blade geometry are missing from polar file: "
            f"{missing_airfoils}. Available polars: {list(polar_data.keys())}"
        )

    r_start = np.min(rR_csv)

    dr = (1.0 - r_start) / n

    r = (
        r_start
        + dr / 2
        + np.arange(n) * dr
    )

    chord_r = linear_interp_extrap(
        rR_csv,
        chord_csv,
        r
    )

    twist_r = linear_interp_extrap(
        rR_csv,
        twist_csv,
        r
    )

    airfoil_r = np.empty(n, dtype=object)

    for i in range(n):

        idx = np.searchsorted(
            rR_csv,
            r[i],
            side="right"
        ) - 1

        idx = np.clip(
            idx,
            0,
            len(rR_csv) - 1
        )

        airfoil_r[i] = airfoil_csv[idx]

    sigma_r = (
        Nb * chord_r
        / (2 * np.pi * r * Rmax)
    )

    # ------------------------------------------------------------
    # Inflow solution
    # ------------------------------------------------------------

    lam = 0.06 * np.ones(n)

    Ftip = np.ones(n)

    tol = 1e-6
    maxit = 200
    relax = 0.25

    for iteration in range(1, maxit + 1):

        f = (
            Nb / 2.0
            * (1.0 - r)
            / np.maximum(lam, 1e-6)
        )

        Ftip = (
            2.0 / np.pi
            * np.arccos(np.exp(-f))
        )

        Ftip = np.clip(
            Ftip,
            1e-4,
            1.0
        )

        phi = np.arctan2(
            lam,
            r
        )

        AoA_deg = (
            twist_r
            + alpha_collective
            - phi * 180.0 / np.pi
        )

        Cl_local = np.zeros(n)
        Cd_local = np.zeros(n)

        for i in range(n):

            af = airfoil_r[i]

            alpha_data = polar_data[af]["alpha"]
            Cl_data = polar_data[af]["Cl"]
            Cd_data = polar_data[af]["Cd"]

            Cl_local[i] = np.interp(
                AoA_deg[i],
                alpha_data,
                Cl_data
            )

            Cd_local[i] = np.interp(
                AoA_deg[i],
                alpha_data,
                Cd_data
            )

        rhs = (
            sigma_r
            * Cl_local
            * r
            / (8.0 * Ftip)
        )

        rhs = np.maximum(
            rhs,
            1e-8
        )

        lam_calc = np.sqrt(rhs)

        lam_new = (
            (1.0 - relax) * lam
            + relax * lam_calc
        )

        if np.max(
            np.abs(lam_new - lam)
        ) < tol:

            lam = lam_new
            break

        lam = lam_new

    # ------------------------------------------------------------
    # Final converged solution
    # ------------------------------------------------------------

    phi = np.arctan2(
        lam,
        r
    )

    AoA_deg = (
        twist_r
        + alpha_collective
        - phi * 180.0 / np.pi
    )

    Cl_local = np.zeros(n)
    Cd_local = np.zeros(n)

    for i in range(n):

        af = airfoil_r[i]

        alpha_data = polar_data[af]["alpha"]
        Cl_data = polar_data[af]["Cl"]
        Cd_data = polar_data[af]["Cd"]

        Cl_local[i] = np.interp(
            AoA_deg[i],
            alpha_data,
            Cl_data
        )

        Cd_local[i] = np.interp(
            AoA_deg[i],
            alpha_data,
            Cd_data
        )

    # ------------------------------------------------------------
    # Thrust coefficient
    # ------------------------------------------------------------

    dCt_dr = (
        0.5
        * sigma_r
        * Cl_local
        * r**2
    )

    Ct = np.sum(
        dr * dCt_dr
    )

    # ------------------------------------------------------------
    # Power coefficient
    # ------------------------------------------------------------

    dCpi = (
        dr
        * dCt_dr
        * lam
    )

    dCpo = (
        0.5
        * sigma_r
        * Cd_local
        * r**3
        * dr
    )

    Cp = np.sum(
        dCpo + dCpi
    )

    # ------------------------------------------------------------
    # Figure of Merit
    # ------------------------------------------------------------

    if Cp > 0.0 and Ct > 0.0:

        FoM = (
            (Ct**1.5)
            / np.sqrt(2.0)
        ) / Cp

    else:

        FoM = np.nan

    # ------------------------------------------------------------
    # Thrust and power
    # ------------------------------------------------------------

    Ad = np.pi * Rmax**2

    rev = (
        rpm
        * 2.0
        * np.pi
        / 60.0
    )

    Thrust = (
        Ct
        * rho
        * Ad
        * (rev * Rmax)**2
    )

    Power = (
        Cp
        * rho
        * Ad
        * (rev * Rmax)**3
    )

    # ------------------------------------------------------------
    # Solidity and tip-loss outputs
    # ------------------------------------------------------------

    sigma_mean = np.mean(
        sigma_r
    )

    sigma_tip = sigma_r[-1]

    Fmin = np.min(
        Ftip
    )

    Fmean = np.mean(
        Ftip
    )

    Fouter = Ftip[-1]

    # ------------------------------------------------------------
    # Return values
    # ------------------------------------------------------------

    ret = np.array([
        Ct,
        Cp,
        FoM,
        Thrust,
        Power,
        sigma_mean,
        sigma_tip,
        Fmin,
        Fmean,
        Fouter
    ])

    vect = (
        r,
        lam,
        Ftip,
        AoA_deg,
        chord_r,
        twist_r
    )

    return ret, vect


# ============================================================
# COLLECTIVE TRIM
# ============================================================

def trim_to_thrust(
    CT_target,
    rpm,
    BChar,
    n,
    bladeGeomFile,
    airfoilPolarFile,
    collective_bracket=(0.0, 30.0)
):

    def residual(alpha_collective):

        ret, _ = BEMTsingle(
            alpha_collective,
            rpm,
            BChar,
            n,
            bladeGeomFile,
            airfoilPolarFile
        )

        return ret[0] - CT_target

    lo, hi = collective_bracket

    r_lo = residual(lo)

    r_hi = residual(hi)

    if r_lo * r_hi > 0:

        raise ValueError(
            f"\nCollective bracket [{lo}, {hi}] "
            f"does not contain a root.\n"
            f"Ct at {lo:.2f} deg = "
            f"{r_lo + CT_target:.6f}\n"
            f"Ct at {hi:.2f} deg = "
            f"{r_hi + CT_target:.6f}\n"
            f"Target Ct = {CT_target:.6f}\n"
            f"Widen collective_bracket and retry."
        )

    alpha_trim = brentq(
        residual,
        lo,
        hi,
        xtol=1e-4
    )

    ret, vect = BEMTsingle(
        alpha_trim,
        rpm,
        BChar,
        n,
        bladeGeomFile,
        airfoilPolarFile
    )

    return alpha_trim, ret, vect


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("       BEMT HOVER COLLECTIVE TRIM")
    print("========================================")
    print()

    rho = 1.225

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    mass = float(
        input("Enter rotorcraft mass (kg): ")
    )

    RPM = float(
        input("Enter RPM: ")
    )

    Nb = int(
        input("Enter number of blades, Nb: ")
    )

    Rmax = float(
        input("Enter rotor radius, Rmax (m): ")
    )

    Rmin = float(
        input("Enter root cutout radius, Rmin (m): ")
    )

    BChar = {
        "Nb": Nb,
        "Rmax": Rmax,
        "Rmin": Rmin
    }

    n = int(
        input("Enter number of radial stations, n: ")
    )

    bladeGeomFile = input(
        "Enter blade geometry CSV filename: "
    )

    airfoilPolarFile = input(
        "Enter airfoil polar CSV filename: "
    )

    # --------------------------------------------------------
    # Required hover thrust
    # --------------------------------------------------------

    g = 9.81

    weight = mass * g

    Omega = (
        RPM
        * 2.0
        * np.pi
        / 60.0
    )

    # Full rotor disk area
    # Root cutout is NOT subtracted from disk area

    Ad = np.pi * Rmax**2

    CT_target = (
        weight
        / (
            rho
            * Ad
            * (Omega * Rmax)**2
        )
    )

    print()
    print("----------------------------------------")
    print("Required Hover Condition")
    print("----------------------------------------")

    print(
        f"Weight              = "
        f"{weight:.2f} N"
    )

    print(
        f"Rotor Disk Area     = "
        f"{Ad:.4f} m^2"
    )

    print(
        f"Target CT           = "
        f"{CT_target:.6f}"
    )

    # --------------------------------------------------------
    # Automatic collective trim
    # --------------------------------------------------------

    alpha_trim, out1, out2 = trim_to_thrust(
        CT_target,
        RPM,
        BChar,
        n,
        bladeGeomFile,
        airfoilPolarFile
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("========================================")
    print("       BEMT HOVER TRIM RESULTS")
    print("========================================")

    print(
        f"Trimmed Collective  = "
        f"{alpha_trim:.3f} deg"
    )

    print(
        f"CT                  = "
        f"{out1[0]:.6f}"
    )

    print(
        f"CP                  = "
        f"{out1[1]:.6f}"
    )

    print(
        f"FoM                 = "
        f"{out1[2]:.4f}"
    )

    print(
        f"Thrust              = "
        f"{out1[3]:.2f} N"
    )

    print(
        f"Power               = "
        f"{out1[4]:.2f} W"
    )

    print()
    print("----------------------------------------")
    print("Solidity and Tip Loss")
    print("----------------------------------------")

    print(
        f"Mean Solidity       = "
        f"{out1[5]:.5f}"
    )

    print(
        f"Tip Solidity        = "
        f"{out1[6]:.5f}"
    )

    print(
        f"Minimum Tip Loss F  = "
        f"{out1[7]:.5f}"
    )

    print(
        f"Mean Tip Loss F     = "
        f"{out1[8]:.5f}"
    )

    print(
        f"Tip Loss F at Tip   = "
        f"{out1[9]:.5f}"
    )

    # --------------------------------------------------------
    # Extract radial distributions
    # --------------------------------------------------------

    r = out2[0]

    lam = out2[1]

    Ftip = out2[2]

    AoA_deg = out2[3]

    chord_r = out2[4]

    twist_r = out2[5]

    sigma_r = (
        Nb
        * chord_r
        / (np.pi * Rmax)
    )

    # ========================================================
    # FIGURE 1: PRANDTL TIP LOSS FACTOR
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        r,
        Ftip,
        linewidth=2,
        label="Prandtl Tip Loss Factor, F"
    )

    plt.xlabel(
        "Radial Position, r/R"
    )

    plt.ylabel(
        "Prandtl Tip Loss Factor, F"
    )

    plt.title(
        "Prandtl Tip Loss Factor Distribution"
    )

    plt.legend()

    plt.grid(True)

    plt.xlim(
        np.min(r),
        1.0
    )

    plt.tight_layout()

    # ========================================================
    # FIGURE 2: INDUCED INFLOW RATIO
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        r,
        lam,
        linewidth=2,
        label="Induced Inflow Ratio, λ"
    )

    plt.xlabel(
        "Radial Position, r/R"
    )

    plt.ylabel(
        "Inflow Ratio, λ"
    )

    plt.title(
        "Induced Inflow Ratio Distribution"
    )

    plt.legend()

    plt.grid(True)

    plt.xlim(
        np.min(r),
        1.0
    )

    plt.tight_layout()

    # ========================================================
    # FIGURE 3: ANGLE OF ATTACK
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        r,
        AoA_deg,
        linewidth=2,
        label="Angle of Attack, α"
    )

    plt.xlabel(
        "Radial Position, r/R"
    )

    plt.ylabel(
        "Angle of Attack, α (deg)"
    )

    plt.title(
        "Blade Angle of Attack Distribution"
    )

    plt.legend()

    plt.grid(True)

    plt.xlim(
        np.min(r),
        1.0
    )

    plt.tight_layout()

    # ========================================================
    # FIGURE 4: CHORD DISTRIBUTION
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        r,
        chord_r,
        linewidth=2,
        label="Blade Chord"
    )

    plt.xlabel(
        "Radial Position, r/R"
    )

    plt.ylabel(
        "Chord (m)"
    )

    plt.title(
        "Blade Chord Distribution"
    )

    plt.legend()

    plt.grid(True)

    plt.xlim(
        np.min(r),
        1.0
    )

    plt.tight_layout()

    # ========================================================
    # FIGURE 5: TWIST DISTRIBUTION
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        r,
        twist_r,
        linewidth=2,
        label="Blade Twist"
    )

    plt.xlabel(
        "Radial Position, r/R"
    )

    plt.ylabel(
        "Twist (deg)"
    )

    plt.title(
        "Blade Twist Distribution"
    )

    plt.legend()

    plt.grid(True)

    plt.xlim(
        np.min(r),
        1.0
    )

    plt.tight_layout()

    # ========================================================
    # FIGURE 6: LOCAL SOLIDITY
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        r,
        sigma_r,
        linewidth=2,
        label="Local Solidity, σ(r)"
    )

    plt.xlabel(
        "Radial Position, r/R"
    )

    plt.ylabel(
        "Local Solidity, σ(r)"
    )

    plt.title(
        "Blade Local Solidity Distribution"
    )

    plt.legend()

    plt.grid(True)

    plt.xlim(
        np.min(r),
        1.0
    )

    plt.tight_layout()

    # ========================================================
    # FIGURE 7: TIP LOSS + INFLOW
    # ========================================================

    fig, ax1 = plt.subplots(
        figsize=(8, 5)
    )

    line1 = ax1.plot(
        r,
        Ftip,
        linewidth=2,
        label="Prandtl Tip Loss Factor, F"
    )

    ax1.set_xlabel(
        "Radial Position, r/R"
    )

    ax1.set_ylabel(
        "Prandtl Tip Loss Factor, F"
    )

    ax2 = ax1.twinx()

    line2 = ax2.plot(
        r,
        lam,
        linewidth=2,
        linestyle="--",
        label="Induced Inflow Ratio, λ"
    )

    ax2.set_ylabel(
        "Inflow Ratio, λ"
    )

    ax1.set_title(
        "Tip Loss Factor and Inflow Distribution"
    )

    lines = line1 + line2

    labels = [
        line.get_label()
        for line in lines
    ]

    ax1.legend(
        lines,
        labels,
        loc="best"
    )

    ax1.grid(True)

    ax1.set_xlim(
        np.min(r),
        1.0
    )

    fig.tight_layout()

    # ========================================================
    # FIGURE 8: CHORD + TWIST
    # ========================================================

    fig, ax1 = plt.subplots(
        figsize=(8, 5)
    )

    line1 = ax1.plot(
        r,
        chord_r,
        linewidth=2,
        label="Blade Chord"
    )

    ax1.set_xlabel(
        "Radial Position, r/R"
    )

    ax1.set_ylabel(
        "Chord (m)"
    )

    ax2 = ax1.twinx()

    line2 = ax2.plot(
        r,
        twist_r,
        linewidth=2,
        linestyle="--",
        label="Blade Twist"
    )

    ax2.set_ylabel(
        "Twist (deg)"
    )

    ax1.set_title(
        "Blade Geometry Distribution"
    )

    lines = line1 + line2

    labels = [
        line.get_label()
        for line in lines
    ]

    ax1.legend(
        lines,
        labels,
        loc="best"
    )

    ax1.grid(True)

    ax1.set_xlim(
        np.min(r),
        1.0
    )

    fig.tight_layout()

    plt.show()