import pandas as pd
import numpy as np

# ============================================================
# UH-60A BEMT DATABASE CLEANING
# ============================================================

INPUT_FILE = "UH60A_BEMT_DATABASE_2000.csv"
OUTPUT_FILE = "UH60A_BEMT_DATABASE_CLEAN.csv"

# ============================================================
# 1. LOAD DATABASE
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("==============================================")
print(" UH-60A DATABASE CLEANING")
print("==============================================")

print(f"Original rows : {len(df)}")
print(f"Original cols : {len(df.columns)}")


# ============================================================
# 2. REMOVE COMPLETELY EMPTY ROWS
# ============================================================

df = df.dropna(how="all").copy()

print(f"\nAfter empty-row removal : {len(df)}")


# ============================================================
# 3. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
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

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# 4. REMOVE NaN VALUES
# ============================================================

before = len(df)

df = df.dropna(
    subset=numeric_columns
).copy()

print(
    f"After NaN removal      : {len(df)} "
    f"({before - len(df)} removed)"
)


# ============================================================
# 5. REMOVE INFINITE VALUES
# ============================================================

before = len(df)

df = df[
    np.isfinite(
        df[numeric_columns]
    ).all(axis=1)
].copy()

print(
    f"After Inf removal      : {len(df)} "
    f"({before - len(df)} removed)"
)


# ============================================================
# 6. REMOVE DUPLICATE DESIGN POINTS
# ============================================================

twist_columns = [
    "Twist_0.19R",
    "Twist_0.47R",
    "Twist_0.50R",
    "Twist_0.74R",
    "Twist_0.82R",
    "Twist_0.85R",
    "Twist_0.86R",
    "Twist_0.93R",
    "Twist_1.00R"
]

before = len(df)

df = df.drop_duplicates(
    subset=twist_columns
).copy()

print(
    f"After duplicate removal : {len(df)} "
    f"({before - len(df)} removed)"
)


# ============================================================
# 7. PHYSICAL VALIDITY CHECK
# ============================================================

before = len(df)

valid = (
    (df["CP"] > 0) &
    (df["FoM"] > 0) &
    (df["Thrust_N"] > 0) &
    (df["Power_W"] > 0) &
    (df["Mean_Solidity"] > 0) &
    (df["Mean_Tip_Loss"] > 0) &
    (df["Mean_Tip_Loss"] <= 1.0) &
    (df["Trim_Collective"] >= 0)
)

df = df[valid].copy()

print(
    f"After physical check   : {len(df)} "
    f"({before - len(df)} removed)"
)


# ============================================================
# 8. CHECK TARGET CT
# ============================================================

CT_TARGET = 0.005958

CT_TOLERANCE = 1e-5

ct_error = np.abs(
    df["CT"] - CT_TARGET
)

before = len(df)

df = df[
    ct_error <= CT_TOLERANCE
].copy()

print(
    f"After CT check          : {len(df)} "
    f"({before - len(df)} removed)"
)


# ============================================================
# 9. SORT BY SAMPLE NUMBER
# ============================================================

df = df.sort_values(
    by="Sample"
).reset_index(drop=True)


# ============================================================
# 10. RESET SAMPLE NUMBER
# ============================================================

df["Sample"] = np.arange(
    1,
    len(df) + 1
)


# ============================================================
# 11. FINAL DATABASE CHECK
# ============================================================

print("\n==============================================")
print(" FINAL DATABASE CHECK")
print("==============================================")

print(f"Final rows : {len(df)}")
print(f"Final cols : {len(df.columns)}")

print("\nMissing values:")
print(df.isna().sum())

print("\nFoM range:")
print(
    f"Min = {df['FoM'].min():.6f}"
)
print(
    f"Max = {df['FoM'].max():.6f}"
)

print("\nCP range:")
print(
    f"Min = {df['CP'].min():.8f}"
)
print(
    f"Max = {df['CP'].max():.8f}"
)

print("\nCT range:")
print(
    f"Min = {df['CT'].min():.8f}"
)
print(
    f"Max = {df['CT'].max():.8f}"
)


# ============================================================
# 12. SAVE CLEAN DATABASE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==============================================")
print(" CLEANING COMPLETE")
print("==============================================")
print(f"Saved as: {OUTPUT_FILE}")
print("==============================================")