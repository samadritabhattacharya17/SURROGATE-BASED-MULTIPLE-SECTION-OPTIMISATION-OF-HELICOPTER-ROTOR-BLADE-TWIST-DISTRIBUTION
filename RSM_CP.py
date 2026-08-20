import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# ============================================================
# RSM-BASED SURROGATE MODEL FOR POWER COEFFICIENT (CP)
# ============================================================

INPUT_FILE = "UH60A_BEMT_DATABASE_CLEAN.csv"

# ============================================================
# 1. LOAD CLEAN DATABASE
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("==============================================")
print(" RSM-BASED SURROGATE MODEL : CP")
print("==============================================")

print(f"Database size : {len(df)} rows")
print()


# ============================================================
# 2. DEFINE INPUT VARIABLES
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

X = df[twist_columns].values

# Response
y = df["CP"].values


# ============================================================
# 3. STANDARDIZE INPUTS
# ============================================================

# Standardization makes the polynomial coefficients
# numerically better behaved.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# 4. SECOND-ORDER RESPONSE SURFACE
# ============================================================

# Degree = 2 gives:
#
#   Linear terms
#   Quadratic terms
#   Two-factor interaction terms
#
# For 9 variables:
#
# 1 + 9 + 9 + 36 = 55 terms

poly = PolynomialFeatures(
    degree=2,
    include_bias=True
)

X_rsm = poly.fit_transform(X_scaled)


print("Number of input variables :", X.shape[1])
print("Number of RSM terms       :", X_rsm.shape[1])
print()


# ============================================================
# 5. FIT RSM SURROGATE
# ============================================================

model = LinearRegression(
    fit_intercept=False
)

model.fit(
    X_rsm,
    y
)

y_pred = model.predict(X_rsm)


# ============================================================
# 6. TRAINING PERFORMANCE
# ============================================================

R2 = r2_score(
    y,
    y_pred
)

RMSE = np.sqrt(
    mean_squared_error(
        y,
        y_pred
    )
)

MAE = mean_absolute_error(
    y,
    y_pred
)

n = len(y)
p = X_rsm.shape[1] - 1

Adjusted_R2 = (
    1
    - (1 - R2)
    * (n - 1)
    / (n - p - 1)
)


print("==============================================")
print(" TRAINING PERFORMANCE")
print("==============================================")

print(f"R²          : {R2:.8f}")
print(f"Adjusted R² : {Adjusted_R2:.8f}")
print(f"RMSE        : {RMSE:.10e}")
print(f"MAE         : {MAE:.10e}")


# ============================================================
# 7. 5-FOLD CROSS-VALIDATION
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

y_cv = cross_val_predict(
    model,
    X_rsm,
    y,
    cv=kf
)

CV_R2 = r2_score(
    y,
    y_cv
)

CV_RMSE = np.sqrt(
    mean_squared_error(
        y,
        y_cv
    )
)

CV_MAE = mean_absolute_error(
    y,
    y_cv
)


print()
print("==============================================")
print(" 5-FOLD CROSS-VALIDATION")
print("==============================================")

print(f"CV R²       : {CV_R2:.8f}")
print(f"CV RMSE     : {CV_RMSE:.10e}")
print(f"CV MAE      : {CV_MAE:.10e}")


# ============================================================
# 8. SAVE PREDICTIONS
# ============================================================

prediction_database = df.copy()

prediction_database["CP_Predicted"] = y_pred
prediction_database["CP_CV_Predicted"] = y_cv

prediction_database["CP_Error"] = (
    y - y_pred
)

prediction_database["CP_CV_Error"] = (
    y - y_cv
)

prediction_database.to_csv(
    "UH60A_CP_RSM_predictions.csv",
    index=False
)


# ============================================================
# 9. SAVE RSM COEFFICIENTS
# ============================================================

feature_names = poly.get_feature_names_out(
    twist_columns
)

coefficients = pd.DataFrame({
    "Term": feature_names,
    "Coefficient": model.coef_
})

coefficients.to_csv(
    "UH60A_CP_RSM_coefficients.csv",
    index=False
)


# ============================================================
# 10. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y,
    y_pred,
    s=18,
    color="royalblue",
    label="RSM prediction"
)

minimum = min(y.min(), y_pred.min())
maximum = max(y.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Perfect prediction"
)

plt.xlabel("Actual $C_P$")
plt.ylabel("Predicted $C_P$")
plt.title("RSM Surrogate: Actual vs Predicted $C_P$")
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "UH60A_CP_RSM_actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ============================================================
# 11. CROSS-VALIDATED ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y,
    y_cv,
    s=18,
    color="darkorange",
    label="5-Fold CV prediction"
)

minimum = min(y.min(), y_cv.min())
maximum = max(y.max(), y_cv.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Perfect prediction"
)

plt.xlabel("Actual $C_P$")
plt.ylabel("5-Fold CV Predicted $C_P$")
plt.title("5-Fold Cross-Validation: $C_P$")
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "UH60A_CP_RSM_CV.png",
    dpi=300
)

plt.show()


# ============================================================
# 12. RESIDUAL PLOT
# ============================================================

residuals = y - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals,
    s=12
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted CP")
plt.ylabel("Residual")
plt.title("RSM Residuals: CP")

plt.tight_layout()

plt.savefig(
    "UH60A_CP_RSM_residuals.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. SAVE MODEL INFORMATION
# ============================================================

with open(
    "UH60A_CP_RSM_model_info.txt",
    "w"
) as f:

    f.write(
        "UH-60A CP RSM-BASED SURROGATE MODEL\n"
    )

    f.write(
        "====================================\n\n"
    )

    f.write(
        f"Number of samples: {n}\n"
    )

    f.write(
        f"Number of input variables: {X.shape[1]}\n"
    )

    f.write(
        f"Polynomial degree: 2\n"
    )

    f.write(
        f"Number of RSM terms: {X_rsm.shape[1]}\n\n"
    )

    f.write(
        f"Training R2: {R2:.10f}\n"
    )

    f.write(
        f"Training Adjusted R2: "
        f"{Adjusted_R2:.10f}\n"
    )

    f.write(
        f"Training RMSE: "
        f"{RMSE:.12e}\n"
    )

    f.write(
        f"Training MAE: "
        f"{MAE:.12e}\n\n"
    )

    f.write(
        f"5-Fold CV R2: {CV_R2:.10f}\n"
    )

    f.write(
        f"5-Fold CV RMSE: "
        f"{CV_RMSE:.12e}\n"
    )

    f.write(
        f"5-Fold CV MAE: "
        f"{CV_MAE:.12e}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("==============================================")
print(" CP RSM SURROGATE COMPLETE")
print("==============================================")

print("Generated files:")
print("  UH60A_CP_RSM_predictions.csv")
print("  UH60A_CP_RSM_coefficients.csv")
print("  UH60A_CP_RSM_model_info.txt")
print("  UH60A_CP_RSM_actual_vs_predicted.png")
print("  UH60A_CP_RSM_CV.png")
print("  UH60A_CP_RSM_residuals.png")

print("==============================================")