import pandas as pd
import numpy as np
from time import time

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error

# Models
from sklearn.linear_model import SGDRegressor, HuberRegressor, PoissonRegressor
from sklearn.linear_model import TweedieRegressor

# --- Load data ---
df = pd.read_csv("Final_Cleaned_Dataset_OPTIC_7.csv")

# --- Target & baseline ---
TARGET = "ACTUAL_SURGERY_DURATION"
BASELINE = "PLANNED_SURGERY_DURATION"

# Make sure target/baseline are numeric
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df[BASELINE] = pd.to_numeric(df[BASELINE], errors="coerce")
df = df.dropna(subset=[TARGET, BASELINE])

# --- Feature lists ---
categorical_features = [
    "LOCATION",
    "ROOM",
    "CASE_STATUS",
    "OPERATION_TYPE",
    "EMERGENCY_PRIORITY",
    "SURGICAL_CODE",
    "DISCIPLINE",
    "SURGEON",
    "ANAESTHETIST_TEAM",
    "ANESTHESIA",
    "ADMISSION_TYPE",
    "ADMISSION_CLASS_TYPE",
    "ADMISSION_STATUS",
]
numeric_features = ["ENTER_START_DELAY", "KNIFE_START_DELAY", "EXIT_OR_DELAY"]
boolean_features = ["AOH", "CANCER_INDICATOR", "TRAUMA_INDICATOR"]

# Keep only needed columns
cols_needed = (
    categorical_features + numeric_features + boolean_features + [TARGET, BASELINE]
)
df = df[cols_needed].copy()

# Ensure booleans are actually bool (or 0/1)
for b in boolean_features:
    if df[b].dtype != bool:
        df[b] = df[b].astype(bool)

# --- Train/test split indexes first (keep raw df rows for sparse pipeline) ---
train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42
)
df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()

y_train = df_train[TARGET].values
y_test = df_test[TARGET].values
baseline_test = df_test[BASELINE].values

# --- Preprocessor: sparse-safe ---
# 1) OneHotEncoder with frequency capping to avoid insane width (adjust min_frequency as needed)
# 2) Scale numeric only (keeps matrix sparse overall)
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("bool", "passthrough", boolean_features),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=50,  # group very rare categories; tweak if needed
                sparse_output=True,
            ),
            categorical_features,
        ),
    ],
    sparse_threshold=1.0,  # keep output sparse
)

# --- Models to compare ---
models = {
    # Fast, scalable linear with elastic-net penalty (no ridge)
    "SGD (ElasticNet)": SGDRegressor(
        loss="huber",  # robust to outliers
        penalty="elasticnet",
        alpha=1e-4,  # overall regularization strength
        l1_ratio=0.15,  # mix between L1 and L2 (but not pure ridge)
        max_iter=2000,
        tol=1e-3,
        random_state=42,
    ),
    # Robust regression (handles heavy tails)
    "HuberRegressor": HuberRegressor(
        epsilon=1.35,
        alpha=1e-4,
        max_iter=5000,
        tol=1e-4,  # ↑ stronger convergence settings
    ),
    # Count-like / non-negative target models that accept sparse input well
    "PoissonRegressor": PoissonRegressor(alpha=1e-6, max_iter=1000, tol=1e-6),
    "TweedieRegressor (power=1.5)": TweedieRegressor(
        power=1.5, alpha=1e-6, max_iter=1000
    ),
}

results = []
baseline_mae = mean_absolute_error(y_test, baseline_test)
print(f"Baseline MAE (Actual vs Planned): {baseline_mae:.4f}")

for name, model in models.items():
    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
    t0 = time()
    pipe.fit(df_train.drop(columns=[TARGET, BASELINE]), y_train)
    preds = pipe.predict(df_test.drop(columns=[TARGET, BASELINE]))
    mae = mean_absolute_error(y_test, preds)
    elapsed = time() - t0
    results.append((name, mae, baseline_mae - mae, elapsed))
    print(
        f"{name:30s}  MAE={mae:.4f}  ΔvsBaseline={baseline_mae - mae:.4f}  time={elapsed:.1f}s"
    )

# ================================
# DETAILED DIAGNOSTICS FOR HUBER
# (Place AFTER the comparison loop)
# ================================
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.inspection import permutation_importance

# Rebuild the Huber-only pipe (same preprocess + current params)
huber_pipe = Pipeline(steps=[("prep", preprocess), ("model", models["HuberRegressor"])])
huber_pipe.fit(df_train.drop(columns=[TARGET, BASELINE]), y_train)
pred = huber_pipe.predict(df_test.drop(columns=[TARGET, BASELINE]))

# --- Metrics (version-safe RMSE) ---
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = float(np.sqrt(mse))  # <- no 'squared=False' (works on older sklearn)
r2 = r2_score(y_test, pred)
mape = float((np.abs((y_test - pred) / np.maximum(1e-8, y_test))).mean() * 100)
medae = median_absolute_error(y_test, pred)

print("\n[Huber] Detailed metrics")
print(f"MAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"R^2:   {r2:.4f}")
print(f"MAPE:  {mape:.2f}%")
print(f"MedAE: {medae:.4f}")

# --- Simple residual summary ---
resid = y_test - pred
print("\nResiduals:")
print(pd.Series(resid).describe([0.05, 0.25, 0.5, 0.75, 0.95]).to_string())

# --- Coefficients / effects ---
try:
    feat_names = huber_pipe.named_steps["prep"].get_feature_names_out()
except AttributeError:
    # Fallback if older sklearn lacks get_feature_names_out on ColumnTransformer
    feat_names = np.array(
        [f"f{i}" for i in range(len(huber_pipe.named_steps["model"].coef_))]
    )

coefs = huber_pipe.named_steps["model"].coef_
coef_s = pd.Series(coefs, index=feat_names).sort_values(key=np.abs, ascending=False)

print("\nTop positive effects:")
print(coef_s[coef_s > 0].head(15).round(4).to_string())

print("\nTop negative effects:")
print(coef_s[coef_s < 0].head(15).round(4).to_string())

print("\nIntercept:", round(huber_pipe.named_steps["model"].intercept_, 4))

# --- Slice by DISCIPLINE (if present) ---
diagnose = df_test.copy()
diagnose["pred"] = pred
diagnose["abs_err_model"] = np.abs(diagnose[TARGET] - diagnose["pred"])
diagnose["abs_err_planned"] = np.abs(diagnose[TARGET] - diagnose[BASELINE])

if "DISCIPLINE" in diagnose.columns:
    grp = (
        diagnose.groupby("DISCIPLINE")
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "MAE_model": g["abs_err_model"].mean(),
                    "MAE_planned": g["abs_err_planned"].mean(),
                    "ΔvsPlanned": g["abs_err_planned"].mean()
                    - g["abs_err_model"].mean(),
                }
            )
        )
        .sort_values("ΔvsPlanned", ascending=False)
    )
    print("\nTop disciplines by improvement (ΔvsPlanned):")
    print(grp.head(12).to_string())
