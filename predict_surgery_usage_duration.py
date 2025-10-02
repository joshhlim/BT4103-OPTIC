import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack

import warnings
warnings.filterwarnings("ignore")

# === Step 1: Read Excel with selected columns ===
cols_to_read = [
    'LOCATION', 'ROOM', 'CASE_STATUS', 'OPERATION_TYPE', 'EMERGENCY_PRIORITY',
    'SURGICAL_CODE', 'DISCIPLINE', 'SURGEON', 'ANAESTHETIST_TEAM',
    'ANESTHESIA', 'EQUIPMENT', 'ADMISSION_STATUS', 'ADMISSION_CLASS_TYPE', 'ADMISSION_TYPE',
    'ADMISSION_WARD', 'ADMISSION_BED', 'AOH', 'BLOOD', 'IMPLANT', 'DIAGNOSIS',
    'CANCER_INDICATOR', 'TRAUMA_INDICATOR', 'DIFF_SURGERY_DURATION', 'DIFF_USAGE_DURATION',
    'ACTUAL_SURGERY_DURATION', 'ACTUAL_USAGE_DURATION'
]

df = pd.read_excel("Final_Cleaned_Dataset_OPTIC_7.xlsx", usecols=cols_to_read)

# === Step 1a: Outlier Filtering ===
def remove_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr # can adjust multiplier
    upper = q3 + 1.5 * iqr # can adjust multiplier
    return series.between(lower, upper)

n_before = len(df)

mask_surg = remove_outliers_iqr(df['DIFF_SURGERY_DURATION'])
mask_usage = remove_outliers_iqr(df['DIFF_USAGE_DURATION'])
df = df[mask_surg & mask_usage]

n_after = len(df)
n_removed = n_before - n_after
pct_removed = (n_removed / n_before) * 100

print(f"Rows removed: {n_removed}")
print(f"Percentage removed: {pct_removed:.2f}%")

# === Step 2: Clean / preprocess columns ===
df['EMERGENCY_PRIORITY'].replace("0", np.nan, inplace=True)
df['EQUIPMENT'].replace("0", np.nan, inplace=True)
df['IMPLANT'].replace("0", np.nan, inplace=True)
df['DIAGNOSIS'].replace("not recorded", np.nan, inplace=True)

df['BLOOD'] = df['BLOOD'].apply(lambda x: False if str(x).strip().upper() == 'NIL' else True)

def categorize_text(text):
    if pd.isna(text):
        return 'Unknown'
    text = str(text).lower()
    if 'cancer' in text or 'ca' in text:
        return 'Cancer'
    if 'stone' in text:
        return 'Gallstone/Kidney'
    if 'rupture' in text or 'tear' in text:
        return 'Trauma'
    if 'pregnancy' in text:
        return 'Pregnancy'
    if 'hydrocephalus' in text:
        return 'Neuro'
    if 'subfert' in text:
        return 'Fertility'
    return 'Other'

df['IMPLANT_CAT'] = df['IMPLANT'].apply(categorize_text)
df['DIAGNOSIS_CAT'] = df['DIAGNOSIS'].apply(categorize_text)

# === Step 3: Feature Engineering ===
categorical_cols = [
    'LOCATION', 'ROOM', 'CASE_STATUS', 'OPERATION_TYPE', 'EMERGENCY_PRIORITY',
    'DISCIPLINE', 'SURGEON', 'ANAESTHETIST_TEAM',
    'ANESTHESIA', 'EQUIPMENT', 'ADMISSION_STATUS', 'ADMISSION_CLASS_TYPE',
    'ADMISSION_TYPE', 'ADMISSION_WARD', 'ADMISSION_BED', 'IMPLANT_CAT', 'DIAGNOSIS_CAT'
]

for col in categorical_cols:
    df[col] = df[col].astype(str)

bool_cols = ['AOH', 'BLOOD', 'CANCER_INDICATOR', 'TRAUMA_INDICATOR']

# Dependent variables
y = df[['DIFF_SURGERY_DURATION', 'DIFF_USAGE_DURATION']]
surgical_code_df = df[['SURGICAL_CODE']]
X_raw = df[categorical_cols + bool_cols]

# === Step 4: Define Custom Transformers ===
class SurgicalCodeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mlb = MultiLabelBinarizer(sparse_output=True)
        self.mlb.fit(X['SURGICAL_CODE'].fillna('').apply(self._split_codes))
        return self

    def transform(self, X):
        return self.mlb.transform(X['SURGICAL_CODE'].fillna('').apply(self._split_codes))

    def _split_codes(self, val):
        return [code.strip() for code in str(val).split(';') if code.strip()]

# === Step 5: Preprocessing Pipeline ===
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols),
        ('bool', 'passthrough', bool_cols)
    ],
    sparse_threshold=1.0
)

class FeatureCombiner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.surgical = SurgicalCodeTransformer()
        self.preprocessor = preprocessor

    def fit(self, X, y=None):
        self.surgical.fit(X[1])
        self.preprocessor.fit(X[0])
        return self

    def transform(self, X):
        X_main = self.preprocessor.transform(X[0])
        X_surg = self.surgical.transform(X[1])
        return hstack([X_main, X_surg])

# === Step 6: Train/Test Split ===
X_train_raw, X_test_raw, surg_train, surg_test, y_train, y_test = train_test_split(
    X_raw, surgical_code_df, y, test_size=0.2, random_state=42
)

# Yeo-Johnson transform outcomes
pt_y = PowerTransformer(method='yeo-johnson')
y_train_trans = pt_y.fit_transform(y_train)
y_test_trans = pt_y.transform(y_test)

X_combiner = FeatureCombiner()
X_combiner.fit((X_train_raw, surg_train))

X_train = X_combiner.transform((X_train_raw, surg_train))
X_test = X_combiner.transform((X_test_raw, surg_test))

# === Step 7: Train Model ===
model = Ridge(alpha=1.0)
model.fit(X_train, y_train_trans)

# === Step 8: Predict & Evaluate ===
y_pred_trans = model.predict(X_test)
y_pred = pt_y.inverse_transform(y_pred_trans)  # inverse transform back

mse_surg = mean_squared_error(y_test['DIFF_SURGERY_DURATION'], y_pred[:, 0])
mse_usage = mean_squared_error(y_test['DIFF_USAGE_DURATION'], y_pred[:, 1])

print("MSE (DIFF_SURGERY_DURATION):", mse_surg)
print("MSE (DIFF_USAGE_DURATION):", mse_usage)

# === Step 9: Save results to txt ===
feature_names = X_combiner.preprocessor.get_feature_names_out(categorical_cols + bool_cols)
surg_features = X_combiner.surgical.mlb.classes_
all_feature_names = list(feature_names) + list(surg_features)

with open("ridge_results.txt", "w") as f:
    f.write(f"MSE (DIFF_SURGERY_DURATION): {mse_surg}\n")
    f.write(f"MSE (DIFF_USAGE_DURATION): {mse_usage}\n\n")

    for i, target in enumerate(['DIFF_SURGERY_DURATION', 'DIFF_USAGE_DURATION']):
        f.write(f"{target} = {model.intercept_[i]:.6f}")
        for coef, name in zip(model.coef_[i], all_feature_names):
            f.write(f" + ({coef:.6f} * {name})")
        f.write("\n\n")

# Naive Approach (no transformation)
# MSE (DIFF_SURGERY_DURATION): 1555.9589672336979
# MSE (DIFF_USAGE_DURATION): 7362.460400832329

# With Yeo-Johnson Transformation
# MSE (DIFF_SURGERY_DURATION): 1554.0293511102093
# MSE (DIFF_USAGE_DURATION): 7278.200171609524

# Exclude 3 * IQR Outliers, With Yeo-Johnson Transformation
# Rows removed: 13075
# Percentage removed: 5.26%
# MSE (DIFF_SURGERY_DURATION): 652.6590966672973
# MSE (DIFF_USAGE_DURATION): 793.8700146592308

# Exclude 1.5 * IQR Outliers, With Yeo-Johnson Transformation
# Rows removed: 37040
# Percentage removed: 14.89%
# MSE (DIFF_SURGERY_DURATION): 345.6797313099851
# MSE (DIFF_USAGE_DURATION): 442.9399566765906

# ============= Actual Data using Gamma GLM =============================

from sklearn.linear_model import TweedieRegressor

# === Step 1: Define new outcomes ===
y_actual = df[['ACTUAL_SURGERY_DURATION', 'ACTUAL_USAGE_DURATION']]

# Same train/test split as before
X_train_raw, X_test_raw, surg_train, surg_test, y_train, y_test = train_test_split(
    X_raw, surgical_code_df, y_actual, test_size=0.2, random_state=42
)

X_combiner = FeatureCombiner()
X_combiner.fit((X_train_raw, surg_train))

X_train = X_combiner.transform((X_train_raw, surg_train))
X_test = X_combiner.transform((X_test_raw, surg_test))

# === Step 2: Train two Gamma GLMs (separately) ===
glm_surg = TweedieRegressor(power=2, link="log", alpha=1.0)
glm_usage = TweedieRegressor(power=2, link="log", alpha=1.0)

# Add small positive constant to avoid log(0)
epsilon = 1e-3  # small positive constant
y_train_surg = y_train['ACTUAL_SURGERY_DURATION'] + epsilon
y_train_usage = y_train['ACTUAL_USAGE_DURATION'] + epsilon
y_test_surg = y_test['ACTUAL_SURGERY_DURATION'] + epsilon
y_test_usage = y_test['ACTUAL_USAGE_DURATION'] + epsilon

glm_surg.fit(X_train, y_train_surg)
glm_usage.fit(X_train, y_train_usage)

# === Step 3: Predict ===
y_pred_surg = glm_surg.predict(X_test)
y_pred_usage = glm_usage.predict(X_test)

# === Step 4: Evaluate ===
mse_surg_actual = mean_squared_error(y_test['ACTUAL_SURGERY_DURATION'], y_pred_surg)
mse_usage_actual = mean_squared_error(y_test['ACTUAL_USAGE_DURATION'], y_pred_usage)

print("MSE (ACTUAL_SURGERY_DURATION):", mse_surg_actual)
print("MSE (ACTUAL_USAGE_DURATION):", mse_usage_actual)

# === Step 5: Save results ===
with open("gamma_glm_results.txt", "w") as f:
    f.write(f"MSE (ACTUAL_SURGERY_DURATION): {mse_surg_actual}\n")
    f.write(f"MSE (ACTUAL_USAGE_DURATION): {mse_usage_actual}\n\n")
    f.write("GLM Surgery Coefficients:\n")
    f.write(str(glm_surg.coef_))
    f.write("\nIntercept:\n")
    f.write(str(glm_surg.intercept_))
    f.write("\n\nGLM Usage Coefficients:\n")
    f.write(str(glm_usage.coef_))
    f.write("\nIntercept:\n")
    f.write(str(glm_usage.intercept_))

# Exclude 1.5 * IQR Outliers, With Yeo-Johnson Transformation
# Rows removed: 37040
# Percentage removed: 14.89%
# MSE (ACTUAL_SURGERY_DURATION): 1862.2790855135293
# MSE (ACTUAL_USAGE_DURATION): 2628.7053676040346