"""Loading, cleaning, harmonisation and missing-data handling.

Missing-data policy (the requirement was: do NOT just drop rows or fill a constant):
  1. Detect *disguised* missing values - physiologically impossible zeros (Pima) and
     implausible entries (DiaBD: height 0.36 m, BMI 574, pulse 5 ...) become NaN.
  2. Keep the information that a value was missing: explicit missing-indicator columns.
  3. Model-based conditional imputation (MICE / IterativeImputer with BayesianRidge),
     fitted on training folds only, so each missing value is estimated from that
     patient's other attributes - not replaced by a global mean/median/constant.
  4. Tree boosting (XGBoost) uses native sparsity-aware splits (learns a default
     direction for missing values) - no imputation at all.
  5. At inference the app checks whether enough valid inputs exist and abstains
     ("insufficient data") instead of inventing a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from .config import DIABD_CSV, MGDL_PER_MMOL, PIMA_CSV, RANDOM_STATE

# ---------------------------------------------------------------- feature sets
PIMA_FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                 "BMI", "DiabetesPedigreeFunction", "Age"]

DIABD_FEATURES = ["age", "gender_male", "pulse_rate", "systolic_bp", "diastolic_bp",
                  "glucose", "bmi", "family_diabetes", "hypertensive",
                  "family_hypertension", "cardiovascular_disease", "stroke"]

# Harmonised attributes shared by both datasets AND collectable from the clinician GUI
GENERAL_FEATURES = ["age", "gender_male", "bmi", "glucose", "systolic_bp", "diastolic_bp",
                    "family_diabetes", "hypertensive", "family_hypertension",
                    "cardiovascular_disease", "stroke"]

# Hypertension model (DiaBD label `hypertensive`)
HTN_FEATURES = ["age", "gender_male", "bmi", "glucose", "systolic_bp", "diastolic_bp",
                "family_hypertension", "family_diabetes", "cardiovascular_disease", "stroke"]

# Plausibility ranges (values outside -> treated as missing, never silently kept)
PLAUSIBLE = {
    "age": (1, 120),
    "height_m": (1.20, 2.20),
    "weight": (25, 250),
    "bmi": (12, 70),
    "pulse_rate": (35, 200),
    "systolic_bp": (70, 250),
    "diastolic_bp": (40, 150),
    "glucose_mgdl": (27, 630),   # 1.5 - 35 mmol/L
}


def _out_of_range(s: pd.Series, key: str) -> pd.Series:
    lo, hi = PLAUSIBLE[key]
    return s.notna() & ((s < lo) | (s > hi))


# ---------------------------------------------------------------- Pima / diabetes.csv
def load_pima(dedupe: bool = True) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(PIMA_CSV)
    rep = {"dataset": "diabetes.csv (Pima-style)", "rows_raw": int(len(df)),
           "exact_duplicate_rows": int(df.duplicated().sum())}
    if dedupe:
        # 8,276 of 9,054 rows are exact copies -> keeping them leaks test rows into training
        df = df.drop_duplicates().reset_index(drop=True)
    rep["rows_after_dedup"] = int(len(df))
    df = df.copy()
    rep["disguised_missing_zeros"] = {}
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        n = int((df[c] == 0).sum())
        rep["disguised_missing_zeros"][c] = n
        df[c] = df[c].astype(float).where(df[c] != 0, np.nan)
    rep["missing_after_cleaning"] = {c: int(v) for c, v in df[PIMA_FEATURES].isna().sum().items()}
    rep["class_balance"] = {str(k): int(v) for k, v in df["Outcome"].value_counts().items()}
    return df, rep


# ---------------------------------------------------------------- DiaBD (Bangladesh)
def load_diabd() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(DIABD_CSV)
    df = raw.copy()
    rep = {"dataset": "DiaBD (Bangladesh)", "rows_raw": int(len(df)),
           "exact_duplicate_rows": int(df.duplicated().sum()), "implausible_set_missing": {}}
    df["gender_male"] = (df["gender"].str.strip().str.lower() == "male").astype(float)
    df["diabetic"] = (df["diabetic"].str.strip().str.lower() == "yes").astype(int)
    df["glucose_mmol"] = df["glucose"].astype(float)
    df["glucose"] = df["glucose_mmol"] * MGDL_PER_MMOL  # harmonise to mg/dL

    checks = {"height": "height_m", "weight": "weight", "pulse_rate": "pulse_rate",
              "systolic_bp": "systolic_bp", "diastolic_bp": "diastolic_bp",
              "glucose": "glucose_mgdl", "age": "age"}
    for col, key in checks.items():
        df[col] = df[col].astype(float)
        bad = _out_of_range(df[col], key)
        rep["implausible_set_missing"][col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    bad_bp = df["diastolic_bp"] >= df["systolic_bp"]
    rep["implausible_set_missing"]["bp_diastolic_ge_systolic"] = int(bad_bp.sum())
    df.loc[bad_bp, ["systolic_bp", "diastolic_bp"]] = np.nan

    # BMI: recompute from validated height/weight; recorded BMI only used if plausible
    computed = df["weight"] / df["height"] ** 2
    recorded = df["bmi"].astype(float).where(~_out_of_range(df["bmi"].astype(float), "bmi"))
    rep["implausible_set_missing"]["bmi_recorded"] = int(_out_of_range(df["bmi"].astype(float), "bmi").sum())
    rep["bmi_recorded_vs_computed_mismatch_gt1"] = int(((computed - df["bmi"]).abs() > 1).sum())
    df["bmi"] = computed.where(computed.notna(), recorded)
    df.loc[_out_of_range(df["bmi"], "bmi"), "bmi"] = np.nan

    rep["missing_after_cleaning"] = {c: int(v) for c, v in df[DIABD_FEATURES].isna().sum().items()}
    rep["class_balance"] = {str(k): int(v) for k, v in df["diabetic"].value_counts().items()}
    rep["hypertensive_balance"] = {str(k): int(v) for k, v in df["hypertensive"].value_counts().items()}
    return df, rep


# ---------------------------------------------------------------- harmonised / general
def load_combined() -> pd.DataFrame:
    """Both datasets mapped onto GENERAL_FEATURES. Attributes a source never recorded
    stay NaN (structurally missing) and are handled by the missing-aware models."""
    p, _ = load_pima()
    d, _ = load_diabd()
    pg = pd.DataFrame({
        "age": p["Age"].astype(float), "gender_male": 0.0, "bmi": p["BMI"],
        "glucose": p["Glucose"], "systolic_bp": np.nan, "diastolic_bp": p["BloodPressure"],
        "family_diabetes": np.nan, "hypertensive": np.nan, "family_hypertension": np.nan,
        "cardiovascular_disease": np.nan, "stroke": np.nan,
        "target": p["Outcome"].astype(int), "source": "pima"})
    dg = d[GENERAL_FEATURES].copy()
    dg["target"] = d["diabetic"].astype(int)
    dg["source"] = "diabd"
    return pd.concat([pg, dg], ignore_index=True)


# ---------------------------------------------------------------- imputer factory
class SafeIterativeImputer(IterativeImputer):
    """IterativeImputer that always works on a writable float copy (pandas 3 copy-on-write
    hands out read-only arrays, which sklearn's in-place initial imputation cannot write to)."""

    def fit(self, X, y=None, **kw):
        return super().fit(np.array(X, dtype=float, copy=True), y, **kw)

    def fit_transform(self, X, y=None, **kw):
        return super().fit_transform(np.array(X, dtype=float, copy=True), y, **kw)

    def transform(self, X):
        return super().transform(np.array(X, dtype=float, copy=True))


def make_mice_imputer() -> IterativeImputer:
    """MICE-style chained-equation imputer + missing-indicator columns."""
    return SafeIterativeImputer(estimator=BayesianRidge(), max_iter=20, tol=1e-3,
                            initial_strategy="median", add_indicator=True,
                            keep_empty_features=True, random_state=RANDOM_STATE)
