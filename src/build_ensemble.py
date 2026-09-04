"""
Step 7: Ensemble LightGBM + Prophet predictions + Isolation Forest anomaly detection.

Ensemble logic:
  - When both models have a prediction: 0.65 * lgbm + 0.35 * prophet
  - When only LightGBM has a prediction: use LightGBM only (fallback)
  - When neither: guard with explicit error

Why 65/35?
  LightGBM outperforms Prophet on corn and soybeans. Prophet roughly ties
  on wheat. The blend respects LightGBM's overall edge while capturing the
  genuine trend signal Prophet adds for out-of-range years where tree
  splits can't extrapolate.

Anomaly detection:
  Isolation Forest trained on the training set only (year <= 2018).
  Features are deviations from each row's own historical context, not
  absolute values — a yield of 50 bu/acre is normal for soybeans and
  catastrophic for corn, so everything is expressed as relative deviation.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURE_MATRIX_PATH = "data/processed/feature_matrix.parquet"
PROPHET_PATH        = "data/processed/prophet_predictions.parquet"
LGBM_MODEL_PATH     = "models/lightgbm_baseline.txt"
LGBM_META_PATH      = "models/lightgbm_baseline_metadata.json"
OUTPUT_PATH         = "data/processed/ensemble_predictions.parquet"
ISO_MODEL_PATH      = "models/isolation_forest.joblib"
SCALER_PATH         = "models/anomaly_scaler.joblib"

LGBM_WEIGHT   = 0.65
PROPHET_WEIGHT = 0.35


def load_lgbm_model() -> tuple[lgb.Booster, dict]:
    model = lgb.Booster(model_file=LGBM_MODEL_PATH)
    with open(LGBM_META_PATH) as f:
        meta = json.load(f)
    return model, meta


def prepare_features(fm: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Prepare the feature matrix for LightGBM inference.

    Categorical columns must be cast to pandas Categorical with exactly
    the same levels and order as during training. LightGBM uses the
    category codes (integers) internally — if the levels differ, the
    model silently produces wrong predictions.
    """
    df = fm.copy()

    cat_features = meta["categorical_features"]
    cat_levels   = meta["category_levels"]

    for col in cat_features:
        levels = cat_levels[col]
        df[col] = pd.Categorical(df[col], categories=levels)

    # Select only the columns the model was trained on, in training order
    feature_cols = meta["feature_cols"]
    return df[feature_cols]


def make_lgbm_predictions(
    fm: pd.DataFrame,
    model: lgb.Booster,
    meta: dict
) -> pd.DataFrame:
    """
    Run LightGBM inference on the full feature matrix.
    Returns a DataFrame with identifiers, prediction, actual, and
    features needed downstream for anomaly detection.
    """
    X = prepare_features(fm, meta)
    preds = model.predict(X)

    return pd.DataFrame({
        "county_fips":        fm["county_fips"].values,
        "county_name":        fm["county_name"].values,
        "state_alpha":        fm["state_alpha"].values,
        "commodity":          fm["commodity"].values,
        "year":               fm["year"].values,
        "lgbm_pred":          preds,
        "actual":             fm["yield_bu_per_acre"].values,
        "data_quality":       fm["data_quality"].values,
        # Carry through for anomaly features
        "yield_lag_1":           fm["yield_lag_1"].values,
        "yield_trend_slope_5":   fm["yield_trend_slope_5"].values,
        "gdd":                   fm["gdd"].values,
        "heat_stress":           fm["heat_stress"].values,
        "prcp_total":            fm["prcp_total"].values,
    })


def build_ensemble(
    lgbm_df: pd.DataFrame,
    prophet_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Join LightGBM and Prophet predictions and compute weighted ensemble.

    Prophet is absent for two structural reasons:
      1. insufficient_history (< 5 training years)
      2. County-commodity never appears in Prophet output at all
         (discontinued reporting with no test-set rows in Prophet)
    Both cases fall back to LightGBM-only. We never drop these rows.
    """
    prophet_slim = prophet_df[
        ["county_fips", "commodity", "year", "yhat", "status"]
    ].rename(columns={"yhat": "prophet_pred", "status": "prophet_status"})

    merged = lgbm_df.merge(
        prophet_slim,
        on=["county_fips", "commodity", "year"],
        how="left"
    )

    # Prophet contributes only when status is 'ok' and prediction is present
    merged["prophet_available"] = (
        (merged["prophet_status"] == "ok") &
        merged["prophet_pred"].notna()
    )

    # Weighted blend where Prophet is available, LightGBM-only otherwise
    merged["ensemble_pred"] = np.where(
        merged["prophet_available"],
        LGBM_WEIGHT * merged["lgbm_pred"] + PROPHET_WEIGHT * merged["prophet_pred"],
        merged["lgbm_pred"]
    )

    # Model disagreement — only meaningful when both models contributed
    merged["model_disagreement"] = np.where(
        merged["prophet_available"],
        (merged["lgbm_pred"] - merged["prophet_pred"]).abs(),
        np.nan
    )

    return merged


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for Isolation Forest.

    All features are relative deviations — not absolute values — so the
    detector learns what's anomalous for each county's own history rather
    than flagging low-yield counties as universally anomalous.
    """
    features = pd.DataFrame(index=df.index)

    # Prediction vs last year's actual (pct deviation)
    lag1 = df["yield_lag_1"].replace(0, np.nan)
    features["pred_vs_lag1"] = (df["ensemble_pred"] - lag1) / lag1

    # Recent trend slope (already a relative measure)
    features["yield_trend_slope_5"] = df["yield_trend_slope_5"].fillna(0)

    # Model disagreement normalized by LightGBM prediction magnitude
    lgbm_abs = df["lgbm_pred"].abs().replace(0, np.nan)
    features["model_disagreement_pct"] = (
        df["model_disagreement"] / lgbm_abs
    ).fillna(0)

    # Climate signals — scaler will normalize these across the dataset
    features["gdd"]         = df["gdd"].fillna(df["gdd"].median())
    features["heat_stress"] = df["heat_stress"].fillna(0)
    features["prcp_total"]  = df["prcp_total"].fillna(df["prcp_total"].median())

    return features.fillna(0)


def train_anomaly_detector(
    df: pd.DataFrame,
    contamination: float = 0.05
) -> tuple[IsolationForest, StandardScaler]:
    """
    Fit Isolation Forest on training set rows only (year <= 2018).

    We never fit on test data — that would let the test distribution
    influence what the model considers anomalous.
    """
    train_df = df[df["year"] <= 2018].copy()
    print(f"  Fitting on {len(train_df):,} training rows...")

    features = build_anomaly_features(train_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_scaled)

    return iso, scaler


def score_anomalies(
    df: pd.DataFrame,
    iso: IsolationForest,
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    Score every row. anomaly_score is flipped so higher = more anomalous.
    is_anomaly flags the ~5% most anomalous rows per the fitted threshold.
    """
    df = df.copy()
    features = build_anomaly_features(df)
    X_scaled = scaler.transform(features)

    df["anomaly_score"] = -iso.decision_function(X_scaled)
    df["is_anomaly"]    = iso.predict(X_scaled) == -1

    return df


def evaluate(df: pd.DataFrame, label: str, year_min: int, year_max: int):
    subset = df[
        (df["year"] >= year_min) &
        (df["year"] <= year_max) &
        (df["data_quality"] != "partial")
    ].dropna(subset=["actual", "ensemble_pred"])

    print(f"\n── {label} ({year_min}-{year_max}, {len(subset):,} rows) ───────────────")
    for commodity in ["CORN", "SOYBEANS", "WHEAT"]:
        c = subset[subset["commodity"] == commodity]
        if len(c) < 10:
            continue
        mae = mean_absolute_error(c["actual"], c["ensemble_pred"])
        r2  = r2_score(c["actual"], c["ensemble_pred"])
        rel = mae / c["actual"].mean() * 100
        n_p = c["prophet_available"].sum()
        lgbm_mae = mean_absolute_error(c["actual"], c["lgbm_pred"])
        print(f"  {commodity:10s} MAE={mae:.2f} (lgbm={lgbm_mae:.2f})  "
              f"R²={r2:.3f}  rel={rel:.1f}%  "
              f"prophet={n_p}/{len(c)} ({n_p/len(c)*100:.0f}%)")

    anom = subset["is_anomaly"].sum()
    print(f"  Anomalies: {anom:,} / {len(subset):,} ({anom/len(subset)*100:.1f}%)")


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Loading feature matrix...")
    fm = pd.read_parquet(FEATURE_MATRIX_PATH)
    print(f"  {len(fm):,} rows, {fm['year'].min()}-{fm['year'].max()}")

    print("\nLoading LightGBM model and metadata...")
    model, meta = load_lgbm_model()
    print(f"  Features: {meta['feature_cols']}")

    print("\nRunning LightGBM inference...")
    lgbm_df = make_lgbm_predictions(fm, model, meta)
    print(f"  {len(lgbm_df):,} predictions generated")

    print("\nLoading Prophet predictions...")
    prophet_df = pd.read_parquet(PROPHET_PATH)
    print(f"  {len(prophet_df):,} rows")
    print(f"  Status breakdown: {prophet_df['status'].value_counts().to_dict()}")

    print("\nBuilding ensemble...")
    ensemble_df = build_ensemble(lgbm_df, prophet_df)
    n_both  = ensemble_df["prophet_available"].sum()
    n_total = len(ensemble_df)
    print(f"  Both models: {n_both:,} / {n_total:,} ({n_both/n_total*100:.1f}%)")
    print(f"  LightGBM-only fallback: {n_total - n_both:,} rows")

    print("\nTraining anomaly detector...")
    iso, scaler = train_anomaly_detector(ensemble_df)

    print("\nScoring anomalies on full dataset...")
    ensemble_df = score_anomalies(ensemble_df, iso, scaler)

    # Evaluate on validation and test sets
    evaluate(ensemble_df, "Validation", 2019, 2020)
    evaluate(ensemble_df, "Test",       2021, 2025)

    # Save outputs
    print(f"\nSaving ensemble predictions → {OUTPUT_PATH}")
    ensemble_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saving anomaly detector → {ISO_MODEL_PATH}")
    joblib.dump(iso, ISO_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # Show top anomalies in test set
    print("\nTop 10 most anomalous counties in test set (2021-2025):")
    top = (
        ensemble_df[
            (ensemble_df["year"] >= 2021) &
            ensemble_df["is_anomaly"]
        ]
        .sort_values("anomaly_score", ascending=False)
        .head(10)
    )[["county_name", "state_alpha", "commodity", "year",
       "actual", "ensemble_pred", "lgbm_pred", "prophet_pred",
       "anomaly_score", "model_disagreement"]]
    print(top.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()