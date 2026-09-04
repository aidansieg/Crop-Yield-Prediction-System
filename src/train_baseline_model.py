"""
Step 5: Train a LightGBM baseline crop yield model.

Design decisions:

  Time-based train/val/test split (NOT random):
    Train <= 2018, validate 2019-2020 (early stopping), test 2021-2025.
    A random split would let a 2020 row's lag features sit statistically
    close to its own neighbors in training purely by chance, producing an
    inflated score that says nothing about real forecasting ability. In
    production you always predict years you haven't seen — the eval
    should simulate exactly that.

  Label quality filter (drop, don't impute):
    Rows with data_quality == "partial" (2024 spring/durum wheat, missing
    at the county level as of ingestion) have a KNOWN-WRONG target value,
    not just an uncertain one — winter-wheat-only yield understates the
    true combined figure. Training on a wrong label teaches the model
    something false. 57 of 170,317 rows; dropping costs nothing.

  Feature quality (pass through, let LightGBM handle it natively):
    Climate features with missing/sparse station coverage are left as
    real NaN, not imputed. LightGBM learns the optimal split direction
    for missing values from the data itself during training — better
    than any fill-value heuristic we could pick by hand, and consistent
    with every fix made in Step 3 (an honest "unknown" beats a fabricated
    plausible number). climate_quality is passed in as a feature so the
    model can learn to weight low-confidence climate differently, rather
    than us deciding that upfront.

  Single global model across all three commodities:
    `commodity` is a categorical feature rather than training 3 separate
    models. LightGBM splits on commodity early and effectively learns
    per-crop behavior from there, while keeping the pipeline (and the
    interview explanation) simpler, at the cost of nothing meaningful
    given tree-based models handle this well.

  county_fips included as a (high-cardinality, ~2700-level) categorical:
    Captures location-specific baseline yield differences not otherwise
    explained by climate/lag features — especially useful in a county's
    early years before lag features exist. Worth watching in the feature
    importance output; if it dominates suspiciously or overfits, the
    fix is to drop it or replace with a coarser (state-level) identifier.
"""

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURE_MATRIX_PATH = Path("data/processed/feature_matrix.parquet")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "lightgbm_baseline.txt"
METADATA_PATH = MODEL_DIR / "lightgbm_baseline_metadata.json"

TRAIN_END_YEAR = 2018   # inclusive
VAL_END_YEAR = 2020     # inclusive; (TRAIN_END_YEAR, VAL_END_YEAR] is validation
# test is everything after VAL_END_YEAR

NUMERIC_FEATURES = [
    "yield_lag_1", "yield_lag_2", "yield_lag_3",
    "yield_roll_mean_3", "yield_roll_mean_5",
    "yield_trend_slope_5",
    "tmax_avg", "tmin_avg", "prcp_total", "gdd", "heat_stress", "prcp_days",
    "year",
]
CATEGORICAL_FEATURES = ["commodity", "climate_quality", "county_fips"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "yield_bu_per_acre"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading feature matrix...")
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    logger.info(f"  {len(df):,} rows loaded")

    n_before = len(df)
    df = df[df["data_quality"] == "complete"].copy()
    logger.info(f"  Dropped {n_before - len(df):,} rows with known-wrong yield labels (partial data_quality)")

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    train_df = df[df["year"] <= TRAIN_END_YEAR]
    val_df = df[(df["year"] > TRAIN_END_YEAR) & (df["year"] <= VAL_END_YEAR)]
    test_df = df[df["year"] > VAL_END_YEAR]

    logger.info(
        f"\nTrain: {len(train_df):,} rows ({int(train_df['year'].min())}-{int(train_df['year'].max())})"
    )
    logger.info(
        f"Val:   {len(val_df):,} rows ({int(val_df['year'].min())}-{int(val_df['year'].max())})"
    )
    logger.info(
        f"Test:  {len(test_df):,} rows ({int(test_df['year'].min())}-{int(test_df['year'].max())})"
    )

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )

    logger.info("\nTraining...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)],
    )
    logger.info(f"Best iteration: {model.best_iteration_}")

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    logger.info(f"\nTest set performance ({len(test_df):,} rows, {int(test_df['year'].min())}-{int(test_df['year'].max())}):")
    logger.info(f"  MAE:  {mae:.2f} bu/acre")
    logger.info(f"  RMSE: {rmse:.2f} bu/acre")
    logger.info(f"  R2:   {r2:.4f}")

    logger.info("\nPer-commodity test performance:")
    test_eval = test_df.copy()
    test_eval["pred"] = preds
    for commodity, group in test_eval.groupby("commodity", observed=True):
        c_mae = mean_absolute_error(group[TARGET_COL], group["pred"])
        c_rmse = np.sqrt(mean_squared_error(group[TARGET_COL], group["pred"]))
        c_r2 = r2_score(group[TARGET_COL], group["pred"])
        logger.info(f"  {commodity}: MAE={c_mae:.2f}  RMSE={c_rmse:.2f}  R2={c_r2:.4f}  (n={len(group):,})")

    importance = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    logger.info("\nFeature importance (split count):")
    logger.info(importance.to_string())

    model.booster_.save_model(str(MODEL_PATH))
    metadata = {
        "feature_cols": FEATURE_COLS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_col": TARGET_COL,
        "train_end_year": TRAIN_END_YEAR,
        "val_end_year": VAL_END_YEAR,
        "category_levels": {
            col: df[col].cat.categories.tolist() for col in CATEGORICAL_FEATURES
        },
        "best_iteration": int(model.best_iteration_),
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_r2": float(r2),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nModel saved to {MODEL_PATH}")
    logger.info(f"Metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()