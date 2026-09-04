"""
Step 6: Per-county-commodity Prophet trend models.

Design:

  One independent Prophet model per (county_fips, commodity) — 7,013
  groups from Step 4. Prophet is deliberately univariate here (yield
  history only, no climate features): its job in the ensemble is the
  smooth, decades-long trend from genetics/fertilizer/practice
  improvements that a gradient-boosted tree structurally can't
  extrapolate past its training range (see Step 5's `year` feature
  limitation). LightGBM already owns the year-to-year, climate-driven
  variation — Prophet is there to complement it, not duplicate it.

  Seasonality disabled entirely. Prophet's yearly/weekly/daily
  seasonality components model WITHIN-period patterns from sub-period
  data. With exactly one observation per year, there is no within-year
  pattern to find — leaving seasonality on would fit noise and call it
  signal. Trend only.

  Same time split as Step 5, for a fair, apples-to-apples ensemble
  comparison later: train <=2018, predict 2019-2025. Unlike LightGBM,
  Prophet doesn't need a validation set for early stopping, but we
  keep the 2019-2020 / 2021-2025 split anyway — 2019-2020 to tune
  Step 7's ensemble blend weights, 2021-2025 as the final held-out test.

  Minimum training history: 5 years. Fitting a trend through 2-3 points
  is close to meaningless, and is exactly the kind of "technically
  produces a number" result this project has repeatedly caught and
  fixed (Steps 3-4). Groups below this are skipped and flagged
  `insufficient_history`, not fit with a low-confidence line — Step 7's
  ensemble will lean entirely on LightGBM for those rows.

Output: data/processed/prophet_predictions.parquet
  county_fips, commodity, year, yhat, yhat_lower, yhat_upper,
  actual yield, status, n_train_years
"""

import contextlib
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Prophet/cmdstanpy are extremely verbose per-fit — with 7,000+ fits this
# would flood the log. Quiet everything except our own messages.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

FEATURE_MATRIX_PATH = Path("data/processed/feature_matrix.parquet")
OUTPUT_PATH = Path("data/processed/prophet_predictions.parquet")
OUTPUT_CSV_PATH = Path("data/processed/prophet_predictions.csv")

TRAIN_END_YEAR = 2018      # same cutoff as Step 5, for a fair comparison
VAL_END_YEAR = 2020
MIN_TRAIN_YEARS = 5        # groups with fewer real training years are skipped
N_JOBS = -1                # all available cores; lower this if resource-constrained


def fit_and_predict_one_group(county_fips: str, commodity: str, group: pd.DataFrame) -> pd.DataFrame:
    """
    Fit one Prophet model on a single (county_fips, commodity) series'
    training years, predict all validation+test years, and return a
    tidy result frame — regardless of whether fitting was possible.
    """
    from prophet import Prophet  # imported inside the worker for joblib multiprocessing safety

    train = group[group["year"] <= TRAIN_END_YEAR].dropna(subset=["yield_bu_per_acre"])
    predict_years = group[group["year"] > TRAIN_END_YEAR]["year"].unique()

    result_rows = []

    if len(predict_years) == 0:
        # This group has training history but nothing after TRAIN_END_YEAR
        # at all — e.g. NASS discontinued county-level reporting for this
        # specific county/commodity. Nothing to predict or score here;
        # skip cleanly rather than hand Prophet an empty future frame.
        return pd.DataFrame(columns=[
            "county_fips", "commodity", "year", "yhat", "yhat_lower", "yhat_upper",
            "actual", "status", "n_train_years",
        ])

    if len(train) < MIN_TRAIN_YEARS:
        for year in predict_years:
            actual_row = group[group["year"] == year]
            actual = actual_row["yield_bu_per_acre"].iloc[0] if len(actual_row) else np.nan
            result_rows.append({
                "county_fips": county_fips, "commodity": commodity, "year": year,
                "yhat": np.nan, "yhat_lower": np.nan, "yhat_upper": np.nan,
                "actual": actual, "status": "insufficient_history",
                "n_train_years": len(train),
            })
        return pd.DataFrame(result_rows)

    prophet_train = pd.DataFrame({
        "ds": pd.to_datetime(train["year"], format="%Y"),
        "y": train["yield_bu_per_acre"].values,
    })

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        growth="linear",
    )

    # Prophet/cmdstanpy print raw Stan sampler output straight to stdout,
    # bypassing the logging module entirely — redirect it away so 7,000+
    # fits don't flood the terminal.
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        model.fit(prophet_train)

    future = pd.DataFrame({"ds": pd.to_datetime(predict_years, format="%Y")})
    forecast = model.predict(future)

    for i, year in enumerate(predict_years):
        actual_row = group[group["year"] == year]
        actual = actual_row["yield_bu_per_acre"].iloc[0] if len(actual_row) else np.nan
        result_rows.append({
            "county_fips": county_fips, "commodity": commodity, "year": year,
            "yhat": forecast["yhat"].iloc[i],
            "yhat_lower": forecast["yhat_lower"].iloc[i],
            "yhat_upper": forecast["yhat_upper"].iloc[i],
            "actual": actual, "status": "ok",
            "n_train_years": len(train),
        })

    return pd.DataFrame(result_rows)


def main():
    logger.info("Loading feature matrix...")
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df = df[df["data_quality"] == "complete"].copy()  # same label-quality filter as Step 5
    logger.info(f"  {len(df):,} rows after dropping known-wrong labels")

    groups = list(df.groupby(["county_fips", "commodity"], observed=True))
    logger.info(f"  {len(groups):,} (county_fips, commodity) groups to fit")

    logger.info(f"\nFitting Prophet models (n_jobs={N_JOBS})... this will take a while.")
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(fit_and_predict_one_group)(county_fips, commodity, group)
        for (county_fips, commodity), group in groups
    )

    predictions = pd.concat(results, ignore_index=True)
    predictions["year"] = predictions["year"].astype(int)

    n_groups_total = len(groups)
    n_groups_with_output = predictions[["county_fips", "commodity"]].drop_duplicates().shape[0]
    n_groups_no_future_data = n_groups_total - n_groups_with_output
    logger.info(
        f"\n{n_groups_no_future_data:,} / {n_groups_total:,} groups had no rows after "
        f"{TRAIN_END_YEAR} at all (discontinued reporting) — skipped, nothing to score"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUTPUT_PATH, index=False)
    predictions.to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info(f"\nSaved {len(predictions):,} predictions to {OUTPUT_PATH}")

    logger.info("\nStatus breakdown:")
    logger.info(predictions["status"].value_counts().to_string())

    for split_name, year_filter in [
        ("Validation (2019-2020)", (predictions["year"] > TRAIN_END_YEAR) & (predictions["year"] <= VAL_END_YEAR)),
        ("Test (2021-2025)", predictions["year"] > VAL_END_YEAR),
    ]:
        split_df = predictions[year_filter & (predictions["status"] == "ok")].dropna(subset=["yhat", "actual"])
        logger.info(f"\n{split_name} — {len(split_df):,} scoreable rows:")
        mae = mean_absolute_error(split_df["actual"], split_df["yhat"])
        rmse = np.sqrt(mean_squared_error(split_df["actual"], split_df["yhat"]))
        r2 = r2_score(split_df["actual"], split_df["yhat"])
        logger.info(f"  Overall: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")

        for commodity, group in split_df.groupby("commodity", observed=True):
            c_mae = mean_absolute_error(group["actual"], group["yhat"])
            c_rmse = np.sqrt(mean_squared_error(group["actual"], group["yhat"]))
            c_r2 = r2_score(group["actual"], group["yhat"])
            logger.info(f"    {commodity}: MAE={c_mae:.2f}  RMSE={c_rmse:.2f}  R2={c_r2:.4f}  (n={len(group):,})")


if __name__ == "__main__":
    main()