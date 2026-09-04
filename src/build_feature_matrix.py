"""
Step 4: Join county climate with USDA yield data and build the full
feature matrix for modeling.

Pipeline:
  1. Left-join yield (county_fips, year, commodity) with climate
     (county_fips, year) — many yield rows (one per commodity) share
     the same climate row for a given county-year, which is correct.
  2. Within each (county_fips, commodity) time series, sorted by year,
     compute:
       - yield_lag_1/2/3        last 3 years' yields individually
       - yield_roll_mean_3/5    trailing N-year rolling mean yield
       - yield_trend_slope_5    trailing 5-year OLS slope of yield vs year
                                (captures gradual improvement from
                                 genetics/fertilizer/practices, distinct
                                 from short-term lag signal)

Why shift() before rolling():
  A naive `.rolling(3).mean()` on the raw yield column includes the
  CURRENT row in its own 3-year average — meaning the model would be
  fed a feature partially derived from the very value it's trying to
  predict. That's leakage: it looks great in backtesting and is
  useless in production, where you never actually have this year's
  yield when predicting this year's yield. Every lag/rolling/trend
  feature here is computed on `.shift(1)` — yesterday's information
  only, exactly what you'd have at actual prediction time.

Data quality passthrough:
  Rather than silently drop or impute rows with known issues (partial
  2024 wheat data, sparse/missing climate coverage), we carry the
  existing `data_quality` (yield) and `climate_quality` (climate) flags
  straight into the output. Filtering/weighting by these is a modeling
  decision for Step 5, not something to hide upstream.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

YIELD_PATH = Path("data/processed/county_yields.parquet")
CLIMATE_PATH = Path("data/processed/county_climate_annual.parquet")
OUTPUT_PATH = Path("data/processed/feature_matrix.parquet")
OUTPUT_CSV_PATH = Path("data/processed/feature_matrix.csv")

LAG_YEARS = [1, 2, 3]
ROLLING_WINDOWS = [3, 5]
TREND_WINDOW = 5


def compute_trend_slope(series: pd.Series, years: pd.Series) -> pd.Series:
    """
    Trailing TREND_WINDOW-year OLS slope of yield vs year, computed on
    shift(1)'d data so the current year's yield never contributes to its
    own trend feature. Returns NaN wherever fewer than 2 valid prior
    points are available (can't fit a line through <2 points).
    """
    shifted_yield = series.shift(1)
    shifted_year = years.shift(1)

    slopes = pd.Series(np.nan, index=series.index)
    for i in range(len(series)):
        window_start = max(0, i - TREND_WINDOW + 1)
        y_window = shifted_yield.iloc[window_start:i + 1]
        x_window = shifted_year.iloc[window_start:i + 1]

        valid = y_window.notna() & x_window.notna()
        if valid.sum() < 2:
            continue

        slope = np.polyfit(x_window[valid], y_window[valid], deg=1)[0]
        slopes.iloc[i] = slope

    return slopes


def compute_group_features(group: pd.DataFrame) -> dict:
    """
    Compute lag/rolling/trend features for one (county_fips, commodity)
    time series (already sorted by year). Returns a dict of
    {column_name: pd.Series} aligned to `group`'s index, rather than a
    modified copy of the whole group DataFrame.

    Why not return a DataFrame (the more obvious approach): pandas
    changed how groupby().apply() handles the columns you grouped by —
    recent versions strip county_fips/commodity out of what's handed to
    the applied function by default (see the `include_groups` param,
    whose default differs across pandas versions). A function that
    receives and returns the "full group" silently loses those columns
    depending on which pandas is installed — exactly what corrupted the
    first version of this script. Returning a plain dict of new Series
    and assigning them back into `merged` by index (see main()) never
    touches the grouping columns at all, so this bug can't recur
    regardless of pandas version.
    """
    yield_col = group["yield_bu_per_acre"]
    year_col = group["year"]

    features = {}
    for lag in LAG_YEARS:
        features[f"yield_lag_{lag}"] = yield_col.shift(lag)

    # shift(1) FIRST, then rolling — the window looks at the 3 (or 5)
    # years strictly before the current one, never including it.
    shifted = yield_col.shift(1)
    for window in ROLLING_WINDOWS:
        features[f"yield_roll_mean_{window}"] = shifted.rolling(window, min_periods=2).mean()

    features[f"yield_trend_slope_{TREND_WINDOW}"] = compute_trend_slope(yield_col, year_col)

    return features


def main():
    logger.info("Loading yield and climate data...")
    yield_df = pd.read_parquet(YIELD_PATH)
    climate_df = pd.read_parquet(CLIMATE_PATH)
    logger.info(f"  {len(yield_df):,} yield rows, {len(climate_df):,} climate rows")

    logger.info("Joining yield with climate on (county_fips, year)...")
    merged = yield_df.merge(climate_df, on=["county_fips", "year"], how="left")
    logger.info(f"  {len(merged):,} rows after join")

    climate_feature_cols = ["tmax_avg", "tmin_avg", "prcp_total", "gdd", "heat_stress", "prcp_days"]
    unmatched = merged[climate_feature_cols].isna().all(axis=1).sum()
    logger.info(f"  {unmatched:,} yield rows have no matching climate data at all")

    logger.info(f"Computing lag/rolling/trend features per (county_fips, commodity)...")
    merged = merged.sort_values(["county_fips", "commodity", "year"]).reset_index(drop=True)

    lag_cols = [f"yield_lag_{lag}" for lag in LAG_YEARS]
    roll_cols = [f"yield_roll_mean_{w}" for w in ROLLING_WINDOWS]
    trend_col = f"yield_trend_slope_{TREND_WINDOW}"
    new_cols = lag_cols + roll_cols + [trend_col]

    feature_matrix = merged.copy()
    for col in new_cols:
        feature_matrix[col] = np.nan

    n_groups = feature_matrix.groupby(["county_fips", "commodity"], sort=False).ngroups
    logger.info(f"  {n_groups:,} (county_fips, commodity) groups to process")

    for (county, commodity), group in feature_matrix.groupby(["county_fips", "commodity"], sort=False):
        computed = compute_group_features(group)
        for col, series in computed.items():
            feature_matrix.loc[group.index, col] = series

    # Guard against the exact failure this replaced: verify the grouping
    # columns are still present and fully populated before going any
    # further. This should be structurally impossible now (we never touch
    # county_fips/commodity in compute_group_features), but a silent loss
    # of the join key is exactly the kind of bug that doesn't announce
    # itself until much later — worth a hard check, not an assumption.
    for required_col in ["county_fips", "commodity"]:
        if required_col not in feature_matrix.columns:
            raise RuntimeError(f"'{required_col}' missing from feature_matrix — do not proceed")
        if feature_matrix[required_col].isna().any():
            raise RuntimeError(f"'{required_col}' has null values in feature_matrix — do not proceed")

    logger.info("\nFeature coverage (non-null counts):")
    for col in lag_cols + roll_cols + [trend_col]:
        n_valid = feature_matrix[col].notna().sum()
        logger.info(f"  {col}: {n_valid:,} / {len(feature_matrix):,}")

    logger.info("\nData quality flag distribution (yield):")
    logger.info(feature_matrix["data_quality"].value_counts().to_string())
    logger.info("\nData quality flag distribution (climate):")
    logger.info(feature_matrix["climate_quality"].value_counts(dropna=False).to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_parquet(OUTPUT_PATH, index=False)
    feature_matrix.to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info(f"\nSaved {len(feature_matrix):,} rows to {OUTPUT_PATH} and {OUTPUT_CSV_PATH}")

    logger.info("\nSample rows (one county, several years):")
    sample_county = feature_matrix["county_fips"].iloc[0]
    sample_commodity = feature_matrix[feature_matrix["county_fips"] == sample_county]["commodity"].iloc[0]
    sample = feature_matrix[
        (feature_matrix["county_fips"] == sample_county) & (feature_matrix["commodity"] == sample_commodity)
    ].head(8)
    cols_to_show = ["county_fips", "commodity", "year", "yield_bu_per_acre"] + lag_cols + roll_cols + [trend_col]
    logger.info(f"\n{sample[cols_to_show].to_string()}")


if __name__ == "__main__":
    main()