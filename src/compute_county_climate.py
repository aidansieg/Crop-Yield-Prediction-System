"""
Step 3: Compute county-level annual climate features.

Takes 128M daily station observations and produces one row per
county per year with the features our model needs.

Pipeline:
  1. Filter to growing season (April-September)
  2. Compute per-station per-year features from daily data
  3. Join with IDW weights from county_station_mapping.parquet
  4. Weighted-average across stations to get county-level values

Features produced:
  - tmax_avg       Average daily max temp during growing season (°C)
  - tmin_avg       Average daily min temp during growing season (°C)
  - prcp_total     Total precipitation Apr-Sep (mm)
  - gdd            Growing Degree Days (base 10°C, cap 30°C) — corn standard
  - heat_stress    Days where TMAX > 34°C during growing season
  - prcp_days      Number of days with any measurable precipitation

Why GDD base 10°C?
  Growing Degree Days measure accumulated heat above a base temperature
  that represents the minimum for crop growth. Base 10°C is the
  international standard for corn. Each day contributes:
    GDD = max(0, ((min(TMAX, 30) + TMIN) / 2) - 10)
  The cap at 30°C reflects that crops don't benefit from heat above
  that threshold — in fact it becomes damaging, which is captured
  separately in heat_stress.

Why heat_stress separately?
  GDD caps at 30°C so it doesn't capture the damage from extreme heat.
  A day at 38°C and a day at 30°C contribute the same GDD, but the
  38°C day causes real yield damage. heat_stress counts those days
  explicitly so the model can learn from them independently.
"""

import pandas as pd
import numpy as np
import os

CLIMATE_PATH = "data/processed/climate_daily.parquet"
MAPPING_PATH = "data/processed/county_station_mapping.parquet"
OUTPUT_PATH  = "data/processed/county_climate_annual.parquet"

GROW_START_MONTH = 4   # April
GROW_END_MONTH   = 9   # September
GDD_BASE         = 10.0
GDD_CAP          = 30.0
HEAT_THRESHOLD   = 34.0


def compute_station_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    From daily station observations, compute annual growing-season
    features at the station level.

    Input:  long-format daily data (station_id, date, element, value)
    Output: one row per station per year with climate features
    """
    # Filter to growing season months
    daily = daily[
        (daily["date"].dt.month >= GROW_START_MONTH) &
        (daily["date"].dt.month <= GROW_END_MONTH)
    ].copy()

    daily["year"] = daily["date"].dt.year

    # Pivot so each element becomes a column
    # This makes per-day calculations straightforward
    pivoted = daily.pivot_table(
        index=["station_id", "year", "date"],
        columns="element",
        values="value",
        aggfunc="first"  # Should only be one value per station/date/element
    ).reset_index()
    pivoted.columns.name = None

    # Ensure all columns exist even if a station is missing one element
    for col in ["TMAX", "TMIN", "PRCP"]:
        if col not in pivoted.columns:
            pivoted[col] = np.nan

    # ── Per-day derived values ─────────────────────────────────────────────
    # GDD: cap TMAX at GDD_CAP, floor the result at 0
    tmax_capped = pivoted["TMAX"].clip(upper=GDD_CAP)
    pivoted["gdd_day"] = ((tmax_capped + pivoted["TMIN"]) / 2 - GDD_BASE).clip(lower=0)

    # Heat stress: 1 if TMAX exceeds threshold, else 0.
    # NOTE: `pivoted["TMAX"] > HEAT_THRESHOLD` evaluates NaN > x as False,
    # not NaN — so a day with no TMAX reading would otherwise silently
    # become "confirmed not a heat stress day" (0.0) instead of "unknown."
    # np.where preserves the NaN explicitly so a missing reading stays
    # missing all the way through aggregation.
    pivoted["heat_stress_day"] = np.where(
        pivoted["TMAX"].isna(), np.nan, (pivoted["TMAX"] > HEAT_THRESHOLD).astype(float)
    )

    # Precip day: 1 if any measurable precipitation. Same NaN-comparison
    # issue as heat_stress_day above.
    pivoted["prcp_day"] = np.where(
        pivoted["PRCP"].isna(), np.nan, (pivoted["PRCP"] > 0).astype(float)
    )

    # ── Aggregate to station-year ──────────────────────────────────────────
    # sum(min_count=1) — NOT plain "sum" — is required here: pandas' groupby
    # .sum() defaults an all-NaN group to 0.0 rather than NaN (the same
    # failure mode we already hit once in apply_idw_weights). Without
    # min_count=1, a station-year with zero real TMAX/PRCP readings would
    # silently report a fabricated gdd/heat_stress/prcp_total of 0.0
    # instead of correctly propagating "we don't know."
    station_annual = pivoted.groupby(["station_id", "year"]).agg(
        tmax_avg      = ("TMAX",          "mean"),
        tmin_avg      = ("TMIN",          "mean"),
        prcp_total    = ("PRCP",          lambda s: s.sum(min_count=1)),
        gdd           = ("gdd_day",       lambda s: s.sum(min_count=1)),
        heat_stress   = ("heat_stress_day", lambda s: s.sum(min_count=1)),
        prcp_days     = ("prcp_day",      lambda s: s.sum(min_count=1)),
        obs_count     = ("date",          "count"),  # QA: how many days did we observe?
    ).reset_index()

    # Flag station-years with sparse coverage (< 120 of ~183 growing season days)
    station_annual["coverage_ok"] = station_annual["obs_count"] >= 120

    return station_annual


def apply_idw_weights(
    station_annual: pd.DataFrame,
    mapping: pd.DataFrame
) -> pd.DataFrame:
    feature_cols = ["tmax_avg", "tmin_avg", "prcp_total", "gdd", "heat_stress", "prcp_days"]

    merged = mapping.merge(
        station_annual,
        on="station_id",
        how="left"
    ).reset_index(drop=True)

    coverage_ok = merged["coverage_ok"].fillna(False).astype(bool)
    poor_coverage = ~coverage_ok
    for col in feature_cols:
        merged.loc[poor_coverage, col] = np.nan

    # Each feature can have a DIFFERENT set of real contributing stations
    # — e.g. a station may report precipitation reliably while missing
    # temperature entirely that year. Treating "valid" as one flag shared
    # across all six features (the original `.notna().any(axis=1)`) wrongly
    # counted such a station as a full contributor, letting its genuinely
    # missing features silently sum to 0.0 — pandas' groupby .sum() default
    # for an all-NaN group is 0, not NaN (unlike .mean(), which correctly
    # returns NaN). Fix: normalize IDW weights and aggregate PER FEATURE,
    # independently, with min_count=1 so a county-year with zero real
    # contributors for a given feature comes out NaN, not a fabricated 0.0.
    county_annual = None
    stations_used_by_feature = {}

    for col in feature_cols:
        valid = merged[col].notna()
        weight_if_valid = merged["weight"].where(valid)
        weight_sum = weight_if_valid.groupby([merged["county_fips"], merged["year"]]).transform("sum")
        weight_norm = weight_if_valid / weight_sum

        weighted_val = merged[col] * weight_norm
        agg = (
            weighted_val
            .groupby([merged["county_fips"], merged["year"]])
            .sum(min_count=1)  # all-NaN group -> NaN, never a fabricated 0.0
            .rename(col)
        )
        stations_used_by_feature[col] = (
            valid.groupby([merged["county_fips"], merged["year"]]).sum().rename(col)
        )

        county_annual = agg.to_frame() if county_annual is None else county_annual.join(agg)

    # A county-year's overall quality is only as good as its WORST-covered
    # feature — e.g. strong temperature coverage but only 1 station
    # reporting precipitation should still be flagged "sparse", not "good".
    stations_used_df = pd.concat(stations_used_by_feature.values(), axis=1)
    county_annual["stations_used"] = stations_used_df.min(axis=1)

    county_annual = county_annual.reset_index()

    county_annual["climate_quality"] = "good"
    county_annual.loc[county_annual["stations_used"] < 2, "climate_quality"] = "sparse"
    county_annual.loc[county_annual["stations_used"] == 0, "climate_quality"] = "missing"

    # NOTE: no blanket null-out here on purpose. Each feature's per-feature
    # aggregation above (sum(min_count=1)) already independently produces
    # NaN wherever THAT feature had zero real contributing stations —
    # correctly, on a column-by-column basis. `climate_quality` is a
    # confidence LABEL (driven by the worst-covered feature), not a
    # gate on the data itself; blanket-nulling every feature whenever
    # the label says "missing" would destroy genuinely good data in any
    # column that wasn't the reason for the low score (e.g. real
    # precipitation in a county-year where only temperature was missing).

    county_annual["year"] = county_annual["year"].astype(int)

    return county_annual


def main():
    os.makedirs("data/processed", exist_ok=True)

    print("Loading daily climate data...")
    daily = pd.read_parquet(CLIMATE_PATH)
    print(f"  {len(daily):,} records loaded")

    print("\nLoading county-station mapping...")
    mapping = pd.read_parquet(MAPPING_PATH)
    print(f"  {len(mapping):,} county-station pairs")

    print("\nComputing station-level growing season features...")
    print("  (This will take a few minutes — pivoting 128M rows)")
    station_features = compute_station_features(daily)
    print(f"  Done: {len(station_features):,} station-year records")
    print(f"  Coverage OK: {station_features['coverage_ok'].sum():,} / {len(station_features):,}")

    # Free the daily data — we don't need it anymore
    del daily
    print("\nFreed daily climate data from memory")

    print("\nApplying IDW weights to compute county-level features...")
    county_climate = apply_idw_weights(station_features, mapping)
    print(f"  Done: {len(county_climate):,} county-year records")

    print("\nClimate quality summary:")
    print(county_climate["climate_quality"].value_counts().to_string())

    print("\nFeature ranges (sanity check):")
    for col in ["tmax_avg", "tmin_avg", "prcp_total", "gdd", "heat_stress"]:
        print(f"  {col}: min={county_climate[col].min():.1f}  "
              f"max={county_climate[col].max():.1f}  "
              f"mean={county_climate[col].mean():.1f}  "
              f"nulls={county_climate[col].isna().sum()}")

    print(f"\nSample rows:")
    print(county_climate.head(5).to_string())

    county_climate.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()