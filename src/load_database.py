"""
Load parquet data into PostgreSQL.

Creates three tables:
  - yield_records:      raw USDA yield data (from feature matrix)
  - county_climate:     annual climate features per county
  - predictions:        ensemble predictions + anomaly scores

Why load from parquet rather than re-running the pipeline?
  The parquet files are our single source of truth — already cleaned,
  validated, and debugged through 7 steps. Loading from them is faster
  and avoids any risk of reintroducing bugs.

Why these three tables?
  The dashboard needs three distinct query patterns:
    1. Map view: all counties for a given year/commodity (predictions)
    2. County trend: full history for one county (predictions + yield_records)
    3. Anomaly panel: flagged counties sorted by score (predictions)
  Keeping them in separate tables with the right indexes makes all
  three fast without complex joins.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/crop_yield")


SCHEMA = """
-- Drop and recreate for a clean load
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS yield_records CASCADE;
DROP TABLE IF EXISTS county_climate CASCADE;

-- Raw yield records (one row per county/commodity/year)
CREATE TABLE yield_records (
    id              SERIAL PRIMARY KEY,
    county_fips     VARCHAR(5)  NOT NULL,
    county_name     VARCHAR(100),
    state_alpha     VARCHAR(2),
    year            INTEGER     NOT NULL,
    commodity       VARCHAR(20) NOT NULL,
    yield_bu_per_acre FLOAT,
    yield_source    VARCHAR(20),
    data_quality    TEXT
);

-- Annual county-level climate features
CREATE TABLE county_climate (
    id              SERIAL PRIMARY KEY,
    county_fips     VARCHAR(5)  NOT NULL,
    year            INTEGER     NOT NULL,
    tmax_avg        FLOAT,
    tmin_avg        FLOAT,
    prcp_total      FLOAT,
    gdd             FLOAT,
    heat_stress     FLOAT,
    prcp_days       FLOAT,
    stations_used   INTEGER,
    climate_quality VARCHAR(20)
);

-- Ensemble predictions + anomaly scores
CREATE TABLE predictions (
    id                   SERIAL PRIMARY KEY,
    county_fips          VARCHAR(5)   NOT NULL,
    county_name          VARCHAR(100),
    state_alpha          VARCHAR(2),
    year                 INTEGER      NOT NULL,
    commodity            VARCHAR(20)  NOT NULL,
    lgbm_pred            FLOAT,
    prophet_pred         FLOAT,
    ensemble_pred        FLOAT,
    actual               FLOAT,
    data_quality         TEXT,
    prophet_available    BOOLEAN,
    model_disagreement   FLOAT,
    anomaly_score        FLOAT,
    is_anomaly           BOOLEAN,
    yield_lag_1          FLOAT,
    yield_trend_slope_5  FLOAT,
    gdd                  FLOAT,
    heat_stress          FLOAT,
    prcp_total           FLOAT
);

-- Indexes for the three main dashboard query patterns
CREATE INDEX idx_predictions_year_commodity
    ON predictions (year, commodity);

CREATE INDEX idx_predictions_county_commodity
    ON predictions (county_fips, commodity);

CREATE INDEX idx_predictions_anomaly
    ON predictions (year, commodity, is_anomaly)
    WHERE is_anomaly = true;

CREATE INDEX idx_yield_county_commodity
    ON yield_records (county_fips, commodity, year);

CREATE INDEX idx_climate_county_year
    ON county_climate (county_fips, year);
"""


def load_yield_records(engine):
    print("Loading yield records...")
    fm = pd.read_parquet("data/processed/feature_matrix.parquet")

    yield_cols = [
        "county_fips", "county_name", "state_alpha", "year",
        "commodity", "yield_bu_per_acre", "yield_source", "data_quality"
    ]
    df = fm[yield_cols].copy()

    df.to_sql("yield_records", engine, if_exists="append", index=False, chunksize=500)
    print(f"  Loaded {len(df):,} yield records")


def load_climate(engine):
    print("Loading climate features...")
    climate = pd.read_parquet("data/processed/county_climate_annual.parquet")

    climate_cols = [
        "county_fips", "year", "tmax_avg", "tmin_avg",
        "prcp_total", "gdd", "heat_stress", "prcp_days",
        "stations_used", "climate_quality"
    ]

    # Only keep columns that exist
    available = [c for c in climate_cols if c in climate.columns]
    df = climate[available].copy()

    # stations_used should be int
    if "stations_used" in df.columns:
        df["stations_used"] = df["stations_used"].astype("Int64")

    df.to_sql("county_climate", engine, if_exists="append", index=False, chunksize=500)
    print(f"  Loaded {len(df):,} climate records")


def load_predictions(engine):
    print("Loading ensemble predictions...")
    preds = pd.read_parquet("data/processed/ensemble_predictions.parquet")

    pred_cols = [
        "county_fips", "county_name", "state_alpha", "year", "commodity",
        "lgbm_pred", "prophet_pred", "ensemble_pred", "actual", "data_quality",
        "prophet_available", "model_disagreement", "anomaly_score", "is_anomaly",
        "yield_lag_1", "yield_trend_slope_5", "gdd", "heat_stress", "prcp_total"
    ]

    available = [c for c in pred_cols if c in preds.columns]
    df = preds[available].copy()

    # Replace numpy NaN with None for PostgreSQL compatibility
    df = df.where(pd.notna(df), other=None)

    df.to_sql("predictions", engine, if_exists="append", index=False, chunksize=500)
    print(f"  Loaded {len(df):,} prediction records")


def verify(engine):
    print("\nVerifying row counts...")
    with engine.connect() as conn:
        for table in ["yield_records", "county_climate", "predictions"]:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  {table}: {count:,} rows")

        print("\nSample prediction query (Iowa corn 2023):")
        result = conn.execute(text("""
            SELECT county_name, ensemble_pred, actual, is_anomaly
            FROM predictions
            WHERE state_alpha = 'IA'
              AND commodity = 'CORN'
              AND year = 2023
            ORDER BY ensemble_pred DESC
            LIMIT 5
        """))
        for row in result:
            print(f"  {row.county_name}: pred={row.ensemble_pred:.1f} actual={row.actual:.1f} anomaly={row.is_anomaly}")


def main():
    print(f"Connecting to {DATABASE_URL}...")
    engine = create_engine(DATABASE_URL)

    print("Creating schema...")
    with engine.connect() as conn:
        conn.execute(text(SCHEMA))
        conn.commit()
    print("Schema created.")

    load_yield_records(engine)
    load_climate(engine)
    load_predictions(engine)
    verify(engine)

    print("\nDatabase load complete.")


if __name__ == "__main__":
    main()