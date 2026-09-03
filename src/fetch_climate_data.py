"""
Step 2b: Fetch daily climate data from NOAA GHCN-Daily bulk files.

For each of the 3,699 stations in our county mapping, we download the
full station CSV from NOAA's public S3 bucket, filter to TMAX/TMIN/PRCP
between 1990-2025, exclude quality-flagged records, convert units, and
save a single clean parquet file.

Why bulk S3 instead of NOAA CDO API?
The CDO API has a 1,000 requests/day rate limit. At 3,699 stations x 36
years = 133,164 chunks, that would take 134 days. The S3 bulk files are
the same underlying data with no rate limit — one file per station,
entire history in a single download.

Units after conversion:
  TMAX, TMIN: tenths of degrees Celsius -> degrees Celsius (divide by 10)
  PRCP:       tenths of mm -> mm (divide by 10)
"""

import pandas as pd
import requests
import os
import time
from tqdm import tqdm
from io import StringIO

# ── Config ────────────────────────────────────────────────────────────────────
MAPPING_PATH  = "data/processed/county_station_mapping.parquet"
OUTPUT_PATH   = "data/processed/climate_daily.parquet"
PROGRESS_PATH = "data/processed/climate_fetch_progress.txt"
BASE_URL      = "https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/{station_id}.csv"
START_YEAR    = 1990
END_YEAR      = 2025
ELEMENTS      = {"TMAX", "TMIN", "PRCP"}


def load_completed_stations() -> set:
    """Track which stations we've already downloaded so we can resume."""
    if not os.path.exists(PROGRESS_PATH):
        return set()
    with open(PROGRESS_PATH) as f:
        return set(line.strip() for line in f if line.strip())


def mark_completed(station_id: str):
    with open(PROGRESS_PATH, "a") as f:
        f.write(station_id + "\n")


def fetch_station(station_id: str) -> pd.DataFrame | None:
    """
    Download one station's full history and return filtered records.
    Returns None if the download fails.
    """
    url = BASE_URL.format(station_id=station_id)
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        df = pd.read_csv(
            StringIO(resp.text),
            header=0,
            names=["station_id", "date", "element", "value", "m_flag", "q_flag", "s_flag", "obs_time"],
            dtype={"date": str, "value": "Int64", "q_flag": str},
            skiprows=1
        )

        # Filter to elements we care about
        df = df[df["element"].isin(ELEMENTS)]

        # Drop quality-flagged records (non-empty q_flag = NOAA flagged as bad)
        df = df[df["q_flag"].isna() | (df["q_flag"] == "")]

        # Parse date and filter to our year range
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[(df["date"].dt.year >= START_YEAR) & (df["date"].dt.year <= END_YEAR)]

        if df.empty:
            return None

        # Convert units: tenths -> actual values
        df["value"] = df["value"] / 10.0

        return df[["station_id", "date", "element", "value"]].copy()

    except Exception as e:
        print(f"\n  Warning: failed to fetch {station_id}: {e}")
        return None


def main():
    os.makedirs("data/processed", exist_ok=True)

    # Load the list of stations we need
    mapping = pd.read_parquet(MAPPING_PATH)
    all_stations = mapping["station_id"].unique().tolist()
    print(f"Total stations to fetch: {len(all_stations)}")

    # Resume support — skip already-downloaded stations
    completed = load_completed_stations()
    remaining = [s for s in all_stations if s not in completed]
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All stations already fetched.")
        return

    # Load existing data if we're resuming
    all_frames = []
    if os.path.exists(OUTPUT_PATH) and completed:
        print("Loading existing data to append to...")
        all_frames.append(pd.read_parquet(OUTPUT_PATH))

    # Fetch remaining stations
    failed = []
    for station_id in tqdm(remaining, desc="Fetching stations"):
        df = fetch_station(station_id)

        if df is not None and not df.empty:
            all_frames.append(df)
            mark_completed(station_id)
        else:
            failed.append(station_id)
            mark_completed(station_id)  # Mark failed too so we don't retry endlessly

        # Save checkpoint every 100 stations so we don't lose progress
        if len(all_frames) % 100 == 0 and all_frames:
            combined = pd.concat(all_frames, ignore_index=True)
            combined.to_parquet(OUTPUT_PATH, index=False)

        time.sleep(0.05)  # Polite pause — S3 has no rate limit but be reasonable

    # Final save
    if all_frames:
        print("\nSaving final dataset...")
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_parquet(OUTPUT_PATH, index=False)
        print(f"Saved {len(combined):,} records to {OUTPUT_PATH}")
        print(f"Stations fetched: {combined['station_id'].nunique():,}")
        print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
        print(f"Elements: {combined['element'].value_counts().to_dict()}")
    else:
        print("No data fetched.")

    if failed:
        print(f"\nFailed stations ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    main()