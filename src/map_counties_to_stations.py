"""
Step 2a: Map every county in our yield data to its 5 nearest NOAA GHCN-Daily
weather stations, with inverse-distance-weighted (IDW) interpolation weights.

What this does:
    1. Loads county centroids from the Census Bureau's Gazetteer file.
    2. Loads the full GHCN-Daily station catalog + per-station data
       coverage (which years/elements each station actually reports).
    3. Filters to stations with real TMAX/TMIN/PRCP coverage across our
       1990-2025 study period (a nearby station with no data is useless).
    4. For each county, finds the 5 nearest qualifying stations via a
       haversine BallTree (correct for lat/lon; flat Euclidean distance
       is wrong at this scale) and computes IDW weights (1/distance²,
       normalized to sum to 1).

Why IDW over nearest-station or gridded reanalysis (PRISM/nClimGrid):
    Nearest-station breaks down badly for large, sparse western counties
    where the closest station can be 50+ miles away. Gridded reanalysis
    is more accurate but is a different data source/format (raster,
    not the NOAA CDO API this project already scoped) — real added
    complexity for a portfolio project's actual goal. IDW multi-station
    is the defensible middle ground: meaningfully better than
    nearest-station, explainable in one sentence, stays inside NOAA CDO.

Output:
    data/processed/county_station_mapping.parquet / .csv
    Columns: county_fips, rank (1-5), station_id, station_name,
             distance_km, weight (IDW weight, sums to 1.0 per county)

Setup:
    No API token needed for this step — station catalog + inventory are
    public flat files. (You'll need your NOAA CDO token for Step 2b,
    the actual daily weather pull.)
    python map_counties_to_stations.py
"""

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
K_NEAREST_STATIONS = 5

# Study period our yield data covers — used to filter out stations with
# no meaningful overlap in reporting history.
STUDY_START_YEAR = 1990
STUDY_END_YEAR = 2025
MIN_COVERAGE_FIRSTYEAR = 1995  # station must have started by this year...
MIN_COVERAGE_LASTYEAR = 2020   # ...and still been reporting by this year

REQUIRED_ELEMENTS = ["TMAX", "TMIN", "PRCP"]  # needed for GDD + drought index later

GAZETTEER_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_counties_national.zip"
GHCND_STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
GHCND_INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"

CACHE_DIR = Path("data/raw/station_mapping_cache")
YIELD_DATA_PATH = Path("data/processed/county_yields.parquet")
OUTPUT_PATH = Path("data/processed/county_station_mapping.parquet")
OUTPUT_CSV_PATH = Path("data/processed/county_station_mapping.csv")


def _download_cached(url: str, cache_file: Path) -> bytes:
    """Download a file once and cache it — these are large, static reference
    files that don't change often, no reason to refetch on every rerun."""
    if cache_file.exists():
        logger.info(f"Using cached {cache_file.name}")
        return cache_file.read_bytes()

    logger.info(f"Downloading {url} ...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(response.content)
    return response.content


def load_county_centroids(county_fips_filter: set[str]) -> pd.DataFrame:
    """
    Load county centroids from the Census Gazetteer file, restricted to
    the counties that actually appear in our yield data — no reason to
    map stations for counties we'll never join weather data against.
    """
    raw = _download_cached(GAZETTEER_URL, CACHE_DIR / "gazetteer_counties.zip")

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        inner_name = next(n for n in zf.namelist() if n.endswith(".txt"))
        with zf.open(inner_name) as f:
            # Census Gazetteer files are tab-delimited but have inconsistent
            # trailing whitespace on column names across vintages.
            df = pd.read_csv(f, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={"GEOID": "county_fips", "INTPTLAT": "lat", "INTPTLONG": "lon", "NAME": "county_name"})
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df["county_fips"] = df["county_fips"].str.zfill(5)

    df = df[df["county_fips"].isin(county_fips_filter)].reset_index(drop=True)
    logger.info(f"Loaded centroids for {len(df)} counties (filtered from full Gazetteer)")
    return df[["county_fips", "county_name", "lat", "lon"]]


def load_station_catalog() -> pd.DataFrame:
    """
    Parse the fixed-width GHCN-Daily station list.
    Format (columns are 0-indexed, end-exclusive, per NOAA's format spec):
        ID          1-11
        LATITUDE    13-20
        LONGITUDE   22-30
        ELEVATION   32-37
        STATE       39-40
        NAME        42-71
    """
    raw = _download_cached(GHCND_STATIONS_URL, CACHE_DIR / "ghcnd-stations.txt")
    lines = raw.decode("utf-8", errors="replace").splitlines()

    records = []
    for line in lines:
        if len(line) < 71:
            continue
        records.append(
            {
                "station_id": line[0:11].strip(),
                "lat": float(line[12:20].strip()),
                "lon": float(line[21:30].strip()),
                "state": line[38:40].strip(),
                "station_name": line[41:71].strip(),
            }
        )
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} total GHCN-Daily stations (global)")

    # Restrict to US stations — GHCN-Daily station IDs are prefixed by
    # country code (e.g. "US1..."). Cross-border stations could arguably
    # help a handful of northern-border counties, but the added complexity
    # of handling non-US station metadata isn't worth it for a small edge
    # case; documenting this as a known simplification.
    df = df[df["station_id"].str.startswith("US")].reset_index(drop=True)
    logger.info(f"Restricted to {len(df)} US stations")
    return df


def load_qualifying_station_ids() -> set[str]:
    """
    Parse the fixed-width GHCN-Daily inventory file and return the set of
    station IDs that have real coverage of TMAX, TMIN, and PRCP spanning
    our study period. Format:
        ID          1-11
        LATITUDE    13-20
        LONGITUDE   22-30
        ELEMENT     32-35
        FIRSTYEAR   37-40
        LASTYEAR    42-45
    """
    raw = _download_cached(GHCND_INVENTORY_URL, CACHE_DIR / "ghcnd-inventory.txt")
    lines = raw.decode("utf-8", errors="replace").splitlines()

    records = []
    for line in lines:
        if len(line) < 45:
            continue
        records.append(
            {
                "station_id": line[0:11].strip(),
                "element": line[31:35].strip(),
                "firstyear": int(line[36:40].strip()),
                "lastyear": int(line[41:45].strip()),
            }
        )
    inv = pd.DataFrame(records)
    logger.info(f"Loaded {len(inv)} inventory rows covering {inv['station_id'].nunique()} stations")

    inv = inv[
        inv["element"].isin(REQUIRED_ELEMENTS)
        & (inv["firstyear"] <= MIN_COVERAGE_FIRSTYEAR)
        & (inv["lastyear"] >= MIN_COVERAGE_LASTYEAR)
    ]

    # A station only qualifies if it has ALL THREE required elements
    # meeting the coverage bar — partial coverage (e.g. precip only,
    # no temperature) isn't enough for GDD calculation later.
    element_counts = inv.groupby("station_id")["element"].nunique()
    qualifying = set(element_counts[element_counts == len(REQUIRED_ELEMENTS)].index)
    logger.info(
        f"{len(qualifying)} stations have TMAX+TMIN+PRCP coverage "
        f"from <= {MIN_COVERAGE_FIRSTYEAR} through >= {MIN_COVERAGE_LASTYEAR}"
    )
    return qualifying


def build_mapping(counties: pd.DataFrame, stations: pd.DataFrame, k: int = K_NEAREST_STATIONS) -> pd.DataFrame:
    """
    For each county centroid, find the k nearest qualifying stations using
    a haversine BallTree, and compute IDW weights (1/distance²,
    normalized per county to sum to 1).
    """
    station_coords_rad = np.radians(stations[["lat", "lon"]].values)
    tree = BallTree(station_coords_rad, metric="haversine")

    county_coords_rad = np.radians(counties[["lat", "lon"]].values)
    distances_rad, indices = tree.query(county_coords_rad, k=k)
    distances_km = distances_rad * EARTH_RADIUS_KM

    rows = []
    for county_idx, county_row in counties.iterrows():
        for rank in range(k):
            station_row = stations.iloc[indices[county_idx, rank]]
            dist_km = distances_km[county_idx, rank]
            rows.append(
                {
                    "county_fips": county_row["county_fips"],
                    "rank": rank + 1,
                    "station_id": station_row["station_id"],
                    "station_name": station_row["station_name"],
                    "distance_km": dist_km,
                }
            )

    mapping = pd.DataFrame(rows)

    # IDW weight = 1/distance², normalized per county so each county's
    # weights sum to 1. Guard against distance=0 (station sits exactly at
    # the centroid) — extremely unlikely but would divide by zero.
    mapping["distance_km"] = mapping["distance_km"].clip(lower=0.1)
    mapping["inv_dist_sq"] = 1.0 / (mapping["distance_km"] ** 2)
    mapping["weight"] = mapping.groupby("county_fips")["inv_dist_sq"].transform(lambda x: x / x.sum())
    mapping = mapping.drop(columns=["inv_dist_sq"])

    return mapping


def main():
    if not YIELD_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{YIELD_DATA_PATH} not found — run Step 1 (ingest_nass_yield.py) first, "
            "we need it to know which counties to map."
        )

    yield_df = pd.read_parquet(YIELD_DATA_PATH)
    county_fips_filter = set(yield_df["county_fips"].unique())
    logger.info(f"{len(county_fips_filter)} unique counties present in yield data")

    counties = load_county_centroids(county_fips_filter)

    all_stations = load_station_catalog()
    qualifying_ids = load_qualifying_station_ids()
    stations = all_stations[all_stations["station_id"].isin(qualifying_ids)].reset_index(drop=True)
    logger.info(f"{len(stations)} US stations qualify after coverage filtering")

    if len(stations) < K_NEAREST_STATIONS:
        raise RuntimeError(f"Only {len(stations)} qualifying stations found — check coverage thresholds")

    mapping = build_mapping(counties, stations)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(OUTPUT_PATH, index=False)
    mapping.to_csv(OUTPUT_CSV_PATH, index=False)

    logger.info(f"Wrote {len(mapping)} county-station pairs to {OUTPUT_PATH} and {OUTPUT_CSV_PATH}")
    logger.info(f"Counties mapped: {mapping['county_fips'].nunique()} / {len(counties)}")
    logger.info(f"\nDistance summary (km) across all mappings:\n{mapping['distance_km'].describe()}")
    logger.info(f"\nSample mapping (first county):\n{mapping[mapping['county_fips'] == mapping['county_fips'].iloc[0]]}")


if __name__ == "__main__":
    main()