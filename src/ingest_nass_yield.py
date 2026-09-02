"""
Step 1: Ingest county-level crop yield data from the USDA NASS QuickStats API.

What this does:
    Pulls annual SURVEY yield data (bushels/acre) at the COUNTY level for
    corn, soybeans, and wheat, from 1990 to present, and writes it to a
    local Parquet cache + a tidy CSV.

Why it's structured this way:
    - Chunked by year: NASS rate-limits and can time out on huge requests.
      Pulling one year at a time also means a failed request only costs
      you one retry, not the whole historical pull.
    - Raw JSON cached to disk: lets you re-run the parsing/cleaning logic
      as you iterate without re-hitting the API and burning your quota.
      Only SUCCESSFUL responses are cached — a failed request is never
      written to disk, so a rerun retries it instead of treating it as
      permanently empty.
    - Wheat schema change (confirmed via NASS's get_param_values endpoint):
      NASS published a single combined "WHEAT - YIELD, MEASURED IN BU / ACRE"
      series at the county level through 2007. From 2008 onward, that
      combined series does not exist — only class-level series do
      (WHEAT, WINTER / WHEAT, SPRING, (EXCL DURUM) / WHEAT, SPRING, DURUM).
      For 2008+, we derive a comparable combined yield ourselves: sum each
      class's production (bu) and acres harvested per county-year, then
      divide (production-weighted average yield). A straight average of
      the three classes' bu/acre would be wrong — it ignores that one
      class may dominate a given county's planted acreage. Every row is
      tagged with `yield_source` ("direct" vs "derived_weighted") so this
      methodology difference is visible downstream rather than hidden.

Setup:
    1. Get a free API key: https://quickstats.nass.usda.gov/api
    2. export NASS_API_KEY="your_key_here"
    3. python ingest_nass_yield.py
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

NASS_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
API_KEY = os.environ.get("NASS_API_KEY")


class NassApiError(Exception):
    """Raised when NASS returns an error we don't recognize as 'legitimately no data'."""


# Commodities pulled directly via a single combined yield series.
# Wheat is handled separately below because that combined series stops
# existing after 2007 (see module docstring).
COMMODITIES = {
    "CORN": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
    "SOYBEANS": "SOYBEANS - YIELD, MEASURED IN BU / ACRE",
}

WHEAT_DIRECT_SHORT_DESC = "WHEAT - YIELD, MEASURED IN BU / ACRE"
WHEAT_DIRECT_LAST_YEAR = 2007  # last year the combined series exists
WHEAT_DERIVED_FIRST_YEAR = 2008  # first year we must derive it from class-level series

# The three wheat classes NASS breaks county-level data into from 2008 on.
# Confirmed via: GET get_param_values?param=short_desc&commodity_desc=WHEAT&agg_level_desc=COUNTY&year=2008
WHEAT_CLASSES = ["WHEAT, WINTER", "WHEAT, SPRING, (EXCL DURUM)", "WHEAT, SPRING, DURUM"]

START_YEAR = 1990
END_YEAR = 2025  # most recent full survey year available; adjust as new data is published

CACHE_DIR = Path("data/raw/nass_cache")
OUTPUT_PATH = Path("data/processed/county_yields.parquet")
OUTPUT_CSV_PATH = Path("data/processed/county_yields.csv")

REQUEST_DELAY_SECONDS = 1.0  # be a polite API citizen; NASS will 429 you if you hammer it


def _validate_api_key() -> None:
    if not API_KEY:
        raise EnvironmentError(
            "NASS_API_KEY is not set. Get a free key at "
            "https://quickstats.nass.usda.gov/api and run:\n"
            "    export NASS_API_KEY='your_key_here'"
        )


def fetch_year(short_desc: str, year: int) -> Optional[list[dict]]:
    """
    Pull county-level NASS data for one exact short_desc statistic and year.

    Returns the list of record dicts from NASS, or None if nothing was
    legitimately published for this exact query that year.
    """
    params = {
        "key": API_KEY,
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "commodity_desc": short_desc.split(",")[0].split(" - ")[0].strip(),
        "short_desc": short_desc,
        "agg_level_desc": "COUNTY",
        "year": year,
        "format": "JSON",
    }

    response = requests.get(NASS_BASE_URL, params=params, timeout=30)

    # NASS returns 400 (not 200-with-empty-body) when a query legitimately
    # has zero matching rows. It puts the reason in the response body, so
    # we surface that instead of guessing — "no data" wording specifically
    # means "nothing published for this exact query", which is a real,
    # expected condition. Any OTHER 400/error body means something is
    # actually wrong with the request itself, and we should not silently
    # swallow it (this is how we diagnosed the wheat schema change).
    if response.status_code == 400:
        if "no data" in response.text.lower() or "no matching" in response.text.lower():
            logger.info(f"  {short_desc[:40]}... {year}: no data published for this query")
            return None
        raise NassApiError(f"{short_desc} {year}: unexpected 400 response — {response.text[:300]}")

    response.raise_for_status()
    payload = response.json()
    return payload.get("data", [])


def fetch_cached(short_desc: str, year: int, cache_file: Path) -> list[dict]:
    """
    Fetch one (short_desc, year) query, transparently caching successful
    results to disk. Failures are never cached, so a rerun retries them.
    Raises (requests.HTTPError, NassApiError) on failure — caller decides
    how to handle it.
    """
    if cache_file.exists():
        with open(cache_file) as f:
            records = json.load(f)
        logger.info(f"  {short_desc[:40]}... {year}: loaded {len(records)} records from cache")
        return records

    records = fetch_year(short_desc, year) or []
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(records, f)
    logger.info(f"  {short_desc[:40]}... {year}: fetched {len(records)} records")
    time.sleep(REQUEST_DELAY_SECONDS)
    return records


def fetch_commodity_direct(commodity_name: str, short_desc: str, start_year: int, end_year: int) -> list[dict]:
    """Loop year-by-year for one commodity's single combined yield series."""
    all_records = []
    commodity_cache_dir = CACHE_DIR / commodity_name.lower()

    for year in range(start_year, end_year + 1):
        cache_file = commodity_cache_dir / f"{year}.json"
        try:
            records = fetch_cached(short_desc, year, cache_file)
        except (requests.HTTPError, NassApiError) as e:
            logger.warning(f"  {commodity_name} {year}: request failed ({e}) — NOT cached, will retry next run")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue
        all_records.extend(records)

    return all_records


def clean_direct_records(records: list[dict], commodity_name: str) -> pd.DataFrame:
    """
    Convert raw NASS records (single combined series) into a tidy DataFrame:
    county_fips | county_name | state_alpha | year | commodity | yield_bu_per_acre | yield_source
    """
    columns = ["county_fips", "county_name", "state_alpha", "year", "commodity", "yield_bu_per_acre", "yield_source", "data_quality"]
    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)

    # NASS zero-pads state and county FIPS codes separately; concatenate
    # into the standard 5-digit county FIPS used by every other public
    # dataset you'll join against (NOAA, Census, TIGER shapefiles).
    df["county_fips"] = df["state_fips_code"].str.zfill(2) + df["county_code"].str.zfill(3)

    # NASS reserves county_code "998" as a synthetic "OTHER (COMBINED)
    # COUNTIES" pseudo-county: when individual small counties' data is
    # disclosure-suppressed, NASS sometimes still publishes their SUM
    # under this fake FIPS code rather than dropping it entirely. It
    # doesn't correspond to any real geography (no centroid, no weather
    # station mapping possible) and blends multiple counties' yields
    # into one row, so it must never reach per-county modeling.
    n_before = len(df)
    df = df[~df["county_code"].str.zfill(3).eq("998")]
    if n_before != len(df):
        logger.info(f"  dropped {n_before - len(df)} 'OTHER (COMBINED) COUNTIES' pseudo-county rows (county_code=998)")

    # NASS uses "Value" as a string and marks suppressed/withheld data with
    # non-numeric placeholders (e.g. "(D)" for disclosure-suppressed).
    # Coercing to numeric and dropping NaNs removes those cleanly.
    df["yield_bu_per_acre"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["yield_bu_per_acre"])

    df["year"] = df["year"].astype(int)
    df["commodity"] = commodity_name
    df["yield_source"] = "direct"
    df["data_quality"] = "complete"

    result: pd.DataFrame = df[columns]
    return result


def fetch_wheat_class_component(stat: str, wheat_class: str, year: int) -> pd.DataFrame:
    """
    Fetch one wheat class's production or acres-harvested for one year,
    returned as a tidy DataFrame keyed by county_fips.

    stat: "production" or "acres_harvested"
    """
    short_desc = f"{wheat_class} - PRODUCTION, MEASURED IN BU" if stat == "production" \
        else f"{wheat_class} - ACRES HARVESTED"

    class_slug = wheat_class.lower().replace(", ", "_").replace(" ", "_").replace("(", "").replace(")", "")
    cache_file = CACHE_DIR / "wheat_derived" / stat / class_slug / f"{year}.json"

    try:
        records = fetch_cached(short_desc, year, cache_file)
    except (requests.HTTPError, NassApiError) as e:
        logger.warning(f"  {wheat_class} {stat} {year}: request failed ({e}) — NOT cached, will retry next run")
        time.sleep(REQUEST_DELAY_SECONDS)
        records = []

    if not records:
        return pd.DataFrame(columns=["county_fips", "value"])

    df = pd.DataFrame(records)
    df["county_fips"] = df["state_fips_code"].str.zfill(2) + df["county_code"].str.zfill(3)
    # Same "998 = OTHER (COMBINED) COUNTIES" pseudo-county issue as the
    # direct path — must be excluded here too, since this function feeds
    # the derived wheat yield the same way.
    df = df[~df["county_code"].str.zfill(3).eq("998")]
    df["value"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["value"])

    result: pd.DataFrame = df[["county_fips", "value"]]
    return result


def fetch_wheat_derived_year(year: int) -> pd.DataFrame:
    """
    Build a production-weighted combined wheat yield per county for one
    year, from the three class-level series (winter / spring excl durum /
    spring durum). yield = sum(production across classes) / sum(acres
    harvested across classes), per county.

    NASS's final county-level estimates for a given crop year sometimes
    lag the harvest by more than a year, and classes don't always publish
    on the same schedule (e.g. 2024: winter wheat published, spring/durum
    did not, as of this ingestion). Rather than hardcode a specific "bad"
    year, we detect it generically: if any class returns zero records for
    a year, that year's combined figure is missing that class's
    contribution and is tagged `data_quality = "partial"` so it's never
    silently treated as equivalent to a normal, fully-published year.
    """
    production_frames, acres_frames = [], []
    missing_classes = []
    for wheat_class in WHEAT_CLASSES:
        prod = fetch_wheat_class_component("production", wheat_class, year)
        acr = fetch_wheat_class_component("acres_harvested", wheat_class, year)
        if prod.empty and acr.empty:
            missing_classes.append(wheat_class)
        production_frames.append(prod)
        acres_frames.append(acr)

    data_quality = "partial (missing: " + ", ".join(missing_classes) + ")" if missing_classes else "complete"
    if missing_classes:
        logger.warning(f"  WHEAT {year}: no data for {missing_classes} — combined yield will UNDERSTATE true county yield")

    production = pd.concat(production_frames, ignore_index=True)
    acres = pd.concat(acres_frames, ignore_index=True)

    if production.empty or acres.empty:
        return pd.DataFrame(
            columns=["county_fips", "county_name", "state_alpha", "year", "commodity", "yield_bu_per_acre", "yield_source", "data_quality"]
        )

    total_production = production.groupby("county_fips")["value"].sum().rename("total_production_bu")
    total_acres = acres.groupby("county_fips")["value"].sum().rename("total_acres_harvested")

    combined = pd.concat([total_production, total_acres], axis=1).dropna()
    combined = combined[combined["total_acres_harvested"] > 0]  # avoid divide-by-zero
    combined["yield_bu_per_acre"] = combined["total_production_bu"] / combined["total_acres_harvested"]

    combined = combined.reset_index()
    combined["year"] = year
    combined["commodity"] = "WHEAT"
    combined["yield_source"] = "derived_weighted"
    combined["data_quality"] = data_quality
    # county_name / state_alpha aren't carried by this derivation path;
    # left as NaN here and can be joined back from a FIPS lookup table
    # later if the dashboard needs display names for these county-years.
    combined["county_name"] = None
    combined["state_alpha"] = None

    result: pd.DataFrame = combined[
        ["county_fips", "county_name", "state_alpha", "year", "commodity", "yield_bu_per_acre", "yield_source", "data_quality"]
    ]
    return result


def fetch_wheat_full_series() -> pd.DataFrame:
    """Direct combined series through 2007, derived production-weighted series from 2008 on."""
    direct_records = fetch_commodity_direct("WHEAT", WHEAT_DIRECT_SHORT_DESC, START_YEAR, WHEAT_DIRECT_LAST_YEAR)
    direct_df = clean_direct_records(direct_records, "WHEAT")
    logger.info(f"WHEAT (direct, {START_YEAR}-{WHEAT_DIRECT_LAST_YEAR}): {len(direct_df)} clean county-year records")

    derived_frames = []
    for year in range(WHEAT_DERIVED_FIRST_YEAR, END_YEAR + 1):
        year_df = fetch_wheat_derived_year(year)
        logger.info(f"  WHEAT (derived) {year}: {len(year_df)} county records")
        derived_frames.append(year_df)
    derived_df = pd.concat(derived_frames, ignore_index=True)
    logger.info(
        f"WHEAT (derived_weighted, {WHEAT_DERIVED_FIRST_YEAR}-{END_YEAR}): {len(derived_df)} clean county-year records"
    )

    return pd.concat([direct_df, derived_df], ignore_index=True)


def main():
    _validate_api_key()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for commodity_name, short_desc in COMMODITIES.items():
        logger.info(f"Fetching {commodity_name}...")
        raw_records = fetch_commodity_direct(commodity_name, short_desc, START_YEAR, END_YEAR)
        clean_df = clean_direct_records(raw_records, commodity_name)
        logger.info(f"{commodity_name}: {len(clean_df)} clean county-year records")
        frames.append(clean_df)

    logger.info("Fetching WHEAT (direct through 2007, derived class-weighted from 2008)...")
    frames.append(fetch_wheat_full_series())

    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.sort_values(["commodity", "county_fips", "year"]).reset_index(drop=True)

    full_df.to_parquet(OUTPUT_PATH, index=False)
    full_df.to_csv(OUTPUT_CSV_PATH, index=False)

    logger.info(f"Wrote {len(full_df)} total records to {OUTPUT_PATH} and {OUTPUT_CSV_PATH}")
    logger.info(f"\n{full_df.groupby(['commodity', 'yield_source'])['year'].agg(['min', 'max', 'count'])}")


if __name__ == "__main__":
    main()