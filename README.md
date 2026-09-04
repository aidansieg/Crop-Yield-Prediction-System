# Crop Yield Predictor

A county-level crop yield prediction system that ingests data from multiple public agricultural and climate sources, engineers 40+ features, trains an ensemble of ML models, and surfaces anomalies through an interactive dashboard.

**Live demo:** `https://your-app.railway.app`

![Dashboard screenshot](docs/dashboard_preview.png)

---

## What it does

Farmers, agronomists, and agricultural analysts can:
- View predicted corn, soybean, or wheat yields for any US county
- Compare predictions against historical actuals going back to 1990
- Identify counties at risk via an anomaly detection layer that flags unexpected yield deviations before harvest
- Filter by crop type, year, and anomaly status on an interactive choropleth map

---

## Architecture

```
USDA NASS API          NOAA CDO API
     │                      │
     └──────────┬───────────┘
                │
         Data Ingestion
         (src/ingestion/)
                │
         PostgreSQL
                │
       Feature Engineering
       (src/features/engineer.py)
       - Growing Degree Days
       - Drought Index
       - Yield lag features
       - Rolling averages
       - Linear trend slopes
                │
          ┌─────┴──────┐
          │            │
       LightGBM     Prophet
       (tabular)  (time series)
          │            │
          └─────┬──────┘
                │
            Ensemble
           (65% / 35%)
                │
        Isolation Forest
        (anomaly scoring)
                │
         FastAPI + Dash
         (dashboard + API)
```

---

## Data Sources

| Source | Data | Records |
|---|---|---|
| USDA NASS Quick Stats | County yield (bu/acre), acres planted/harvested | ~800K rows |
| NOAA Climate Data Online | Monthly temp, precipitation by station | ~2M rows |

Both APIs are free with a free account. No rate limits for reasonable usage.

---

## Feature Engineering

The feature store computes 15+ engineered features per county/year/crop:

**Climate features**
- `avg_temp_growing` — average temperature April–September
- `total_precip_growing` — total precipitation during growing season
- `growing_degree_days` — GDD with base 50°F (corn standard)
- `extreme_heat_index` — heat stress proxy above 86°F threshold
- `drought_index` — precipitation deviation from 10-year rolling mean

**Yield trend features**
- `yield_lag1`, `yield_lag2` — previous 1 and 2 year yields
- `yield_rolling_5yr` — 5-year rolling average (captures technology adoption)
- `yield_trend` — linear slope over past 10 years

---

## Models

**LightGBM** (65% ensemble weight)
Trained on all tabular features with time-based splits — test set is always the 3 most recent years. Handles missing values natively. Best at capturing climate sensitivity and cross-county patterns.

**Prophet** (35% ensemble weight)
One model per county/crop combination. Captures long-term yield trends and momentum that LightGBM misses. Falls back to LightGBM-only for counties with fewer than 10 years of data.

**Isolation Forest** (anomaly detection)
Scores every county/year prediction for anomalousness based on deviation from historical averages, model disagreement between LightGBM and Prophet, and climate outlier signals. Counties in the top 5% anomaly score get flagged for review with a human-readable reason.

---

## Local Setup

**Prerequisites:** Python 3.11+, PostgreSQL

```bash
# Clone and install
git clone https://github.com/yourusername/crop-yield-predictor.git
cd crop-yield-predictor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add USDA_API_KEY, NOAA_API_KEY, DATABASE_URL

# Initialize database
python -c "from src.db import init_db; init_db()"

# Run the full pipeline
python scripts/run_pipeline.py --step all

# Start the dashboard
python src/dashboard/app.py
# Open http://localhost:8050
```

---

## Project Structure

```
src/
├── db.py                     # Database schema and connection
├── ingestion/
│   ├── usda.py               # USDA NASS yield data ingestion
│   └── noaa.py               # NOAA climate data ingestion
├── features/
│   └── engineer.py           # Feature engineering pipeline
├── models/
│   ├── lgbm_model.py         # LightGBM trainer and predictor
│   ├── prophet_model.py      # Per-county Prophet time series models
│   └── ensemble.py           # Ensemble + Isolation Forest anomaly detection
├── api/
│   └── main.py               # FastAPI backend
└── dashboard/
    └── app.py                # Plotly Dash interactive dashboard

scripts/
└── run_pipeline.py           # Master pipeline runner

data/
├── raw/                      # Downloaded from USDA/NOAA (gitignored)
└── processed/                # Feature matrix and predictions (gitignored)

models/                       # Trained model files (gitignored)
```

---

## Design Decisions

**Why time-based train/test splits?**
Standard random splits leak future data into training when you have temporal structure. A model trained on 2020 data that's tested on 2018 data isn't predicting anything — it's memorizing. Every evaluation in this project uses the 3 most recent years as a held-out test set.

**Why ensemble LightGBM + Prophet?**
LightGBM is great at learning from cross-county climate signals but doesn't naturally capture long-term trends. Prophet is great at trend extrapolation but ignores cross-sectional features. Together they cover each other's blind spots. The 65/35 weighting reflects LightGBM's consistently lower MAE on the test set.

**Why Isolation Forest for anomalies?**
The anomaly feature space is multivariate (model disagreement, climate deviation, yield deviation from average) and unsupervised — we don't have labeled "anomaly" examples. Isolation Forest is well-suited to exactly this setup and is fast enough to score all 3,000+ counties in seconds.

**Why PostgreSQL instead of just CSVs?**
The dashboard's API queries benefit from indexed lookups and aggregations. A county history query that would scan a 800K-row CSV in seconds runs in milliseconds against a properly indexed table.

**Why Data Folder is not tracked?**
Data/ is not tracked — regenerate it by running the scripts in src/ in order (Steps 1→6). GitHub hard-caps files at 100MB, and climate_daily.parquet (363MB) is way over that.

---

## API Endpoints

```
GET /predictions?year=2023&commodity=CORN   County-level predictions
GET /predictions/{state}/{county}           Single county history
GET /anomalies?year=2023&commodity=CORN     Anomalous counties only
GET /summary?year=2023&commodity=CORN       National summary stats
GET /health                                 Health check
```
