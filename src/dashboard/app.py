"""
Step 8b: Plotly Dash dashboard for the crop yield prediction system.

Talks to the FastAPI backend (src/api/main.py) over HTTP rather than
querying Postgres directly. Keeps this file a pure presentation layer
with zero database dependency — the API and dashboard can be deployed,
scaled, or replaced independently (e.g. a future mobile client could
hit the same API without touching this file at all).

IMPORTANT: start the FastAPI backend BEFORE this app — commodities are
fetched once at import time (see fetch_commodities() below), so this
will fail to start at all if the API isn't already running. That's
intentional: a loud startup failure ("connection refused") is more
useful than a dashboard that silently loads with broken dropdowns.

Run: python src/dashboard/app.py
Defaults: dashboard on http://localhost:8050, API on http://localhost:8000
(override with the API_BASE_URL environment variable)
"""

import json
import os
from pathlib import Path

import dash
import pandas as pd
import plotly.express as px
import requests
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
GEOJSON_CACHE_PATH = Path("data/raw/us_counties_geojson.json")
GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

# Diverging color scale (RdBu, centered at 0) makes sense for "error" and
# "anomaly score" — both are naturally centered around a neutral midpoint
# ("as expected"). A sequential scale (Viridis) suits raw yield levels,
# which don't have a meaningful zero-point to center around.
COLOR_METRIC_OPTIONS = {
    "Predicted Yield": ("ensemble_pred", "Viridis", None),
    "Actual Yield": ("actual", "Viridis", None),
    "Prediction Error (Actual - Predicted)": ("error", "RdBu", 0),
    "Anomaly Score": ("anomaly_score", "RdBu", 0),
}


def load_counties_geojson() -> dict:
    """Cache the US counties GeoJSON locally after first fetch — county
    boundaries are static reference data, no reason to re-download on
    every app restart. Safe to fetch at import time: this hits an
    external URL, not our own API, so there's no startup-ordering risk."""
    if GEOJSON_CACHE_PATH.exists():
        with open(GEOJSON_CACHE_PATH) as f:
            return json.load(f)

    response = requests.get(GEOJSON_URL, timeout=30)
    response.raise_for_status()
    geojson = response.json()

    GEOJSON_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GEOJSON_CACHE_PATH, "w") as f:
        json.dump(geojson, f)

    return geojson


def fetch_commodities() -> list:
    response = requests.get(f"{API_BASE_URL}/commodities", timeout=10)
    response.raise_for_status()
    return response.json()


COUNTIES_GEOJSON = load_counties_geojson()

app = Dash(
    __name__,
    # Starlette's Mount STRIPS the "/dashboard" prefix before forwarding
    # to this app, so Dash's own Flask routes must be registered WITHOUT
    # it (routes_pathname_prefix, left at its default "/") — otherwise
    # nothing matches the already-stripped incoming path. But the browser
    # still needs to be told to request assets/callbacks AT "/dashboard/"
    # (requests_pathname_prefix) so those follow-up requests correctly
    # route back through the Mount in the first place. Using
    # url_base_pathname (which sets both to the SAME prefixed value) is
    # what caused the 404: Dash's own routes then expected a prefix that
    # Starlette had already removed.
    requests_pathname_prefix=os.getenv("DASHBOARD_REQUESTS_PATHNAME_PREFIX", "/"),
)
app.title = "Crop Yield Prediction Dashboard"

# IMPORTANT: the commodity dropdown starts EMPTY here on purpose — no
# fetch_commodities() call anywhere near layout construction. An earlier
# version tried deferring that fetch by making app.layout a function,
# assuming Dash would only call it once a real page was requested. It
# doesn't: Dash's `layout` setter eagerly invokes the function once,
# immediately, to validate it — which still happens at import time,
# before the merged app (see src/server.py) is listening on any port at
# all. The fix is to never call our own API anywhere layout is built;
# only from inside an actual Dash callback (see load_commodities below),
# since callbacks by definition only ever run in response to a real
# browser request hitting an already-running server.
app.layout = html.Div(className="app-shell", children=[
    dcc.Location(id="url-location", refresh=False),

    html.Div(className="sidebar", children=[
        html.P("Crop Yield Monitor", className="brand"),
        html.P(
            "County-level yield forecasts and anomaly flags for U.S. "
            "corn, soybean, and wheat production, 1990\u20132025.",
            className="brand-sub",
        ),

        html.P("Filters", className="section-label"),

        html.Div(className="control-block", children=[
            html.Label("Commodity", className="control-label"),
            dcc.Dropdown(id="commodity-dropdown", options=[], clearable=False),
        ]),

        html.Div(className="control-block", children=[
            html.Label("Year", className="control-label"),
            dcc.Dropdown(id="year-dropdown", clearable=False),
        ]),

        html.Div(className="control-block", children=[
            html.Label("Map color", className="control-label"),
            dcc.Dropdown(
                id="color-metric-dropdown",
                options=[{"label": k, "value": k} for k in COLOR_METRIC_OPTIONS],
                value="Predicted Yield",
                clearable=False,
            ),
        ]),

        html.P(
            "Source: USDA NASS county yield surveys + NOAA GHCN-Daily "
            "climate. Ensembled from LightGBM and Prophet with an "
            "Isolation Forest anomaly layer.",
            className="sidebar-footnote",
        ),
    ]),

    html.Div(className="main", children=[
        html.Div(className="page-header", children=[
            html.H1("County-Level Crop Yield Prediction", className="page-title"),
            html.P(
                "Model-based yield forecasts and anomaly flags, refreshed "
                "annually from USDA and NOAA source data.",
                className="page-subtitle",
            ),
        ]),

        html.Details(className="about-details", children=[
            html.Summary("About This Tool"),
            html.Div(className="about-body", children=[
                html.P([
                    html.Strong("What this is: "),
                    "County-level yield forecasts for U.S. corn, soybean, and wheat "
                    "production, built from USDA NASS survey history and NOAA climate "
                    "records. Two independent models, a LightGBM regressor and a "
                    "per-county Prophet trend model, are combined into a single "
                    "ensemble prediction, with an Isolation Forest layer flagging "
                    "counties where the actual reported yield deviated unexpectedly "
                    "from what the models expected.",
                ]),
                html.P([
                    html.Strong("Who it's for: "),
                    "Anyone assessing crop yield risk at the county level; "
                    "agricultural risk analysts, crop insurers, commodity desks, and "
                    "lenders who need to spot under- or over-performing counties "
                    "quickly, not just read a single national yield estimate.",
                ]),
                html.P(html.Strong("How to use it:")),
                html.Ol([
                    html.Li("Pick a commodity and year in the left sidebar."),
                    html.Li(
                        "Switch the map color mode to see predicted yield, actual "
                        "yield, prediction error, or anomaly score for every "
                        "reporting county that year."
                    ),
                    html.Li(
                        "Click any county on the map, or select a row in the "
                        "Flagged Anomalies table, to load its full year-by-year "
                        "history in the County Trend panel."
                    ),
                    html.Li(
                        "In the trend chart, red \u00d7 markers mark years flagged "
                        "as anomalous years where actual yield diverged "
                        "meaningfully from the ensemble's forecast."
                    ),
                ]),
            ]),
        ]),

        html.Div(className="panel", children=[
            dcc.Graph(
                id="choropleth-map",
                style={"height": "560px"},
                config={"displayModeBar": True, "displaylogo": False},
            ),
        ]),

        html.Div(className="panel-row", children=[
            html.Div(className="panel", children=[
                html.Div("Flagged Anomalies", className="panel-header"),
                html.Div(className="panel-body", children=[
                    dash_table.DataTable(
                        id="anomaly-table",
                        columns=[
                            {"name": "County", "id": "county_name"},
                            {"name": "State", "id": "state_alpha"},
                            {"name": "Predicted", "id": "ensemble_pred", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Actual", "id": "actual", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Anomaly Score", "id": "anomaly_score", "type": "numeric", "format": {"specifier": ".3f"}},
                            {"name": "county_fips", "id": "county_fips", "hideable": True},
                        ],
                        hidden_columns=["county_fips"],
                        row_selectable="single",
                        sort_action="native",
                        page_size=10,
                        style_as_list_view=True,
                        style_cell={"textAlign": "left"},
                    ),
                ]),
            ]),

            html.Div(className="panel", children=[
                html.Div("County Trend", className="panel-header"),
                html.Div(className="panel-body", children=[
                    dcc.Graph(id="county-trend-chart", config={"displayModeBar": False}),
                ]),
            ]),
        ]),
    ]),

    dcc.Store(id="selected-county-fips"),
])


DENSE_FONT = {"family": "Inter, sans-serif", "size": 12, "color": "#16181c"}


def apply_dense_theme(fig, title=None):
    """
    Shared styling so every chart matches the sidebar/panel aesthetic:
    flat white background, a small left-aligned monospace title instead
    of Plotly's default large centered one, and no decorative color —
    RdBu/Viridis stay because they're functional (diverging vs.
    sequential quantities), not because they look nice.
    """
    fig.update_layout(
        template="plotly_white",
        font=DENSE_FONT,
        margin={"r": 16, "t": 34 if title else 8, "l": 8, "b": 8},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "font": {"size": 11}},
    )
    if title:
        fig.update_layout(
            title={
                "text": title.upper(),
                "x": 0, "xanchor": "left",
                "font": {"family": "JetBrains Mono, monospace", "size": 11, "color": "#52565f"},
            }
        )
    return fig


@app.callback(
    Output("commodity-dropdown", "options"),
    Output("commodity-dropdown", "value"),
    Input("url-location", "pathname"),
)
def load_commodities(_pathname):
    """
    Fires exactly once when a browser first loads the page — dcc.Location's
    pathname emits its initial value on page load, which by definition can
    only happen after the server is already up and serving requests. This
    is the safe place for the one HTTP call to our own API that a static
    layout or an eagerly-evaluated layout function cannot make safely.
    """
    commodities = fetch_commodities()
    options = [{"label": c.title(), "value": c} for c in commodities]
    return options, (commodities[0] if commodities else None)


@app.callback(
    Output("year-dropdown", "options"),
    Output("year-dropdown", "value"),
    Input("commodity-dropdown", "value"),
)
def populate_years(commodity):
    if not commodity:
        return [], None
    response = requests.get(f"{API_BASE_URL}/years", params={"commodity": commodity})
    if response.status_code != 200:
        return [], None
    years = response.json()
    options = [{"label": str(y), "value": y} for y in years]
    # Default to the most recent year — most useful starting view for
    # a prediction/anomaly dashboard (recent performance matters most).
    return options, years[-1] if years else None


@app.callback(
    Output("choropleth-map", "figure"),
    Input("year-dropdown", "value"),
    Input("commodity-dropdown", "value"),
    Input("color-metric-dropdown", "value"),
)
def update_map(year, commodity, color_metric_label):
    if year is None or commodity is None:
        return apply_dense_theme(px.choropleth())

    response = requests.get(f"{API_BASE_URL}/predictions/map", params={"year": year, "commodity": commodity})
    if response.status_code != 200:
        return apply_dense_theme(px.choropleth(), title="No data for this year/commodity")

    df = pd.DataFrame(response.json())
    if df.empty:
        return apply_dense_theme(px.choropleth(), title="No data for this year/commodity")

    df["error"] = df["actual"] - df["ensemble_pred"]
    color_col, color_scale, color_midpoint = COLOR_METRIC_OPTIONS[color_metric_label]

    fig = px.choropleth(
        df,
        geojson=COUNTIES_GEOJSON,
        locations="county_fips",
        color=color_col,
        color_continuous_scale=color_scale,
        color_continuous_midpoint=color_midpoint,
        scope="usa",
        hover_name="county_name",
        hover_data={
            "state_alpha": True,
            "ensemble_pred": ":.1f",
            "actual": ":.1f",
            "is_anomaly": True,
            "county_fips": False,
        },
    )
    apply_dense_theme(fig, title=f"{commodity.title()} \u2014 {color_metric_label} \u2014 {year}")
    fig.update_layout(margin={"r": 0, "t": 34, "l": 0, "b": 0})
    return fig


@app.callback(
    Output("anomaly-table", "data"),
    Input("year-dropdown", "value"),
    Input("commodity-dropdown", "value"),
)
def update_anomaly_table(year, commodity):
    if year is None or commodity is None:
        return []
    response = requests.get(f"{API_BASE_URL}/anomalies", params={"year": year, "commodity": commodity, "limit": 25})
    if response.status_code != 200:
        return []
    return response.json()


@app.callback(
    Output("selected-county-fips", "data"),
    Input("choropleth-map", "clickData"),
    Input("anomaly-table", "selected_rows"),
    State("anomaly-table", "data"),
)
def select_county(map_click, selected_rows, table_data):
    if ctx.triggered_id == "choropleth-map" and map_click:
        return map_click["points"][0]["location"]

    if ctx.triggered_id == "anomaly-table" and selected_rows and table_data:
        return table_data[selected_rows[0]]["county_fips"]

    return dash.no_update


@app.callback(
    Output("county-trend-chart", "figure"),
    Input("selected-county-fips", "data"),
    Input("commodity-dropdown", "value"),
)
def update_county_trend(county_fips, commodity):
    if not county_fips or not commodity:
        return apply_dense_theme(px.line(), title="Click a county on the map, or select an anomaly row")

    response = requests.get(f"{API_BASE_URL}/predictions/county/{county_fips}", params={"commodity": commodity})
    if response.status_code != 200:
        return apply_dense_theme(px.line(), title="No data for this county")

    df = pd.DataFrame(response.json())
    if df.empty:
        return apply_dense_theme(px.line(), title="No data for this county")

    fig = px.line(
        df, x="year", y=["actual", "ensemble_pred"],
        labels={"value": "Yield (bu/acre)", "variable": "", "year": ""},
        color_discrete_map={"actual": "#16181c", "ensemble_pred": "#2b6cb0"},
    )
    # Two ways to tell the lines apart, not just color: "actual" is the
    # thicker solid line (ground truth), "ensemble_pred" is thinner and
    # dashed (the model's estimate) — holds up even for colorblind users
    # or a black-and-white printout, and the color contrast (near-black
    # vs. a clear medium blue) is also much stronger than the previous
    # near-black/muted-blue-gray pairing, which read as too similar.
    LEGEND_NAMES = {"actual": "Actual", "ensemble_pred": "Predicted (ensemble)"}
    fig.for_each_trace(lambda t: t.update(
        name=LEGEND_NAMES.get(t.name, t.name),
        line={"dash": "dash", "width": 2} if t.name == "ensemble_pred" else {"width": 2.5},
    ))
    apply_dense_theme(fig)

    anomalies = df[df["is_anomaly"] == True]  # noqa: E712 — explicit comparison needed, is_anomaly may be None
    if not anomalies.empty:
        fig.add_scatter(
            x=anomalies["year"], y=anomalies["actual"],
            mode="markers",
            # Bright, saturated alert red — deliberately more vivid than
            # the muted rust used elsewhere in the UI (--accent in
            # style.css). A chart marker meant to grab attention earns
            # more saturation than static chrome; everything else in
            # the palette stays muted specifically so this pops.
            marker={"size": 11, "symbol": "x", "color": "#e63946"},
            name="Anomaly",
        )
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)