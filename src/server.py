"""
Step 9: single-service entry point for deployment.

Locally, the FastAPI backend and Dash dashboard run as two separate
processes (see src/api/main.py and src/dashboard/app.py directly) —
convenient for fast iteration, since each can be restarted independently.

For deployment, this file MERGES them into one ASGI app: Dash's
underlying Flask (WSGI) server is mounted inside the FastAPI (ASGI) app
via a2wsgi's WSGI-to-ASGI adapter. One process, one Railway service,
one Postgres connection pool, no cross-service networking or CORS to
configure — the right tradeoff for a demo deployment that doesn't need
independent scaling of the two halves.

The dashboard is served at /dashboard/; the API and its docs remain at
root (/commodities, /docs, etc.) exactly as they are locally. Visiting
the app's root URL redirects straight to /dashboard/ for convenience.

Run: uvicorn src.server:app --host 0.0.0.0 --port $PORT
"""

import os

from a2wsgi import WSGIMiddleware
from fastapi.responses import RedirectResponse

# Setting this BEFORE importing the dashboard module matters: Dash reads
# it at construction time (see src/dashboard/app.py) to tell the browser
# to request assets/callbacks at "/dashboard/..." — its own internal
# route registration deliberately stays unprefixed (see the comment next
# to requests_pathname_prefix in app.py for why).
os.environ.setdefault("DASHBOARD_REQUESTS_PATHNAME_PREFIX", "/dashboard/")

from .api.main import app  # noqa: E402 — the FastAPI app we extend below
from .dashboard.app import app as dash_app  # noqa: E402

app.mount("/dashboard", WSGIMiddleware(dash_app.server))


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/dashboard/")