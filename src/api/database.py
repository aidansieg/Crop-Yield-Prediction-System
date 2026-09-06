"""
Shared database engine for the FastAPI app.

Reuses the same DATABASE_URL pattern as src/load_database.py so both
scripts point at the same Postgres instance without duplicating config.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/crop_yield")

# pool_pre_ping checks a connection is still alive before handing it out —
# matters once this is deployed (Step 9) since cloud Postgres providers
# often close idle connections, which would otherwise surface as a
# confusing mid-request error instead of a clean reconnect.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)