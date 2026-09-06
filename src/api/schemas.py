"""
Pydantic response models for the crop yield prediction API.

Most numeric fields are Optional: prophet_pred/prophet_available reflect
Step 6's structural gaps (no history, or no future data for that
county-commodity), and anomaly_score/is_anomaly may be null for rows the
Isolation Forest layer didn't score. Modeling these as optional rather
than defaulting to 0 or False keeps "we don't know" distinguishable from
a real value — consistent with every data-quality decision earlier in
this pipeline.
"""

from typing import Optional

from pydantic import BaseModel


class CountyPrediction(BaseModel):
    county_fips: str
    county_name: Optional[str]
    state_alpha: Optional[str]
    year: int
    commodity: str
    lgbm_pred: Optional[float]
    prophet_pred: Optional[float]
    ensemble_pred: Optional[float]
    actual: Optional[float]
    is_anomaly: Optional[bool]
    anomaly_score: Optional[float]
    model_disagreement: Optional[float]
    prophet_available: Optional[bool]


class CountyTrendPoint(BaseModel):
    year: int
    lgbm_pred: Optional[float]
    prophet_pred: Optional[float]
    ensemble_pred: Optional[float]
    actual: Optional[float]
    is_anomaly: Optional[bool]


class AnomalyRecord(BaseModel):
    county_fips: str
    county_name: Optional[str]
    state_alpha: Optional[str]
    year: int
    commodity: str
    ensemble_pred: Optional[float]
    actual: Optional[float]
    anomaly_score: Optional[float]
    model_disagreement: Optional[float]