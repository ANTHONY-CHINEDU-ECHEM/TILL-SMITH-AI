"""REST API.

Start with: python tillsmith.py serve
Interactive documentation is served at /docs once running.

Set TILLSMITH_API_KEY to require an Authorization header of the form
"Bearer <key>" on every endpoint except health.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from . import __version__
from .engine import TillsmithEngine
from .utils import sub

STARTED = time.time()
FilterValue = Union[str, List[str]]


@asynccontextmanager
async def lifespan(app):
    TillsmithEngine.shared()
    yield


app = FastAPI(
    title="Tillsmith AI",
    version=__version__,
    description="Evidence grounded diagnosis and fix ranking for ecommerce store performance incidents, "
                "drawn from a base of historical incidents with verified citations.",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("TILLSMITH_CORS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def require_key(authorization: Optional[str] = Header(default=None)):
    expected = os.environ.get("TILLSMITH_API_KEY")
    if not expected:
        return
    if authorization != "Bearer " + expected:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000,
                          examples=["Checkout conversion collapsed straight after our last release. What should we do?"])
    filters: Optional[Dict[str, FilterValue]] = Field(
        default=None, description="Hard filters on incident metadata, for example {\"platform\": \"Shopify\"}.")
    k: int = Field(default=8, ge=1, le=25, description="Number of precedent incidents in the context.")
    provider: Optional[str] = Field(default=None, description="extractive, anthropic or ollama.")
    strict: bool = Field(default=True, description="Replace model answers that fail verification.")
    include_evidence_pack: bool = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    filters: Optional[Dict[str, FilterValue]] = None
    k: int = Field(default=10, ge=1, le=100)
    mode: str = Field(default="hybrid", pattern="^(hybrid|fusion|bm25|dense)$")


class RecommendRequest(BaseModel):
    incident: str = Field(..., examples=["Checkout failure"])
    vertical: Optional[str] = None
    platform: Optional[str] = None
    business_model: Optional[str] = None


class AssessRequest(BaseModel):
    vertical: str = "Fashion and Apparel"
    business_model: str = "Direct to Consumer"
    platform: str = "Shopify Plus"
    region: str = "UK and Ireland"
    store_size_band: str = "Growth"
    fulfilment_model: str = "Third Party Logistics"
    monitoring_maturity: str = "Standard"
    annual_revenue_gbp: float = Field(default=3_000_000, gt=0)
    average_order_value_gbp: float = Field(default=70, gt=0)
    baseline_conversion_rate_pct: float = Field(default=2.2, gt=0, le=100)
    mobile_traffic_share_pct: float = Field(default=72, ge=0, le=100)
    paid_traffic_share_pct: float = Field(default=35, ge=0, le=100)
    sku_count: float = Field(default=1500, ge=1)
    app_integration_count: float = Field(default=16, ge=0)
    payment_provider_count: float = Field(default=2, ge=1)
    release_frequency_per_month: float = Field(default=6, ge=0)


@app.get("/")
def root():
    return {"name": "Tillsmith AI", "version": __version__, "docs": "/docs"}


@app.get("/health")
def health():
    engine = TillsmithEngine.shared()
    return {"status": "ok", "incidents": engine.index.size, "uptime_seconds": round(sub(time.time(), STARTED), 1)}


@app.get("/stats", dependencies=[Depends(require_key)])
def stats():
    return TillsmithEngine.shared().stats()


@app.get("/options", dependencies=[Depends(require_key)])
def options():
    return TillsmithEngine.shared().options()


@app.post("/ask", dependencies=[Depends(require_key)])
def ask(req: AskRequest):
    result = dict(TillsmithEngine.shared().ask(req.question, filters=req.filters, k=req.k,
                                               provider=req.provider, strict=req.strict))
    if req.include_evidence_pack:
        from .answer import evidence_pack
        result["evidence_pack"] = evidence_pack(req.question, result)
    return result


@app.post("/search", dependencies=[Depends(require_key)])
def search(req: SearchRequest):
    return TillsmithEngine.shared().search(req.query, filters=req.filters, k=req.k, mode=req.mode)


@app.post("/recommend", dependencies=[Depends(require_key)])
def recommend(req: RecommendRequest):
    try:
        return TillsmithEngine.shared().recommend(req.incident, req.vertical, req.platform, req.business_model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'"))


@app.post("/assess", dependencies=[Depends(require_key)])
def assess(req: AssessRequest):
    return TillsmithEngine.shared().assess(req.model_dump())


@app.get("/incidents/{incident_id}", dependencies=[Depends(require_key)])
def incident(incident_id: str):
    found = TillsmithEngine.shared().case(incident_id.upper())
    if not found:
        raise HTTPException(status_code=404, detail="No incident with id " + incident_id)
    return found
