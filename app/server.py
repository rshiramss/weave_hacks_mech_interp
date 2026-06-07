"""FastAPI demo endpoint (Stage 8): POST a structured sim alert -> triage.

Ingests a structured alert (or NL message), converts it to NL once, routes it
via the ROUTER_BACKEND dispatcher, and returns the triage result plus the Weave
project so judges can open the live trace.

Run:
    uvicorn app.server:app --reload
    curl -s localhost:8000/triage -H 'content-type: application/json' \
      -d '{"event": {"summary":"847 failed SSH logins","source_ip":"203.0.113.44","host":"prod-bastion-01"}}'
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.setrecursionlimit(10_000)

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.demo_pipeline import route_alert  # noqa: E402
from src.weave_setup import WEAVE_PROJECT, init_weave  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from dotenv import load_dotenv

    load_dotenv()
    init_weave()
    yield


app = FastAPI(title="SOC Probe Router Demo", lifespan=lifespan)


class TriageRequest(BaseModel):
    event: dict[str, Any] | None = None
    message: str | None = None
    backend: str | None = None  # stub | rag | incontext | probe
    execute: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "weave_project": WEAVE_PROJECT}


@app.post("/triage")
def triage(request: TriageRequest) -> dict:
    payload = request.event if request.event is not None else (request.message or "")
    out = route_alert(payload, backend=request.backend, execute=request.execute)
    out["weave_project"] = WEAVE_PROJECT
    return out
