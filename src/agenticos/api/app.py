"""FastAPI wrapper around the LangGraph app.

This is the deployment target for Docker/AWS (ECS Fargate, App Runner,
Lambda-via-adapter, etc.) — a Streamlit session isn't a good fit behind a
load balancer or for service-to-service calls, but a plain HTTP API is.

Endpoints:
  GET  /health        liveness probe — process is up, doesn't touch Ollama/DB.
  GET  /ready          readiness probe — the graph/model/DB are actually usable.
  POST /investigate    run one investigation, optionally scoped to a conversation_id.
  GET  /conversations/{conversation_id}/history   recent turns for a conversation.
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agenticos.agents.graph import build_graph
from agenticos.agents.state import new_initial_state
from agenticos.logging_config import configure_logging, get_logger
from agenticos.memory.store import ConversationStore
from agenticos.settings import get_settings

logger = get_logger("api")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    try:
        _state["app_graph"] = build_graph(settings=settings)
        _state["memory_store"] = ConversationStore(settings.resolved_memory_db_path) if settings.memory_enabled else None
        _state["settings"] = settings
        _state["ready"] = True
        logger.info("startup complete")
    except Exception:
        logger.exception("startup failed — graph could not be built")
        _state["ready"] = False
    yield
    _state.clear()


app = FastAPI(title="AgenticOS", version="2.0.0", lifespan=lifespan)

_settings_for_cors = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings_for_cors.api_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex)
    start = time.time()
    response = await call_next(request)
    elapsed = round(time.time() - start, 3)
    response.headers["x-request-id"] = request_id
    logger.info(
        "request handled",
        extra={"extra_fields": {"request_id": request_id, "path": request.url.path, "status": response.status_code, "elapsed_s": elapsed}},
    )
    return response


class InvestigateRequest(BaseModel):
    request: str = Field(..., min_length=1, description="The investigation request text.")
    conversation_id: str | None = Field(default=None, description="Reuse memory from a prior conversation.")


class InvestigateResponse(BaseModel):
    final_answer: str
    conversation_id: str | None
    assignments: list[dict]
    agent_errors: dict[str, str]
    timings: dict[str, float]
    token_usage: dict[str, int]
    logs: list[str]


@app.get("/health")
def health() -> dict:
    """Liveness probe: the process is up. Deliberately doesn't touch Ollama
    or the database, so a slow backend never fails container liveness checks
    (which would trigger a restart loop instead of just a 503 on /ready)."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness probe: the graph was built successfully at startup. Use this
    for load-balancer target-group health checks, not /health."""
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ready"}


@app.post("/investigate", response_model=InvestigateResponse)
def investigate(payload: InvestigateRequest) -> InvestigateResponse:
    if not _state.get("ready"):
        raise HTTPException(status_code=503, detail="service not ready")

    settings = _state["settings"]
    memory_store: ConversationStore | None = _state.get("memory_store")

    history = []
    if memory_store and settings.memory_enabled and payload.conversation_id:
        history = memory_store.get_recent_turns(payload.conversation_id, limit=settings.memory_max_turns)

    try:
        result = _state["app_graph"].invoke(
            new_initial_state(payload.request, conversation_id=payload.conversation_id or "", history=history)
        )
    except Exception as exc:
        logger.exception("unhandled error running investigation")
        raise HTTPException(status_code=500, detail=f"investigation failed: {exc}") from exc

    return InvestigateResponse(
        final_answer=result.get("final_answer", ""),
        conversation_id=payload.conversation_id,
        assignments=result.get("assignments", []),
        agent_errors=result.get("agent_errors", {}),
        timings=result.get("timings", {}),
        token_usage=result.get("token_usage", {}),
        logs=result.get("logs", []),
    )


@app.get("/conversations/{conversation_id}/history")
def conversation_history(conversation_id: str, limit: int = 10) -> dict:
    memory_store: ConversationStore | None = _state.get("memory_store")
    if not memory_store:
        raise HTTPException(status_code=404, detail="conversation memory is disabled")
    return {"conversation_id": conversation_id, "turns": memory_store.get_recent_turns(conversation_id, limit=limit)}
