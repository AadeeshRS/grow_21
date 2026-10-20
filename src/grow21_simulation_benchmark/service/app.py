"""HTTP transport for the Grow21 learner service.

Run with:
    python -m uvicorn grow21_simulation_benchmark.service.app:create_app --factory --host 0.0.0.0 --port 8000

Handlers only validate, delegate to LearnerRuntime, and serialise; no model logic lives here.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from .runtime import LearnerRuntime, RuntimeRejection
from .schemas import NextTaskRequest, TaskDirective, TelemetryAck, TelemetryEvent
from .tts import MAX_CHARS, NeuralVoice

DEFAULT_CORS_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def create_app(runtime: Optional[LearnerRuntime] = None, voice: Optional[NeuralVoice] = None) -> FastAPI:
    runtime = runtime or LearnerRuntime.from_environment()
    voice = voice or NeuralVoice()
    app = FastAPI(title="Grow21 learner service", version="1")
    # Native React Native requests are not subject to CORS; this only matters for the Expo web host.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=os.environ.get("GROW21_CORS_ORIGIN_REGEX", DEFAULT_CORS_ORIGIN_REGEX),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(RuntimeRejection)
    async def _rejection(_: Request, exc: RuntimeRejection) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": "Request failed schema validation.", "details": details}},
        )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "schemaVersion": 1, "estimator": "robust_adaptive_filter"}

    @app.get("/api/v1/config")
    def config() -> Dict[str, Any]:
        return runtime.describe()

    @app.post("/api/v1/task/next", response_model=TaskDirective, response_model_exclude_none=True, response_model_by_alias=True)
    def next_task(request: NextTaskRequest) -> Dict[str, Any]:
        return runtime.next_task(request.childId, request.sessionId, request.mode)

    @app.post("/api/v1/telemetry/submit", response_model=TelemetryAck)
    def submit_telemetry(event: TelemetryEvent) -> Dict[str, Any]:
        return runtime.submit_telemetry(event.model_dump(exclude_none=True))

    @app.get("/api/v1/tts")
    def tts(text: str = Query(min_length=1, max_length=MAX_CHARS)) -> Response:
        if not voice.available:
            raise RuntimeRejection(404, "tts_unavailable", "No neural voice is installed; use the device voice.")
        return Response(content=voice.synthesize(text), media_type="audio/wav", headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/v1/learners/{child_id}/state")
    def learner_state(child_id: str) -> Dict[str, Any]:
        return runtime.learner_state(child_id)

    return app
