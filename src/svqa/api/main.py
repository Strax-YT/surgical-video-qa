"""FastAPI application.

Models are loaded lazily by the registry, not at import time, so the process
starts in well under a second and Kubernetes readiness does not wait on a
2 GB checkpoint. `SVQA_WARMUP_KEYS` opts specific models into eager loading
where cold start matters more than boot time.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from svqa import __version__
from svqa.api.deps import build_services
from svqa.api.routers import ask, bench, health, models, videos

logging.basicConfig(
    level=os.getenv("SVQA_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.services = build_services()

    warmup_keys = [k for k in os.getenv("SVQA_WARMUP_KEYS", "").split(",") if k]
    if warmup_keys:
        timings = app.state.services.registry.warmup(warmup_keys)
        logger.info("warmed up: %s", timings)

    try:
        yield
    finally:
        app.state.services.close()
        logger.info("services closed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Surgical Video QA",
        version=__version__,
        description=(
            "Multimodal QA over surgical video: YOLO detection, box-prompted "
            "SAM segmentation, Whisper transcription, a Neo4j knowledge graph, "
            "and hybrid graph + vector retrieval."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("SVQA_CORS_ORIGINS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in (health.router, models.router, videos.router, ask.router, bench.router):
        app.include_router(router)
    return app


app = create_app()
