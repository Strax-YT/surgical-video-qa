"""Application state and FastAPI dependencies.

Everything expensive lives on `app.state.services` and is built once in the
lifespan hook. Routers reach it through `get_services`, which keeps them free
of construction logic and makes them trivial to test with a substituted state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastapi import Request

from svqa.audio.transcribe import Transcriber, build_transcriber
from svqa.config import Settings, get_settings
from svqa.graph.client import GraphStore, build_graph_store
from svqa.graph.schema import DEFAULT_PHASE_RULES
from svqa.retrieval.engine import AskEngine, build_engine
from svqa.retrieval.llm import LLMClient, build_llm
from svqa.retrieval.vector_store import VectorStore, build_vector_store
from svqa.vision.loaders import build_registry
from svqa.vision.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Services:
    settings: Settings
    registry: ModelRegistry
    graph: GraphStore
    vectors: VectorStore
    llm: LLMClient
    engine: AskEngine
    transcriber: Transcriber
    # video_id -> duration, so relative time questions ("the last 2 minutes")
    # can be resolved without a graph round trip.
    durations: dict[str, float]

    def close(self) -> None:
        self.registry.clear()
        self.graph.close()


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    registry = build_registry(settings)
    graph = build_graph_store(settings)
    vectors = build_vector_store(settings)
    llm = build_llm(settings)
    phases = tuple(name for name, _ in DEFAULT_PHASE_RULES)
    engine = build_engine(settings, graph, vectors, llm, known_phases=phases)
    transcriber = build_transcriber(settings)
    logger.info(
        "services ready: backend=%s device=%s graph=%s vectors=%s llm=%s",
        settings.backend, settings.resolved_device(),
        type(graph).__name__, type(vectors).__name__, llm.name,
    )
    return Services(
        settings=settings,
        registry=registry,
        graph=graph,
        vectors=vectors,
        llm=llm,
        engine=engine,
        transcriber=transcriber,
        durations={},
    )


def get_services(request: Request) -> Services:
    services: Any = getattr(request.app.state, "services", None)
    if services is None:  # pragma: no cover - lifespan always sets this
        raise RuntimeError("services not initialised")
    return services
