"""Liveness and readiness."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from svqa import __version__
from svqa.api.deps import Services, get_services
from svqa.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def healthz(services: Services = Depends(get_services)) -> HealthResponse:
    """Liveness. Deliberately does not touch the graph or load a model —
    Kubernetes restarting the pod because Neo4j blipped is not what we want."""
    return HealthResponse(
        status="ok",
        version=__version__,
        backend=services.settings.backend,
        device=services.settings.resolved_device(),
        graph=type(services.graph).__name__,
        vector_store=type(services.vectors).__name__,
        llm=services.llm.name,
    )


@router.get("/readyz", response_model=HealthResponse)
def readyz(services: Services = Depends(get_services)) -> HealthResponse:
    """Readiness. Probes the graph, so a pod with a dead database stops
    receiving traffic without being killed."""
    status = "ok"
    try:
        services.graph.run_template("video_overview", {"video_id": "__probe__"})
    except Exception:  # noqa: BLE001
        status = "degraded"
    return HealthResponse(
        status=status,
        version=__version__,
        backend=services.settings.backend,
        device=services.settings.resolved_device(),
        graph=type(services.graph).__name__,
        vector_store=type(services.vectors).__name__,
        llm=services.llm.name,
    )
