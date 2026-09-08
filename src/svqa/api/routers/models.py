"""Model registry endpoints — inspect, warm, and evict models at runtime."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from svqa.api.deps import Services, get_services
from svqa.api.schemas import LoadModelRequest, LoadModelResponse, ModelStatsResponse
from svqa.vision.registry import ModelNotRegisteredError

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelStatsResponse)
def list_models(services: Services = Depends(get_services)) -> ModelStatsResponse:
    return ModelStatsResponse(**services.registry.stats())


@router.post("/load", response_model=LoadModelResponse)
def load_model(
    body: LoadModelRequest, services: Services = Depends(get_services)
) -> LoadModelResponse:
    """Force a model resident. Useful before a latency-sensitive batch so the
    first request does not pay cold start."""
    registry = services.registry
    was_resident = registry.is_resident(body.key)
    try:
        handle = registry.handle(body.key)
    except ModelNotRegisteredError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        # Missing weights is a configuration problem, not a bad request.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return LoadModelResponse(
        key=body.key,
        loaded=True,
        load_time_ms=round(handle.load_time_ms, 2),
        was_resident=was_resident,
        resident=registry.resident_keys(),
    )


@router.delete("/{key:path}")
def unload_model(key: str, services: Services = Depends(get_services)) -> dict:
    unloaded = services.registry.unload(key)
    return {
        "key": key,
        "unloaded": unloaded,
        "resident": services.registry.resident_keys(),
    }
