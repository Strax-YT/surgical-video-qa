"""Benchmark endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from svqa.api.deps import Services, get_services
from svqa.api.schemas import BenchmarkRequest, BenchmarkResponse, BenchmarkRow
from svqa.bench.benchmark import run_benchmark

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


@router.post("", response_model=BenchmarkResponse)
def benchmark(
    body: BenchmarkRequest, services: Services = Depends(get_services)
) -> BenchmarkResponse:
    path = Path(body.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"video not found: {path}")

    device = services.settings.resolved_device()
    try:
        rows = run_benchmark(
            services.registry,
            path,
            detector_keys=body.detector_keys,
            segmenter_keys=body.segmenter_keys,
            device=device,
            frames=body.frames,
            warmup=body.warmup_frames,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return BenchmarkResponse(
        device=device,
        rows=[BenchmarkRow(**row.as_dict()) for row in rows],
    )
