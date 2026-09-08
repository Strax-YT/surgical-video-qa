"""Video ingest."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from svqa.api.deps import Services, get_services
from svqa.api.schemas import IngestRequest, IngestResponse
from svqa.pipeline.ingest import ingest_summary, ingest_video
from svqa.vision.loaders import DETECTOR_KEY, segmenter_key
from svqa.vision.registry import ModelNotRegisteredError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    body: IngestRequest, services: Services = Depends(get_services)
) -> IngestResponse:
    """Run the full pipeline over one video.

    Synchronous by design at this stage: a real deployment puts this behind a
    task queue, but a synchronous endpoint that returns per-stage timings is
    far more useful while the pipeline is still being tuned.
    """
    path = Path(body.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"video not found: {path}")

    settings = services.settings
    variant = body.segmenter_variant or settings.segmenter_variant

    try:
        detector = services.registry.get(DETECTOR_KEY)
        segmenter = (
            services.registry.get(segmenter_key(variant)) if body.with_masks else None
        )
    except ModelNotRegisteredError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = ingest_video(
            path,
            detector=detector,
            segmenter=segmenter,
            transcriber=services.transcriber if body.with_transcript else None,
            graph=services.graph,
            vectors=services.vectors,
            video_id=body.video_id,
            sample_fps=body.sample_fps,
            scene_change_threshold=body.scene_change_threshold,
            max_frames=body.max_frames,
            gap_tolerance_s=settings.event_gap_tolerance_s,
            min_duration_s=settings.event_min_duration_s,
            with_masks=body.with_masks,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    services.durations[result.video_id] = result.duration_s
    return IngestResponse(**ingest_summary(result))


@router.get("/{video_id}/overview")
def overview(video_id: str, services: Services = Depends(get_services)) -> dict:
    rows = services.graph.run_template("video_overview", {"video_id": video_id})
    if not rows:
        raise HTTPException(status_code=404, detail=f"unknown video: {video_id}")
    return rows[0]


@router.get("/{video_id}/instruments")
def instruments(video_id: str, services: Services = Depends(get_services)) -> dict:
    return {
        "video_id": video_id,
        "instruments": services.graph.run_template(
            "instruments_in_video", {"video_id": video_id}
        ),
    }


@router.get("/{video_id}/phases")
def phases(video_id: str, services: Services = Depends(get_services)) -> dict:
    return {
        "video_id": video_id,
        "phases": services.graph.run_template(
            "phase_summary", {"video_id": video_id}
        ),
    }
