"""End-to-end ingest: video in, populated graph and vector index out.

Stage order and why:

  1. probe        cheap metadata read, fails fast on a bad container
  2. transcribe   before vision, because ASR is CPU-bound and vision is
                  GPU-bound; on a single-GPU box this ordering keeps the GPU
                  free while Whisper works, and the two are independent
  3. detect       batched through YOLO
  4. segment      box-prompted, only on frames that actually had detections
  5. derive       fold detections into intervals, infer phases (pure Python)
  6. persist      one graph transaction plus one vector index write

Every stage records wall time into `timings_ms`. That is not decoration — it is
what the benchmark endpoint reports and what tells you whether a slow ingest is
the decoder, the detector, or the graph write.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from svqa.audio.transcribe import Transcriber
from svqa.graph.client import GraphStore
from svqa.graph.events import derive_events, infer_phases, summarise_events
from svqa.graph.schema import DEFAULT_PHASE_RULES
from svqa.retrieval.vector_store import VectorStore
from svqa.types import Detection, FrameDetections, VideoIngestResult
from svqa.vision import frames as frames_mod
from svqa.vision.detector import Detector
from svqa.vision.segmenter import Segmenter

logger = logging.getLogger(__name__)

DETECT_BATCH_SIZE = 16


@contextmanager
def _timed(timings: dict[str, float], key: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = round((time.perf_counter() - started) * 1000.0, 2)


def make_video_id(path: Path) -> str:
    """Stable-ish id: filename stem plus a short random suffix.

    Not a content hash — hashing a 2 GB video to name it is a poor trade, and
    re-ingesting the same file is meant to be an explicit choice via video_id.
    """
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem)[:40]
    return f"{stem}-{uuid.uuid4().hex[:8]}"


def ingest_video(
    video_path: str | Path,
    *,
    detector: Detector,
    segmenter: Segmenter | None,
    transcriber: Transcriber | None,
    graph: GraphStore,
    vectors: VectorStore,
    video_id: str | None = None,
    sample_fps: float = 2.0,
    scene_change_threshold: float = 0.0,
    max_frames: int | None = None,
    resize_width: int | None = 960,
    gap_tolerance_s: float = 1.0,
    min_duration_s: float = 0.5,
    phase_rules: tuple[tuple[str, frozenset[str]], ...] = DEFAULT_PHASE_RULES,
    with_masks: bool = True,
) -> VideoIngestResult:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    video_id = video_id or make_video_id(video_path)
    timings: dict[str, float] = {}

    with _timed(timings, "probe"):
        meta = frames_mod.probe_video(video_path)
    logger.info(
        "ingesting %s (%.1fs, %.1f fps, %dx%d) as %s",
        video_path.name, meta.duration_s, meta.fps, meta.width, meta.height, video_id,
    )

    transcript = []
    if transcriber is not None:
        with _timed(timings, "transcribe"):
            transcript = transcriber.transcribe(video_path)

    # ---- vision -----------------------------------------------------------
    frame_detections: list[FrameDetections] = []
    detections_total = 0
    detect_ms = 0.0
    segment_ms = 0.0

    batch_images: list[Any] = []
    batch_stamps: list[tuple[int, float]] = []

    def flush_batch() -> None:
        nonlocal detections_total, detect_ms, segment_ms
        if not batch_images:
            return

        started = time.perf_counter()
        batched = detector.detect_batch(batch_images)
        detect_ms += (time.perf_counter() - started) * 1000.0

        for image, (index, timestamp), detections in zip(
            batch_images, batch_stamps, batched, strict=True
        ):
            enriched: tuple[Detection, ...] = tuple(detections)
            # Only pay for segmentation on frames that had a detection.
            if with_masks and segmenter is not None and detections:
                started = time.perf_counter()
                masks = segmenter.segment_boxes(
                    image, [d.bbox for d in detections]
                )
                segment_ms += (time.perf_counter() - started) * 1000.0
                enriched = tuple(
                    Detection(
                        label=d.label,
                        confidence=d.confidence,
                        bbox=d.bbox,
                        mask_area_px=mask.area_px,
                    )
                    for d, mask in zip(detections, masks, strict=False)
                )
            frame_detections.append(
                FrameDetections(
                    frame_index=index,
                    timestamp_s=timestamp,
                    detections=enriched,
                )
            )
            detections_total += len(enriched)

        batch_images.clear()
        batch_stamps.clear()

    with _timed(timings, "vision_total"):
        for sampled in frames_mod.iter_frames(
            video_path,
            target_fps=sample_fps,
            scene_change_threshold=scene_change_threshold,
            max_frames=max_frames,
            resize_width=resize_width,
        ):
            batch_images.append(sampled.image)
            batch_stamps.append((sampled.index, sampled.timestamp_s))
            if len(batch_images) >= DETECT_BATCH_SIZE:
                flush_batch()
        flush_batch()

    timings["detect"] = round(detect_ms, 2)
    timings["segment"] = round(segment_ms, 2)

    # ---- derivation (pure) ------------------------------------------------
    with _timed(timings, "derive_events"):
        events = derive_events(
            frame_detections,
            gap_tolerance_s=gap_tolerance_s,
            min_duration_s=min_duration_s,
        )
        phases = infer_phases(events, phase_rules, duration_s=meta.duration_s)

    result = VideoIngestResult(
        video_id=video_id,
        frames_sampled=len(frame_detections),
        detections_total=detections_total,
        events=events,
        transcript=transcript,
        phases=phases,
        duration_s=meta.duration_s,
        timings_ms=timings,
    )

    # ---- persistence ------------------------------------------------------
    with _timed(timings, "graph_write"):
        graph.ensure_schema()
        counts = graph.ingest(
            result,
            filename=video_path.name,
            fps=meta.fps,
            detector=detector.name,
            segmenter=getattr(segmenter, "name", "none"),
        )

    with _timed(timings, "vector_index"):
        indexed = vectors.index_segments(video_id, transcript)

    timings["total"] = round(sum(
        v for k, v in timings.items()
        if k in {"probe", "transcribe", "vision_total", "derive_events",
                 "graph_write", "vector_index"}
    ), 2)

    logger.info(
        "ingested %s: %d frames, %d detections, %d events, %d phases, "
        "%d segments indexed, graph=%s",
        video_id, result.frames_sampled, detections_total, len(events),
        len(phases), indexed, counts,
    )
    return result


def ingest_summary(result: VideoIngestResult) -> dict[str, Any]:
    """Compact response body for the API."""
    return {
        "video_id": result.video_id,
        "duration_s": round(result.duration_s, 2),
        "frames_sampled": result.frames_sampled,
        "detections_total": result.detections_total,
        "events": len(result.events),
        "phases": [
            {"name": p.name, "start_s": round(p.start_s, 2), "end_s": round(p.end_s, 2)}
            for p in result.phases
        ],
        "transcript_segments": len(result.transcript),
        "instruments": summarise_events(result.events),
        "timings_ms": result.timings_ms,
    }
