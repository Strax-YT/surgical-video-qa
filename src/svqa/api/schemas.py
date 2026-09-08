"""Request and response models for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str
    backend: str
    device: str
    graph: str
    vector_store: str
    llm: str


class ModelStatsResponse(BaseModel):
    capacity: int
    registered: list[str]
    resident: list[dict[str, Any]]
    cold_loads: int
    evictions: int
    hit_rate: float


class LoadModelRequest(BaseModel):
    key: str = Field(..., description="Registered model key, e.g. 'segmenter:mobile_sam'")


class LoadModelResponse(BaseModel):
    key: str
    loaded: bool
    load_time_ms: float
    was_resident: bool
    resident: list[str]


class IngestRequest(BaseModel):
    path: str = Field(..., description="Server-side path to the video file")
    video_id: str | None = None
    sample_fps: float = Field(default=2.0, gt=0, le=30)
    max_frames: int | None = Field(default=None, ge=1)
    scene_change_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    segmenter_variant: str | None = None
    with_masks: bool = True
    with_transcript: bool = True


class IngestResponse(BaseModel):
    video_id: str
    duration_s: float
    frames_sampled: int
    detections_total: int
    events: int
    phases: list[dict[str, Any]]
    transcript_segments: int
    instruments: dict[str, dict[str, float]]
    timings_ms: dict[str, float]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    video_id: str
    top_k: int | None = Field(default=None, ge=1, le=20)


class EvidenceOut(BaseModel):
    source: str
    content: str
    start_s: float | None = None
    end_s: float | None = None
    citation: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    question: str
    answer: str
    strategy: str
    routing_reason: str
    evidence: list[EvidenceOut]
    latency_ms: float


class BenchmarkRequest(BaseModel):
    path: str = Field(..., description="Video to benchmark against")
    detector_keys: list[str] = Field(default_factory=lambda: ["detector"])
    segmenter_keys: list[str] = Field(
        default_factory=lambda: ["segmenter:mobile_sam", "segmenter:fast_sam"]
    )
    frames: int = Field(default=32, ge=1, le=500)
    warmup_frames: int = Field(default=4, ge=0, le=50)


class BenchmarkRow(BaseModel):
    stage: str
    model: str
    device: str
    frames: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    fps: float
    load_time_ms: float


class BenchmarkResponse(BaseModel):
    device: str
    rows: list[BenchmarkRow]
