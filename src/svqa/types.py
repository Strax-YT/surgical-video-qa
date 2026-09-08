"""Domain types shared across the pipeline.

These are deliberately plain dataclasses rather than Neo4j or Chroma objects so
that the interesting logic (event derivation, transcript alignment, query
routing) stays testable without a database or a GPU in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coords


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected instrument in one frame."""

    label: str
    confidence: float
    bbox: BBox
    mask_area_px: int | None = None  # filled in by the segmenter, not the detector
    mask_rle: str | None = None      # run-length encoded mask, kept out of the graph

    @property
    def bbox_area_px(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True, slots=True)
class FrameDetections:
    """All detections for a single sampled frame."""

    frame_index: int
    timestamp_s: float
    detections: tuple[Detection, ...] = ()

    def labels(self) -> set[str]:
        return {d.label for d in self.detections}


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One timestamped chunk of narration or OR audio."""

    start_s: float
    end_s: float
    text: str
    speaker: str | None = None

    def overlaps(self, start_s: float, end_s: float) -> bool:
        return self.start_s < end_s and start_s < self.end_s


@dataclass(frozen=True, slots=True)
class InstrumentEvent:
    """A contiguous interval during which one instrument was present.

    Derived from per-frame detections. This is the unit the knowledge graph
    reasons over — a graph with one node per frame per detection is unusable,
    a graph of intervals answers "when was the grasper in use" in one hop.
    """

    label: str
    start_s: float
    end_s: float
    frame_count: int
    mean_confidence: float
    peak_confidence: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class Phase:
    """A labelled procedure stage covering a time range."""

    name: str
    start_s: float
    end_s: float
    index: int = 0


class RetrievalStrategy(StrEnum):
    """Which retrieval path a question should take."""

    GRAPH = "graph"    # structured / temporal / counting -> Cypher
    VECTOR = "vector"  # open-ended semantic -> embedding search over transcript
    HYBRID = "hybrid"  # needs both, results merged before synthesis


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    strategy: RetrievalStrategy
    reason: str
    confidence: float = 0.0
    matched_signals: tuple[str, ...] = ()


@dataclass(slots=True)
class Evidence:
    """A single retrieved item, carried through to the citation in the answer."""

    source: str  # "graph" | "transcript"
    content: str
    start_s: float | None = None
    end_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def cite(self) -> str:
        if self.start_s is None:
            return f"[{self.source}]"
        return f"[{self.source} @ {_fmt_ts(self.start_s)}]"


@dataclass(slots=True)
class Answer:
    question: str
    text: str
    strategy: RetrievalStrategy
    evidence: list[Evidence] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass(slots=True)
class VideoIngestResult:
    video_id: str
    frames_sampled: int
    detections_total: int
    events: list[InstrumentEvent] = field(default_factory=list)
    transcript: list[TranscriptSegment] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)
    duration_s: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)


def _fmt_ts(seconds: float) -> str:
    """Seconds -> mm:ss, for citations the user can scrub to."""
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"
