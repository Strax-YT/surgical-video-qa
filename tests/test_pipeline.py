"""Frame sampling and the ingest pipeline against a synthetic video."""

from __future__ import annotations

from pathlib import Path

from svqa.audio.transcribe import StubTranscriber
from svqa.graph.client import InMemoryGraphStore
from svqa.pipeline.ingest import ingest_video, make_video_id
from svqa.retrieval.vector_store import InMemoryVectorStore
from svqa.vision import frames as frames_mod
from svqa.vision.detector import StubDetector
from svqa.vision.segmenter import StubSegmenter


def test_sample_interval():
    assert frames_mod.sample_interval(30.0, 2.0) == 15
    assert frames_mod.sample_interval(10.0, 10.0) == 1
    assert frames_mod.sample_interval(0.0, 2.0) == 1  # missing fps metadata


def test_probe_video(sample_video: Path):
    meta = frames_mod.probe_video(sample_video)
    assert meta.frame_count > 0
    assert meta.width == 320 and meta.height == 240
    assert meta.duration_s > 0


def test_iter_frames_respects_max(sample_video: Path):
    sampled = list(frames_mod.iter_frames(sample_video, target_fps=2.0, max_frames=5))
    assert len(sampled) == 5
    assert sampled[0].timestamp_s < sampled[-1].timestamp_s


def test_iter_frames_resizes(sample_video: Path):
    sampled = list(
        frames_mod.iter_frames(sample_video, target_fps=2.0, max_frames=1,
                               resize_width=160)
    )
    assert sampled[0].image.shape[1] == 160


def test_video_ids_are_unique():
    path = Path("/tmp/my video.mp4")
    first, second = make_video_id(path), make_video_id(path)
    assert first != second
    assert first.startswith("my_video-")


def test_stub_detector_is_deterministic_per_call_index():
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    a = StubDetector().detect(frame)
    b = StubDetector().detect(frame)
    assert [d.bbox for d in a] == [d.bbox for d in b]


def test_stub_segmenter_returns_one_mask_per_box():
    import numpy as np

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [(10.0, 10.0, 50.0, 50.0), (60.0, 60.0, 90.0, 90.0)]
    masks = StubSegmenter().segment_boxes(frame, boxes)
    assert len(masks) == 2
    assert all(mask.area_px > 0 for mask in masks)


def test_full_pipeline_populates_graph_and_index(sample_video: Path):
    graph = InMemoryGraphStore()
    vectors = InMemoryVectorStore()

    result = ingest_video(
        sample_video,
        detector=StubDetector(),
        segmenter=StubSegmenter(),
        transcriber=StubTranscriber(),
        graph=graph,
        vectors=vectors,
        sample_fps=2.0,
        max_frames=20,
    )

    assert result.frames_sampled == 20
    assert result.detections_total > 0
    assert result.events
    assert result.transcript
    # Per-stage timings are what the benchmark endpoint reports.
    for stage in ("probe", "transcribe", "vision_total", "derive_events",
                  "graph_write", "vector_index", "total"):
        assert stage in result.timings_ms

    assert graph.known_videos() == [result.video_id]
    assert vectors.search("cystic duct", video_id=result.video_id, k=2)


def test_masks_can_be_skipped(sample_video: Path):
    result = ingest_video(
        sample_video,
        detector=StubDetector(),
        segmenter=None,
        transcriber=None,
        graph=InMemoryGraphStore(),
        vectors=InMemoryVectorStore(),
        sample_fps=2.0,
        max_frames=6,
        with_masks=False,
    )
    assert result.transcript == []
    assert result.timings_ms["segment"] == 0.0
