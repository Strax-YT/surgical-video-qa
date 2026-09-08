"""Event derivation: gap bridging, flicker rejection, phase inference."""

from __future__ import annotations

from svqa.graph.events import (
    align_transcript,
    derive_events,
    infer_phases,
    summarise_events,
)
from svqa.types import Detection, FrameDetections, TranscriptSegment


def frame(index: int, timestamp: float, *labels: str, conf: float = 0.9):
    return FrameDetections(
        frame_index=index,
        timestamp_s=timestamp,
        detections=tuple(
            Detection(label=label, confidence=conf, bbox=(0, 0, 10, 10))
            for label in labels
        ),
    )


def test_contiguous_detections_become_one_event():
    frames = [frame(i, i * 0.5, "grasper") for i in range(6)]
    events = derive_events(frames, gap_tolerance_s=1.0, min_duration_s=0.5)

    assert len(events) == 1
    event = events[0]
    assert event.label == "grasper"
    assert event.start_s == 0.0
    assert event.frame_count == 6
    # Ends one sampling interval past the last observation.
    assert event.end_s == 3.0


def test_short_gap_is_bridged():
    """A single dropped frame must not split one use into two events."""
    timestamps = [0.0, 0.5, 1.0, 2.0, 2.5]  # 1.0 -> 2.0 is a dropped frame
    frames = [frame(i, ts, "hook") for i, ts in enumerate(timestamps)]
    events = derive_events(frames, gap_tolerance_s=1.0, min_duration_s=0.5)

    assert len(events) == 1
    assert events[0].frame_count == 5


def test_long_gap_splits_events():
    timestamps = [0.0, 0.5, 1.0, 20.0, 20.5]
    frames = [frame(i, ts, "hook") for i, ts in enumerate(timestamps)]
    events = derive_events(frames, gap_tolerance_s=1.0, min_duration_s=0.5)

    assert len(events) == 2
    assert events[0].start_s == 0.0
    assert events[1].start_s == 20.0


def test_single_frame_flicker_is_dropped():
    frames = [frame(0, 0.0, "scissors")]
    events = derive_events(
        frames, min_duration_s=2.0, assumed_frame_duration_s=0.5
    )
    assert events == []


def test_min_confidence_filters_detections():
    frames = [frame(i, i * 0.5, "bipolar", conf=0.2) for i in range(6)]
    assert derive_events(frames, min_confidence=0.5) == []
    assert len(derive_events(frames, min_confidence=0.1)) == 1


def test_two_instruments_yield_independent_events():
    frames = [frame(i, i * 0.5, "grasper", "hook") for i in range(4)]
    events = derive_events(frames)
    assert {e.label for e in events} == {"grasper", "hook"}


def test_confidence_aggregation():
    frames = [
        frame(0, 0.0, "clipper", conf=0.5),
        frame(1, 0.5, "clipper", conf=0.9),
    ]
    event = derive_events(frames)[0]
    assert event.peak_confidence == 0.9
    assert abs(event.mean_confidence - 0.7) < 1e-9


def test_empty_input():
    assert derive_events([]) == []


def test_events_are_sorted_by_start():
    frames = [
        frame(0, 5.0, "hook"), frame(1, 5.5, "hook"),
        frame(2, 0.0, "grasper"), frame(3, 0.5, "grasper"),
    ]
    events = derive_events(frames)
    assert [e.label for e in events] == ["grasper", "hook"]


def test_phase_inference_prefers_most_specific_rule():
    frames = (
        [frame(i, i * 0.5, "grasper") for i in range(4)]
        + [frame(i + 4, 2.0 + i * 0.5, "grasper", "hook") for i in range(4)]
    )
    events = derive_events(frames)
    rules = (
        ("preparation", frozenset({"grasper"})),
        ("dissection", frozenset({"grasper", "hook"})),
    )
    phases = infer_phases(events, rules)
    names = [p.name for p in phases]

    assert "dissection" in names
    assert names.index("preparation") < names.index("dissection")


def test_transcript_alignment_links_only_on_overlap():
    frames = [frame(i, i * 0.5, "clipper") for i in range(6)]  # 0.0 - 3.0
    events = derive_events(frames)
    segments = [
        TranscriptSegment(0.0, 2.0, "clipping the cystic duct"),
        TranscriptSegment(50.0, 52.0, "irrigating now"),
    ]
    links = align_transcript(events, segments)

    assert len(links) == 1
    event_index, segment_index, overlap = links[0]
    assert (event_index, segment_index) == (0, 0)
    assert overlap == 2.0


def test_summarise_events_totals():
    frames = (
        [frame(i, i * 0.5, "grasper") for i in range(4)]
        + [frame(i + 10, 30.0 + i * 0.5, "grasper") for i in range(4)]
    )
    summary = summarise_events(derive_events(frames))
    assert summary["grasper"]["uses"] == 2
    assert summary["grasper"]["first_seen_s"] == 0.0
