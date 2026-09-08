"""Turning per-frame detections into intervals.

This is the heart of the graph design. A naive ingest writes one node per
detection: a 40-minute procedure at 2 fps with 2 instruments per frame is
~9,600 nodes that answer nothing useful, because "when was the hook in use"
becomes a scan over every node in the video.

Folding contiguous detections into `InstrumentEvent` intervals gives roughly
20-80 nodes per video and turns that same question into a single indexed range
query. The interesting part is deciding what counts as contiguous:

  * A detector that misses one frame mid-use should not split an event in two,
    so gaps up to `gap_tolerance_s` are bridged.
  * A flicker of one frame at low confidence is a false positive, not a use,
    so events shorter than `min_duration_s` are dropped.

Both thresholds are config, and both are exercised in tests/test_events.py.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence

from svqa.types import FrameDetections, InstrumentEvent, Phase, TranscriptSegment


def derive_events(
    frames: Iterable[FrameDetections],
    *,
    gap_tolerance_s: float = 1.0,
    min_duration_s: float = 0.5,
    min_confidence: float = 0.0,
    assumed_frame_duration_s: float | None = None,
) -> list[InstrumentEvent]:
    """Fold per-frame detections into per-instrument intervals.

    Args:
        frames: sampled frames with detections, any order.
        gap_tolerance_s: bridge detector dropouts up to this long.
        min_duration_s: discard intervals shorter than this.
        min_confidence: ignore detections below this confidence.
        assumed_frame_duration_s: how long a single-frame event is treated as
            lasting. Defaults to the median sampling interval, which is the
            honest answer — a detection at 2 fps tells you the instrument was
            there for about half a second, not for an instant.

    Returns:
        Events sorted by start time, then label.
    """
    ordered = sorted(frames, key=lambda f: f.timestamp_s)
    if not ordered:
        return []

    frame_duration = (
        assumed_frame_duration_s
        if assumed_frame_duration_s is not None
        else _median_interval(ordered)
    )

    # label -> [(timestamp, confidence), ...] in time order
    tracks: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for frame in ordered:
        for detection in frame.detections:
            if detection.confidence < min_confidence:
                continue
            tracks[detection.label].append((frame.timestamp_s, detection.confidence))

    events: list[InstrumentEvent] = []
    for label, samples in tracks.items():
        for run in _split_on_gaps(samples, gap_tolerance_s):
            event = _run_to_event(label, run, frame_duration)
            if event.duration_s + 1e-9 >= min_duration_s:
                events.append(event)

    events.sort(key=lambda e: (e.start_s, e.label))
    return events


def _median_interval(ordered: Sequence[FrameDetections]) -> float:
    """Median gap between consecutive samples; falls back to 0.5 s."""
    if len(ordered) < 2:
        return 0.5
    deltas = [
        b.timestamp_s - a.timestamp_s
        for a, b in zip(ordered, ordered[1:], strict=False)
        if b.timestamp_s > a.timestamp_s
    ]
    return statistics.median(deltas) if deltas else 0.5


def _split_on_gaps(
    samples: list[tuple[float, float]], gap_tolerance_s: float
) -> list[list[tuple[float, float]]]:
    """Split a time-ordered track wherever the gap exceeds the tolerance."""
    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    previous_ts: float | None = None
    for timestamp, confidence in samples:
        if previous_ts is not None and (timestamp - previous_ts) > gap_tolerance_s:
            runs.append(current)
            current = []
        current.append((timestamp, confidence))
        previous_ts = timestamp
    if current:
        runs.append(current)
    return runs


def _run_to_event(
    label: str, run: list[tuple[float, float]], frame_duration_s: float
) -> InstrumentEvent:
    timestamps = [ts for ts, _ in run]
    confidences = [conf for _, conf in run]
    start = timestamps[0]
    # Extend past the last observation by one sampling interval: the instrument
    # was visible *through* that frame, not only at its leading edge.
    end = timestamps[-1] + frame_duration_s
    return InstrumentEvent(
        label=label,
        start_s=start,
        end_s=end,
        frame_count=len(run),
        mean_confidence=sum(confidences) / len(confidences),
        peak_confidence=max(confidences),
    )


# --------------------------------------------------------------------- phases


def infer_phases(
    events: Sequence[InstrumentEvent],
    phase_rules: Sequence[tuple[str, frozenset[str]]],
    *,
    duration_s: float | None = None,
) -> list[Phase]:
    """Assign procedure phases from which instruments are co-present.

    A rule is (phase_name, required_labels). A window belongs to the phase whose
    required labels are all present, preferring the most specific rule that
    matches. Adjacent windows with the same phase are merged.

    This is intentionally rule-based rather than a learned phase classifier —
    it needs no phase-labelled training data, and it is inspectable, which
    matters when a clinician asks why the graph says "dissection".
    """
    if not events:
        return []

    boundaries = sorted({e.start_s for e in events} | {e.end_s for e in events})
    if duration_s is not None:
        boundaries = sorted({b for b in boundaries if b <= duration_s} | {duration_s})

    windows: list[tuple[float, float, str | None]] = []
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        active = {e.label for e in events if e.start_s <= midpoint < e.end_s}
        matched = [
            name for name, required in phase_rules if required <= active
        ]
        best = max(
            matched,
            key=lambda name: len(dict(phase_rules)[name]),
            default=None,
        )
        windows.append((start, end, best))

    merged: list[Phase] = []
    for start, end, name in windows:
        if name is None:
            continue
        if merged and merged[-1].name == name and abs(merged[-1].end_s - start) < 1e-6:
            previous = merged.pop()
            merged.append(
                Phase(
                    name=name,
                    start_s=previous.start_s,
                    end_s=end,
                    index=previous.index,
                )
            )
        else:
            merged.append(
                Phase(name=name, start_s=start, end_s=end, index=len(merged))
            )
    return merged


# ---------------------------------------------------------------- transcript


def align_transcript(
    events: Sequence[InstrumentEvent],
    segments: Sequence[TranscriptSegment],
) -> list[tuple[int, int, float]]:
    """Match transcript segments to overlapping events.

    Returns (event_index, segment_index, overlap_seconds) triples, which the
    ingest writes as `MENTIONED_DURING` edges. This is what lets a single
    question span both modalities: "what did the surgeon say while the clipper
    was out" is one graph traversal instead of two searches and a manual join.
    """
    links: list[tuple[int, int, float]] = []
    for event_index, event in enumerate(events):
        for segment_index, segment in enumerate(segments):
            overlap = min(event.end_s, segment.end_s) - max(
                event.start_s, segment.start_s
            )
            if overlap > 0:
                links.append((event_index, segment_index, overlap))
    return links


def summarise_events(events: Sequence[InstrumentEvent]) -> dict[str, dict[str, float]]:
    """Per-instrument totals, used for the ingest response and the demo UI."""
    summary: dict[str, dict[str, float]] = {}
    for event in events:
        entry = summary.setdefault(
            event.label,
            {"uses": 0, "total_duration_s": 0.0, "first_seen_s": event.start_s,
             "last_seen_s": event.end_s, "mean_confidence": 0.0},
        )
        entry["uses"] += 1
        entry["total_duration_s"] += event.duration_s
        entry["first_seen_s"] = min(entry["first_seen_s"], event.start_s)
        entry["last_seen_s"] = max(entry["last_seen_s"], event.end_s)
        # running mean weighted by use count
        entry["mean_confidence"] += (
            event.mean_confidence - entry["mean_confidence"]
        ) / entry["uses"]
    for entry in summary.values():
        entry["total_duration_s"] = round(entry["total_duration_s"], 2)
        entry["mean_confidence"] = round(entry["mean_confidence"], 3)
    return summary
