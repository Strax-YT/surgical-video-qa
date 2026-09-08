#!/usr/bin/env python3
"""Runs the whole pipeline end to end and prints the answers.

    SVQA_BACKEND=stub PYTHONPATH=src python scripts/demo.py [video.mp4]

With no argument it generates a short synthetic video, so this works on a
clean checkout with no dataset, no weights, no database and no API key.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

from svqa.api.deps import build_services
from svqa.config import get_settings
from svqa.pipeline.ingest import ingest_summary, ingest_video
from svqa.vision.loaders import DETECTOR_KEY, segmenter_key

QUESTIONS = (
    "Which instruments were used?",
    "When was the clipper used?",
    "What did the surgeon say about the adhesion?",
    "What was said while the clipper was out?",
    "What came after the hook?",
    "Summarise the procedure.",
)


def make_sample_video(path: Path, seconds: int = 72, fps: int = 10) -> Path:
    """72 seconds by default: long enough to cover StubTranscriber's script,
    so the MENTIONED_DURING edges are populated and the hybrid path has both
    graph and transcript evidence to merge rather than narration alone."""
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (640, 480)
    )
    if not writer.isOpened():
        raise RuntimeError("no mp4v encoder available")
    rng = np.random.default_rng(0)
    for i in range(seconds * fps):
        frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        frame[:, :, 2] = 90 + (i * 3) % 120           # tissue-ish red drift
        frame = cv2.add(frame, rng.integers(0, 20, frame.shape, dtype=np.uint8))
        writer.write(frame)
    writer.release()
    return path


def main() -> None:
    logging.basicConfig(level="WARNING", format="%(levelname)s %(name)s | %(message)s")
    settings = get_settings()
    services = build_services(settings)

    temp_dir = tempfile.TemporaryDirectory()
    try:
        if len(sys.argv) > 1:
            video = Path(sys.argv[1])
        else:
            video = make_sample_video(Path(temp_dir.name) / "sample.mp4")
            print(f"generated sample video: {video}")

        print(f"\nbackend={settings.backend} device={settings.resolved_device()}")
        print(f"graph={type(services.graph).__name__} llm={services.llm.name}\n")

        result = ingest_video(
            video,
            detector=services.registry.get(DETECTOR_KEY),
            segmenter=services.registry.get(segmenter_key(settings.segmenter_variant)),
            transcriber=services.transcriber,
            graph=services.graph,
            vectors=services.vectors,
            sample_fps=settings.sample_fps,
        )
        services.durations[result.video_id] = result.duration_s
        summary = ingest_summary(result)

        print("=" * 72)
        print(f"INGESTED {summary['video_id']}  ({summary['duration_s']}s)")
        print("=" * 72)
        print(f"frames sampled     : {summary['frames_sampled']}")
        print(f"detections         : {summary['detections_total']}")
        print(f"events derived     : {summary['events']}")
        print(f"transcript segments: {summary['transcript_segments']}")
        print("\ninstrument usage:")
        for name, stats in summary["instruments"].items():
            print(
                f"  {name:<15} uses={int(stats['uses']):<3} "
                f"total={stats['total_duration_s']:>6.1f}s  "
                f"conf={stats['mean_confidence']:.2f}"
            )
        print("\nphases:")
        for phase in summary["phases"]:
            print(f"  {phase['name']:<28} {phase['start_s']:>6.1f}s - {phase['end_s']:>6.1f}s")
        print("\nstage timings (ms):")
        for stage, value in summary["timings_ms"].items():
            print(f"  {stage:<15} {value:>9.1f}")

        print("\n" + "=" * 72)
        print("ASK ANYTHING")
        print("=" * 72)
        for question in QUESTIONS:
            answer = services.engine.ask(
                question, result.video_id, video_duration_s=result.duration_s
            )
            print(f"\nQ: {question}")
            print(f"   route: {answer.strategy.value}  ({answer.latency_ms:.0f} ms)")
            print(f"   A: {answer.text[:400]}")
            for i, item in enumerate(answer.evidence[:3], start=1):
                print(f"      [{i}] {item.cite()} {item.content[:90]}")
    finally:
        services.close()
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
