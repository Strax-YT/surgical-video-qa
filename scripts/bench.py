#!/usr/bin/env python3
"""Benchmark detector and segmenter variants and print a markdown table.

    PYTHONPATH=src python scripts/bench.py video.mp4 --frames 64
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from svqa.bench.benchmark import format_table, run_benchmark
from svqa.config import get_settings
from svqa.vision.loaders import build_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", help="video to benchmark against")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument(
        "--segmenters",
        nargs="*",
        default=["segmenter:mobile_sam", "segmenter:fast_sam"],
    )
    parser.add_argument("--json", metavar="PATH", help="also write raw results")
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)s | %(message)s")
    settings = get_settings()
    registry = build_registry(settings)
    device = settings.resolved_device()

    video = args.video
    if not video:
        import tempfile

        from scripts.demo import make_sample_video  # type: ignore[import-not-found]

        temp = tempfile.mkdtemp()
        video = str(make_sample_video(Path(temp) / "bench.mp4"))
        print(f"no video given; generated {video}")

    rows = run_benchmark(
        registry,
        video,
        detector_keys=["detector"],
        segmenter_keys=args.segmenters,
        device=device,
        frames=args.frames,
        warmup=args.warmup,
    )

    print(f"\ndevice: {device}   backend: {settings.backend}\n")
    print(format_table(rows))

    if args.json:
        Path(args.json).write_text(
            json.dumps([row.as_dict() for row in rows], indent=2)
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
