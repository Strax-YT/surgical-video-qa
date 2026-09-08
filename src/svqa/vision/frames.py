"""Frame extraction with OpenCV.

Surgical video is long and highly redundant — 40 minutes at 25 fps is 60,000
frames, most of them near-identical. Two reductions happen here:

1.  Fixed-rate sampling down to `sample_fps` (2 fps is enough to catch
    instrument entry and exit within half a second).
2.  Optional scene-change filtering: consecutive samples whose HSV histograms
    correlate above a threshold are dropped, which collapses long static
    stretches without losing the moments where something happens.

`grab()` is used to skip frames rather than `read()`, because decoding a frame
we're about to discard is the single biggest waste in the ingest path.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SampledFrame:
    index: int          # index in the source video, not in the sample
    timestamp_s: float
    image: np.ndarray   # BGR, as OpenCV hands it over


@dataclass(frozen=True, slots=True)
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def probe_video(path: str | Path) -> VideoMeta:
    """Read container metadata without decoding the whole file."""
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        return VideoMeta(
            path=path,
            fps=fps,
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def sample_interval(source_fps: float, target_fps: float) -> int:
    """How many source frames to advance between samples (>= 1)."""
    if source_fps <= 0 or target_fps <= 0:
        return 1
    return max(1, int(round(source_fps / target_fps)))


def histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """HSV histogram correlation in [-1, 1]; 1.0 means visually identical.

    Hue and saturation only — surgical scenes shift brightness constantly as
    the scope moves, and including the value channel makes the metric fire on
    lighting changes rather than on content changes.
    """
    hists = []
    for image in (a, b):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        hists.append(hist)
    return float(cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL))


def iter_frames(
    path: str | Path,
    *,
    target_fps: float = 2.0,
    scene_change_threshold: float = 0.0,
    max_frames: int | None = None,
    resize_width: int | None = None,
) -> Iterator[SampledFrame]:
    """Yield sampled frames.

    Args:
        target_fps: sampling rate.
        scene_change_threshold: drop a sample if its histogram correlation with
            the previously *kept* frame exceeds this. 0 disables filtering.
        max_frames: stop after this many yielded frames.
        resize_width: downscale to this width, preserving aspect ratio.
    """
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or target_fps
    stride = sample_interval(source_fps, target_fps)
    logger.info(
        "sampling %s at %.1f fps (source %.1f fps, stride %d)",
        path.name, target_fps, source_fps, stride,
    )

    kept = 0
    index = 0
    previous_kept: np.ndarray | None = None
    try:
        while True:
            ok = capture.grab()  # advance without decoding
            if not ok:
                break
            if index % stride == 0:
                ok, frame = capture.retrieve()  # decode only what we sample
                if not ok or frame is None:
                    index += 1
                    continue

                if resize_width and frame.shape[1] > resize_width:
                    scale = resize_width / frame.shape[1]
                    frame = cv2.resize(
                        frame,
                        (resize_width, int(round(frame.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )

                if scene_change_threshold > 0 and previous_kept is not None:
                    similarity = histogram_similarity(previous_kept, frame)
                    if similarity > scene_change_threshold:
                        index += 1
                        continue

                previous_kept = frame
                yield SampledFrame(
                    index=index,
                    timestamp_s=index / source_fps if source_fps else 0.0,
                    image=frame,
                )
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    break
            index += 1
    finally:
        capture.release()
    logger.info("sampled %d frames from %d source frames", kept, index)


def draw_overlay(
    frame: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    labels: list[str],
    masks: list[np.ndarray] | None = None,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Render boxes and mask overlays for the demo UI and eval spot-checks."""
    canvas = frame.copy()
    palette = [
        (0, 200, 255), (0, 255, 120), (255, 90, 0),
        (200, 0, 255), (255, 220, 0), (0, 120, 255),
    ]

    if masks:
        tint = np.zeros_like(canvas)
        for i, mask in enumerate(masks):
            colour = palette[i % len(palette)]
            tint[mask.astype(bool)] = colour
        canvas = cv2.addWeighted(tint, alpha, canvas, 1 - alpha, 0)

    for i, ((x1, y1, x2, y2), label) in enumerate(zip(boxes, labels, strict=False)):
        colour = palette[i % len(palette)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(canvas, p1, p2, colour, 2)
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            canvas,
            (p1[0], max(0, p1[1] - text_h - 6)),
            (p1[0] + text_w + 6, p1[1]),
            colour,
            -1,
        )
        cv2.putText(
            canvas, label, (p1[0] + 3, max(text_h, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA,
        )
    return canvas
