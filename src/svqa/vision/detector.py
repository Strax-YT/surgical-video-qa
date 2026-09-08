"""Instrument detection.

`Detector` is a Protocol so the pipeline never imports ultralytics directly —
that keeps torch out of the CI image and lets `StubDetector` stand in for the
real thing in tests and in the no-GPU demo path.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from svqa.types import BBox, Detection

logger = logging.getLogger(__name__)

# Class names for the surgical-instrument detector. Keep this aligned with the
# data.yaml used for training (scripts/train_yolo.py writes both).
INSTRUMENT_CLASSES = (
    "grasper",
    "bipolar",
    "hook",
    "scissors",
    "clipper",
    "irrigator",
    "specimen_bag",
)


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]: ...

    @property
    def name(self) -> str: ...


class YoloDetector:
    """Ultralytics YOLO wrapper (v8 through v12 share this API).

    Batching matters here: on GPU, calling the model with 16 frames is close to
    the cost of calling it with one, so the ingest path always goes through
    `detect_batch`.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str = "cpu",
        conf: float = 0.35,
        iou: float = 0.50,
        imgsz: int = 640,
        half: bool | None = None,
    ) -> None:
        self._weights = Path(weights)
        if not self._weights.exists():
            raise FileNotFoundError(
                f"detector weights not found: {self._weights}. "
                "Train with scripts/train_yolo.py or set SVQA_BACKEND=stub."
            )
        from ultralytics import YOLO  # lazy: keeps torch out of import time

        self._model = YOLO(str(self._weights))
        self._device = device
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        # fp16 is a free ~40% latency win on GPU and unsupported on CPU.
        self._half = half if half is not None else device.startswith("cuda")
        logger.info(
            "YoloDetector ready: %s on %s (half=%s)",
            self._weights.name, device, self._half,
        )

    @property
    def name(self) -> str:
        return f"yolo:{self._weights.stem}"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        results = self._model.predict(
            frames,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            half=self._half,
            verbose=False,
        )
        return [self._parse(result) for result in results]

    def _parse(self, result: Any) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        names = result.names if isinstance(result.names, dict) else dict(
            enumerate(result.names)
        )
        detections: list[Detection] = []
        xyxy = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()
        classes = boxes.cls.tolist()
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes, strict=True):
            detections.append(
                Detection(
                    label=str(names.get(int(cls), int(cls))),
                    confidence=float(conf),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return detections

    def close(self) -> None:
        """Called by the registry on eviction."""
        self._model = None


class StubDetector:
    """Deterministic fake detector.

    Output is a pure function of frame content, so a test can assert on exact
    boxes and an ingest run is reproducible. It emits instruments in temporally
    contiguous runs rather than at random, so the event-derivation logic
    downstream gets realistic input to fold.
    """

    def __init__(self, classes: tuple[str, ...] = INSTRUMENT_CLASSES) -> None:
        self._classes = classes
        self._calls = 0

    @property
    def name(self) -> str:
        return "stub:detector"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        out = []
        for frame in frames:
            out.append(self._detect_one(frame, self._calls))
            self._calls += 1
        return out

    def _detect_one(self, frame: np.ndarray, call_index: int) -> list[Detection]:
        height, width = frame.shape[:2]
        seed = int(
            hashlib.sha256(
                f"{call_index}:{width}x{height}".encode()
            ).hexdigest()[:8],
            16,
        )
        rng = np.random.default_rng(seed)

        # A "grasper" is present for most of a procedure; the second instrument
        # changes every ~8 samples, producing intervals worth grouping.
        detections = [self._box(rng, width, height, "grasper", 0.72, 0.95)]
        secondary = self._classes[1 + (call_index // 8) % (len(self._classes) - 1)]
        if call_index % 8 < 5:
            detections.append(self._box(rng, width, height, secondary, 0.45, 0.88))
        return detections

    @staticmethod
    def _box(
        rng: np.random.Generator,
        width: int,
        height: int,
        label: str,
        conf_low: float,
        conf_high: float,
    ) -> Detection:
        box_w = rng.uniform(0.12, 0.30) * width
        box_h = rng.uniform(0.12, 0.30) * height
        x1 = rng.uniform(0, max(1.0, width - box_w))
        y1 = rng.uniform(0, max(1.0, height - box_h))
        bbox: BBox = (
            float(x1), float(y1), float(x1 + box_w), float(y1 + box_h),
        )
        return Detection(
            label=label,
            confidence=float(rng.uniform(conf_low, conf_high)),
            bbox=bbox,
        )

    def close(self) -> None:
        return None
