"""Builds the model registry from settings.

Everything is registered as a closure, so nothing is loaded until a request
actually asks for that key. Swapping SVQA_BACKEND=stub replaces the loaders
without touching a line of pipeline code.
"""

from __future__ import annotations

import logging

from svqa.config import Settings, get_settings
from svqa.vision.detector import StubDetector, YoloDetector
from svqa.vision.registry import ModelRegistry
from svqa.vision.segmenter import (
    SAM_CHECKPOINTS,
    FastSamSegmenter,
    SamSegmenter,
    StubSegmenter,
)

logger = logging.getLogger(__name__)

DETECTOR_KEY = "detector"
SEGMENTER_KEY_PREFIX = "segmenter"

# model_type strings the segment_anything registry expects per variant
SAM_MODEL_TYPES = {"sam": "vit_h", "sam2": "vit_h", "mobile_sam": "vit_t"}


def segmenter_key(variant: str) -> str:
    return f"{SEGMENTER_KEY_PREFIX}:{variant}"


def build_registry(settings: Settings | None = None) -> ModelRegistry:
    settings = settings or get_settings()
    registry = ModelRegistry(capacity=settings.model_cache_size)
    device = settings.resolved_device()

    if settings.backend == "stub":
        registry.register(
            DETECTOR_KEY,
            StubDetector,
            metadata={"backend": "stub", "device": "cpu"},
        )
        for variant in SAM_CHECKPOINTS:
            registry.register(
                segmenter_key(variant),
                StubSegmenter,
                metadata={"backend": "stub", "variant": variant, "device": "cpu"},
            )
        logger.info("registry built in stub mode (%d keys)", len(registry.registered_keys()))
        return registry

    registry.register(
        DETECTOR_KEY,
        lambda: YoloDetector(
            settings.detector_weights,
            device=device,
            conf=settings.detector_conf_threshold,
            iou=settings.detector_iou_threshold,
        ),
        metadata={
            "backend": "ultralytics",
            "weights": str(settings.detector_weights),
            "device": device,
        },
    )

    # Every variant is registered; the LRU cache decides what stays resident.
    weights_dir = settings.detector_weights.parent
    for variant, filename in SAM_CHECKPOINTS.items():
        checkpoint = weights_dir / filename
        if variant == "fast_sam":
            loader = lambda ckpt=checkpoint: FastSamSegmenter(ckpt, device=device)  # noqa: E731
        else:
            loader = lambda ckpt=checkpoint, v=variant: SamSegmenter(  # noqa: E731
                ckpt,
                variant=v,
                device=device,
                model_type=SAM_MODEL_TYPES.get(v, "vit_h"),
            )
        registry.register(
            segmenter_key(variant),
            loader,
            metadata={
                "backend": "fastsam" if variant == "fast_sam" else "sam",
                "variant": variant,
                "checkpoint": str(checkpoint),
                "device": device,
            },
        )

    logger.info(
        "registry built in real mode on %s (%d keys, capacity %d)",
        device, len(registry.registered_keys()), settings.model_cache_size,
    )
    return registry
