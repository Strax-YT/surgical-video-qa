"""Box-prompted instance segmentation.

The detector gives boxes; SAM turns each box into a pixel mask. Prompting SAM
with detector boxes rather than running it in "segment everything" mode is what
makes this tractable: everything-mode returns hundreds of unlabelled masks per
frame with no way to tie them to an instrument class, while box-prompted mode
returns exactly one mask per detection and inherits the detector's label.

Three backends behind one interface, because they have genuinely different
cost profiles and the benchmark harness needs to compare them:

  sam / sam2   ViT-H, best masks, ~2.4 GB, slow on CPU
  mobile_sam   distilled ViT-tiny image encoder, ~40 MB, viable on CPU
  fast_sam     YOLO-seg architecture, fastest, coarser boundaries
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

SAM_CHECKPOINTS = {
    "sam": "sam_vit_h_4b8939.pth",
    "sam2": "sam2_hiera_large.pt",
    "mobile_sam": "mobile_sam.pt",
    "fast_sam": "FastSAM-s.pt",
}


@dataclass(slots=True)
class MaskResult:
    """One mask, aligned by index with the box that prompted it."""

    mask: np.ndarray      # bool array, H x W
    score: float          # model's own quality estimate, where available
    area_px: int

    @classmethod
    def from_mask(cls, mask: np.ndarray, score: float = 1.0) -> MaskResult:
        boolean = mask.astype(bool)
        return cls(mask=boolean, score=float(score), area_px=int(boolean.sum()))


@runtime_checkable
class Segmenter(Protocol):
    def segment_boxes(
        self, frame: np.ndarray, boxes: list[tuple[float, float, float, float]]
    ) -> list[MaskResult]: ...

    @property
    def name(self) -> str: ...


class SamSegmenter:
    """Meta SAM / MobileSAM via the `segment_anything` predictor API."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        variant: str = "mobile_sam",
        device: str = "cpu",
        model_type: str = "vit_t",
    ) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint}. "
                "See README 'Weights' or set SVQA_BACKEND=stub."
            )
        if variant == "mobile_sam":
            from mobile_sam import SamPredictor, sam_model_registry
        else:
            from segment_anything import SamPredictor, sam_model_registry

        model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        model.to(device=device)
        model.eval()
        self._predictor = SamPredictor(model)
        self._variant = variant
        self._device = device
        logger.info("SamSegmenter ready: %s (%s) on %s", variant, model_type, device)

    @property
    def name(self) -> str:
        return f"sam:{self._variant}"

    def segment_boxes(
        self, frame: np.ndarray, boxes: list[tuple[float, float, float, float]]
    ) -> list[MaskResult]:
        if not boxes:
            return []
        import torch

        # set_image runs the expensive image encoder exactly once per frame;
        # every box prompt after that only pays for the light mask decoder.
        rgb = frame[:, :, ::-1]
        self._predictor.set_image(np.ascontiguousarray(rgb))

        box_tensor = torch.as_tensor(
            boxes, dtype=torch.float32, device=self._device
        )
        transformed = self._predictor.transform.apply_boxes_torch(
            box_tensor, frame.shape[:2]
        )
        with torch.inference_mode():
            masks, scores, _ = self._predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed,
                multimask_output=False,
            )
        masks_np = masks[:, 0].detach().cpu().numpy()
        scores_np = scores[:, 0].detach().cpu().numpy()
        return [
            MaskResult.from_mask(mask, score)
            for mask, score in zip(masks_np, scores_np, strict=True)
        ]

    def close(self) -> None:
        self._predictor = None


class FastSamSegmenter:
    """FastSAM through ultralytics, prompted with detector boxes."""

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.4,
    ) -> None:
        from ultralytics import FastSAM

        self._model = FastSAM(str(weights))
        self._device = device
        self._imgsz = imgsz
        self._conf = conf
        logger.info("FastSamSegmenter ready on %s", device)

    @property
    def name(self) -> str:
        return "sam:fast_sam"

    def segment_boxes(
        self, frame: np.ndarray, boxes: list[tuple[float, float, float, float]]
    ) -> list[MaskResult]:
        if not boxes:
            return []
        results = self._model(
            frame,
            device=self._device,
            imgsz=self._imgsz,
            conf=self._conf,
            bboxes=[list(box) for box in boxes],
            verbose=False,
        )
        out: list[MaskResult] = []
        for result in results:
            masks = getattr(result, "masks", None)
            if masks is None:
                continue
            for mask in masks.data.detach().cpu().numpy():
                out.append(MaskResult.from_mask(mask))
        # Pad so the caller's index alignment with `boxes` always holds.
        height, width = frame.shape[:2]
        while len(out) < len(boxes):
            out.append(
                MaskResult.from_mask(np.zeros((height, width), dtype=bool), 0.0)
            )
        return out[: len(boxes)]

    def close(self) -> None:
        self._model = None


class StubSegmenter:
    """Fills each box with an inscribed ellipse.

    Not a real mask, but it has the properties the downstream code depends on:
    area is a plausible fraction of the box, masks are inside frame bounds, and
    output length always matches input length.
    """

    @property
    def name(self) -> str:
        return "stub:segmenter"

    def segment_boxes(
        self, frame: np.ndarray, boxes: list[tuple[float, float, float, float]]
    ) -> list[MaskResult]:
        height, width = frame.shape[:2]
        results: list[MaskResult] = []
        yy, xx = np.mgrid[0:height, 0:width]
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            rx = max(1.0, (x2 - x1) / 2.0)
            ry = max(1.0, (y2 - y1) / 2.0)
            mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
            results.append(MaskResult.from_mask(mask, score=0.9))
        return results

    def close(self) -> None:
        return None
