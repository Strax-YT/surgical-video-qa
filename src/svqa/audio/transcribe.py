"""Audio transcription.

faster-whisper rather than openai-whisper: same weights, CTranslate2 backend,
roughly 4x faster on CPU with int8 quantisation, which matters because the
whole point of the stub/real split is that this runs on a laptop.

Segments come back already timestamped, which is exactly the granularity the
graph needs — a segment is a `Segment` node and its span is what
`MENTIONED_DURING` overlaps against.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from svqa.types import TranscriptSegment

logger = logging.getLogger(__name__)


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, media_path: str | Path) -> list[TranscriptSegment]: ...

    @property
    def name(self) -> str: ...


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        self._model_size = model_size
        logger.info("whisper ready: %s on %s (%s)", model_size, device, compute_type)

    @property
    def name(self) -> str:
        return f"whisper:{self._model_size}"

    def transcribe(self, media_path: str | Path) -> list[TranscriptSegment]:
        # VAD filtering drops the long silent stretches that otherwise produce
        # hallucinated segments — a known Whisper failure on quiet OR audio.
        segments, info = self._model.transcribe(
            str(media_path),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            beam_size=5,
        )
        logger.info(
            "transcribing %s (detected language %s, p=%.2f)",
            Path(media_path).name,
            getattr(info, "language", "?"),
            getattr(info, "language_probability", 0.0),
        )
        out = [
            TranscriptSegment(
                start_s=float(segment.start),
                end_s=float(segment.end),
                text=segment.text.strip(),
            )
            for segment in segments
            if segment.text and segment.text.strip()
        ]
        logger.info("transcribed %d segments", len(out))
        return out

    def close(self) -> None:
        self._model = None


class StubTranscriber:
    """Fixed synthetic narration.

    Written to overlap the instrument intervals StubDetector produces, so the
    MENTIONED_DURING edges and the hybrid retrieval path are exercised rather
    than trivially empty.
    """

    SCRIPT: tuple[tuple[float, float, str], ...] = (
        (0.0, 6.0, "Ports are in and the camera is white balanced, starting to look around."),
        (6.0, 14.0, "Grasper in, retracting the fundus superiorly to expose the triangle."),
        (14.0, 22.0, "Using the hook now to open the peritoneum over the infundibulum."),
        (22.0, 30.0, "Careful here, there is some adhesion from prior inflammation."),
        (30.0, 38.0, "Bipolar for a small bleeder on the liver bed, holding pressure first."),
        (38.0, 46.0, "Clipping the cystic duct now, two clips proximal and one distal."),
        (46.0, 54.0, "Scissors to divide between the clips, checking the clip line is secure."),
        (54.0, 62.0, "Irrigating to clear the field and confirm haemostasis before we close."),
        (62.0, 70.0, "Specimen bag in, retrieving the gallbladder through the epigastric port."),
    )

    @property
    def name(self) -> str:
        return "stub:transcriber"

    def transcribe(self, media_path: str | Path) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(start_s=start, end_s=end, text=text, speaker="surgeon")
            for start, end, text in self.SCRIPT
        ]

    def close(self) -> None:
        return None


def build_transcriber(settings: Any) -> Transcriber:
    if settings.backend == "stub":
        return StubTranscriber()
    return WhisperTranscriber(
        settings.whisper_model,
        device=settings.resolved_device(),
        compute_type=settings.whisper_compute_type,
    )
