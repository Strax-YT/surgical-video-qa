"""Configuration. All values are overridable by SVQA_* environment variables."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Backend = Literal["stub", "real"]
SegmenterVariant = Literal["sam", "sam2", "mobile_sam", "fast_sam"]
Device = Literal["auto", "cpu", "cuda"]

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SVQA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # "stub" swaps every heavy model for a deterministic fake so the pipeline,
    # the API and CI all run on a laptop with no weights downloaded.
    backend: Backend = "stub"

    # --- vision ---
    detector_weights: Path = REPO_ROOT / "data/artifacts/yolov8n-surgical.pt"
    detector_conf_threshold: float = 0.35
    detector_iou_threshold: float = 0.50
    segmenter_variant: SegmenterVariant = "mobile_sam"
    device: Device = "auto"
    model_cache_size: int = Field(default=2, ge=1, le=8)

    # --- frame sampling ---
    sample_fps: float = Field(default=2.0, gt=0)
    scene_change_threshold: float = Field(default=0.0, ge=0.0)  # 0 disables

    # --- event derivation ---
    event_gap_tolerance_s: float = Field(default=1.0, ge=0)
    event_min_duration_s: float = Field(default=0.5, ge=0)

    # --- audio ---
    whisper_model: str = "base"
    whisper_compute_type: str = "int8"

    # --- neo4j ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme-local-only"
    neo4j_database: str = "neo4j"

    # --- vector store ---
    chroma_path: Path = REPO_ROOT / "data/artifacts/chroma"
    chroma_collection: str = "transcript_segments"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_top_k: int = Field(default=5, ge=1, le=50)

    # --- llm ---
    gemini_api_key: str = ""
    llm_model: str = "gemini-1.5-flash"
    llm_timeout_s: float = 30.0

    def resolved_device(self) -> str:
        """'auto' becomes cuda only if torch actually sees a GPU."""
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
