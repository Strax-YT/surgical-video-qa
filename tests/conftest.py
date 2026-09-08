from __future__ import annotations

from pathlib import Path

import pytest

from svqa.config import Settings


@pytest.fixture
def stub_settings(tmp_path: Path) -> Settings:
    return Settings(
        backend="stub",
        chroma_path=tmp_path / "chroma",
        model_cache_size=2,
    )


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny synthetic video, generated with OpenCV so no fixture binary is
    committed to the repo."""
    import cv2
    import numpy as np

    path = tmp_path_factory.mktemp("media") / "sample.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240)
    )
    if not writer.isOpened():
        pytest.skip("no mp4v encoder available in this environment")
    rng = np.random.default_rng(0)
    for i in range(120):  # 12 seconds at 10 fps
        frame = np.full((240, 320, 3), 40, dtype=np.uint8)
        frame[:, :, 1] = (i * 2) % 255
        noise = rng.integers(0, 25, size=frame.shape, dtype=np.uint8)
        writer.write(cv2.add(frame, noise))
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("video encoding produced no output")
    return path
