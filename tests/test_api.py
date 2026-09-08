"""API contract tests, running entirely in stub mode."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from svqa.api.deps import build_services
from svqa.api.main import create_app
from svqa.config import Settings


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(backend="stub", chroma_path=tmp_path / "chroma")
    app = create_app()
    # Substitute the state directly so the test never depends on process env.
    app.dependency_overrides = {}
    app.state.services = build_services(settings)
    with TestClient(app) as test_client:
        yield test_client
    app.state.services.close()


def test_healthz(client: TestClient):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend"] == "stub"


def test_models_start_registered_but_not_resident(client: TestClient):
    body = client.get("/models").json()
    assert "detector" in body["registered"]
    assert any(k.startswith("segmenter:") for k in body["registered"])
    assert body["resident"] == []


def test_load_and_unload_model(client: TestClient):
    loaded = client.post("/models/load", json={"key": "detector"}).json()
    assert loaded["loaded"] is True
    assert loaded["was_resident"] is False

    again = client.post("/models/load", json={"key": "detector"}).json()
    assert again["was_resident"] is True

    removed = client.delete("/models/detector").json()
    assert removed["unloaded"] is True


def test_load_unknown_model_is_404(client: TestClient):
    assert client.post("/models/load", json={"key": "nope"}).status_code == 404


def test_lru_eviction_visible_through_the_api(client: TestClient):
    """Capacity is 2 by default, so a third distinct model evicts the first."""
    for key in ("detector", "segmenter:mobile_sam", "segmenter:fast_sam"):
        assert client.post("/models/load", json={"key": key}).status_code == 200
    stats = client.get("/models").json()
    assert len(stats["resident"]) <= stats["capacity"]
    assert stats["evictions"] >= 1


def test_ingest_missing_file_is_404(client: TestClient):
    response = client.post("/videos/ingest", json={"path": "/no/such/file.mp4"})
    assert response.status_code == 404


def test_route_endpoint(client: TestClient):
    body = client.get("/ask/route", params={"question": "when was the hook used"}).json()
    assert body["strategy"] == "graph"
    assert body["confidence"] > 0.5


def test_ask_validates_question_length(client: TestClient):
    response = client.post("/ask", json={"question": "a", "video_id": "x"})
    assert response.status_code == 422


def test_full_ingest_then_ask(client: TestClient, sample_video: Path):
    ingested = client.post(
        "/videos/ingest",
        json={"path": str(sample_video), "sample_fps": 2.0, "max_frames": 24},
    )
    assert ingested.status_code == 200, ingested.text
    body = ingested.json()
    video_id = body["video_id"]

    assert body["frames_sampled"] > 0
    assert body["detections_total"] > 0
    assert body["events"] > 0
    assert body["transcript_segments"] > 0
    assert "grasper" in body["instruments"]
    assert "total" in body["timings_ms"]

    overview = client.get(f"/videos/{video_id}/overview").json()
    assert overview["events"] == body["events"]

    instruments = client.get(f"/videos/{video_id}/instruments").json()
    assert instruments["instruments"]

    graph_answer = client.post(
        "/ask", json={"question": "which instruments were used?", "video_id": video_id}
    ).json()
    assert graph_answer["strategy"] == "graph"
    assert graph_answer["evidence"]

    narration_answer = client.post(
        "/ask",
        json={
            "question": "what did the surgeon say about the cystic duct?",
            "video_id": video_id,
        },
    ).json()
    assert narration_answer["strategy"] == "vector"
    assert any(e["source"] == "transcript" for e in narration_answer["evidence"])
    assert all(e["citation"] for e in narration_answer["evidence"])


def test_benchmark_endpoint(client: TestClient, sample_video: Path):
    response = client.post(
        "/benchmark",
        json={
            "path": str(sample_video),
            "detector_keys": ["detector"],
            "segmenter_keys": ["segmenter:mobile_sam"],
            "frames": 12,
            "warmup_frames": 2,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    stages = {row["stage"] for row in body["rows"]}
    assert "detect" in stages
    for row in body["rows"]:
        assert row["mean_ms"] >= 0
        assert row["p95_ms"] >= row["p50_ms"]
