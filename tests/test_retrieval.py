"""Graph store, vector store, and the end-to-end ask path."""

from __future__ import annotations

import pytest

from svqa.graph.client import InMemoryGraphStore
from svqa.graph.events import derive_events, infer_phases
from svqa.graph.schema import DEFAULT_PHASE_RULES, TEMPLATE_QUERIES
from svqa.retrieval.engine import AskEngine
from svqa.retrieval.llm import StubLLM
from svqa.retrieval.vector_store import InMemoryVectorStore
from svqa.types import (
    Detection,
    FrameDetections,
    RetrievalStrategy,
    TranscriptSegment,
    VideoIngestResult,
)
from svqa.vision.detector import INSTRUMENT_CLASSES

VIDEO_ID = "test-video"


def _frames():
    """Grasper throughout; hook 2-6 s; clipper 8-12 s."""
    frames = []
    for i in range(24):  # 0.0 - 11.5 s at 0.5 s
        timestamp = i * 0.5
        labels = ["grasper"]
        if 2.0 <= timestamp <= 6.0:
            labels.append("hook")
        if 8.0 <= timestamp <= 12.0:
            labels.append("clipper")
        frames.append(
            FrameDetections(
                frame_index=i,
                timestamp_s=timestamp,
                detections=tuple(
                    Detection(label=label, confidence=0.9, bbox=(0, 0, 10, 10))
                    for label in labels
                ),
            )
        )
    return frames


@pytest.fixture
def ingested():
    events = derive_events(_frames())
    phases = infer_phases(events, DEFAULT_PHASE_RULES, duration_s=12.0)
    transcript = [
        TranscriptSegment(0.0, 4.0, "Grasper in, retracting the fundus."),
        TranscriptSegment(4.0, 8.0, "Hook to open the peritoneum, some adhesion here."),
        TranscriptSegment(8.0, 12.0, "Clipping the cystic duct, two clips proximal."),
    ]
    result = VideoIngestResult(
        video_id=VIDEO_ID,
        frames_sampled=len(_frames()),
        detections_total=40,
        events=events,
        transcript=transcript,
        phases=phases,
        duration_s=12.0,
    )
    graph = InMemoryGraphStore()
    counts = graph.ingest(result, filename="test.mp4", fps=10.0)
    vectors = InMemoryVectorStore()
    vectors.index_segments(VIDEO_ID, transcript)
    return graph, vectors, result, counts


def test_ingest_counts(ingested):
    _, _, _, counts = ingested
    assert counts["events"] >= 3
    assert counts["segments"] == 3
    assert counts["event_segment_links"] > 0
    assert counts["sequence_links"] == counts["events"] - 1


def test_every_template_is_implemented_by_the_in_memory_store(ingested):
    """Guards against a template being added to schema.py with no fake behind
    it — the failure would otherwise only show up at runtime in stub mode."""
    graph, _, _, _ = ingested
    params = {
        "video_id": VIDEO_ID,
        "instrument": "hook",
        "phase": "dissection",
        "start_s": 0.0,
        "end_s": 12.0,
        "limit": 5,
    }
    for name in TEMPLATE_QUERIES:
        rows = graph.run_template(name, params)
        assert isinstance(rows, list)


def test_instruments_in_video(ingested):
    graph, _, _, _ = ingested
    rows = graph.run_template("instruments_in_video", {"video_id": VIDEO_ID})
    labels = {row["instrument"] for row in rows}
    assert {"grasper", "hook", "clipper"} <= labels
    # Sorted by total duration, and the grasper is present throughout.
    assert rows[0]["instrument"] == "grasper"


def test_events_in_window_excludes_out_of_range(ingested):
    graph, _, _, _ = ingested
    rows = graph.run_template(
        "events_in_window", {"video_id": VIDEO_ID, "start_s": 0.0, "end_s": 3.0}
    )
    assert "clipper" not in {row["instrument"] for row in rows}


def test_narration_during_instrument(ingested):
    graph, _, _, _ = ingested
    rows = graph.run_template(
        "narration_during_instrument",
        {"video_id": VIDEO_ID, "instrument": "clipper", "limit": 5},
    )
    assert rows
    assert "clip" in rows[0]["text"].lower()


def test_what_followed(ingested):
    graph, _, _, _ = ingested
    rows = graph.run_template(
        "what_followed", {"video_id": VIDEO_ID, "instrument": "grasper", "limit": 5}
    )
    assert rows and "next_instrument" in rows[0]


def test_generated_cypher_is_refused_by_the_fake(ingested):
    graph, _, _, _ = ingested
    with pytest.raises(NotImplementedError):
        graph.run_cypher("MATCH (v:Video) RETURN v LIMIT 1", {})


def test_vector_search_ranks_the_relevant_segment_first(ingested):
    _, vectors, _, _ = ingested
    hits = vectors.search("adhesion", video_id=VIDEO_ID, k=3)
    assert hits
    assert "adhesion" in hits[0].content.lower()


def test_vector_search_scoped_to_video(ingested):
    _, vectors, _, _ = ingested
    assert vectors.search("adhesion", video_id="other-video", k=3) == []


def _engine(graph, vectors):
    return AskEngine(
        graph,
        vectors,
        StubLLM(),
        known_instruments=INSTRUMENT_CLASSES,
        known_phases=tuple(name for name, _ in DEFAULT_PHASE_RULES),
    )


def test_ask_graph_question(ingested):
    graph, vectors, _, _ = ingested
    answer = _engine(graph, vectors).ask("when was the hook used", VIDEO_ID)
    assert answer.strategy is RetrievalStrategy.GRAPH
    assert answer.evidence
    assert all(item.source == "graph" for item in answer.evidence)
    assert answer.latency_ms >= 0


def test_ask_vector_question(ingested):
    graph, vectors, _, _ = ingested
    answer = _engine(graph, vectors).ask(
        "what did the surgeon say about adhesion", VIDEO_ID
    )
    assert answer.strategy is RetrievalStrategy.VECTOR
    assert any(item.source == "transcript" for item in answer.evidence)


def test_ask_hybrid_draws_on_both_sources(ingested):
    graph, vectors, _, _ = ingested
    answer = _engine(graph, vectors).ask(
        "what was said while the clipper was out", VIDEO_ID
    )
    assert answer.strategy is RetrievalStrategy.HYBRID
    sources = {item.source for item in answer.evidence}
    assert sources == {"graph", "transcript"}
    # Graph evidence is ordered ahead of narration.
    assert answer.evidence[0].source == "graph"


def test_answer_is_cited(ingested):
    graph, vectors, _, _ = ingested
    answer = _engine(graph, vectors).ask("when was the hook used", VIDEO_ID)
    assert "[1]" in answer.text


def test_unknown_video_answers_without_inventing(ingested):
    graph, vectors, _, _ = ingested
    answer = _engine(graph, vectors).ask("when was the hook used", "no-such-video")
    assert answer.evidence == []
    assert "nothing" in answer.text.lower()


def test_stemming_collapses_instrument_variants():
    from svqa.retrieval.vector_store import stem, tokenize

    assert stem("clipping") == stem("clipper")
    assert stem("grasping") == stem("grasper")
    # Short words are left alone rather than mangled.
    assert stem("was") == "was"
    assert "adhesion" in tokenize("some adhesion from prior inflammation")


def test_vector_search_matches_a_morphological_variant(ingested):
    """'clipper' (a class label) must find 'Clipping' (what narration says)."""
    _, vectors, _, _ = ingested
    hits = vectors.search("clipper", video_id=VIDEO_ID, k=3)
    assert hits
    assert "clipping" in hits[0].content.lower()
