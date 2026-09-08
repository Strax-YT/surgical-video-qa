"""Query routing and parameter extraction."""

from __future__ import annotations

import pytest

from svqa.retrieval.planner import (
    classify,
    extract_instrument,
    extract_time_window,
    plan_query,
)
from svqa.types import RetrievalStrategy
from svqa.vision.detector import INSTRUMENT_CLASSES


@pytest.mark.parametrize(
    "question,expected",
    [
        ("When was the clipper used?", RetrievalStrategy.GRAPH),
        ("How many instruments appear?", RetrievalStrategy.GRAPH),
        ("How long was the hook in use?", RetrievalStrategy.GRAPH),
        ("What did the surgeon say about the adhesion?", RetrievalStrategy.VECTOR),
        ("Why did they mention pressure?", RetrievalStrategy.VECTOR),
        ("What was said while the clipper was out?", RetrievalStrategy.HYBRID),
        ("Summarise the procedure", RetrievalStrategy.HYBRID),
    ],
)
def test_routing(question: str, expected: RetrievalStrategy):
    assert classify(question).strategy is expected


def test_unsignalled_question_defaults_to_hybrid_with_low_confidence():
    decision = classify("gallbladder")
    assert decision.strategy is RetrievalStrategy.HYBRID
    assert decision.confidence < 0.6  # engine's cue to consult the LLM router


def test_instrument_extraction_handles_variants_and_plurals():
    assert extract_instrument("where is the grasper", INSTRUMENT_CLASSES) == "grasper"
    assert extract_instrument("two graspers", INSTRUMENT_CLASSES) == "grasper"
    assert (
        extract_instrument("the specimen bag", INSTRUMENT_CLASSES) == "specimen_bag"
    )
    assert extract_instrument("no tools here", INSTRUMENT_CLASSES) is None


def test_time_window_between():
    assert extract_time_window("between 2 and 4 minutes") == (120.0, 240.0)


def test_time_window_first_n():
    assert extract_time_window("in the first 90 seconds") == (0.0, 90.0)


def test_time_window_timestamp():
    start, end = extract_time_window("what happened at 3:20?")
    assert start == 185.0 and end == 215.0


def test_time_window_absent():
    assert extract_time_window("which instruments were used") is None


def test_plan_picks_timeline_template_for_an_instrument():
    plan = plan_query(
        "when was the hook used", "vid1", known_instruments=INSTRUMENT_CLASSES
    )
    assert plan.template == "instrument_timeline"
    assert plan.params["instrument"] == "hook"
    assert plan.params["video_id"] == "vid1"


def test_plan_picks_what_followed():
    plan = plan_query(
        "what came after the clipper", "vid1", known_instruments=INSTRUMENT_CLASSES
    )
    assert plan.template == "what_followed"


def test_plan_picks_window_template():
    plan = plan_query(
        "which instruments were used between 1 and 2 minutes",
        "vid1",
        known_instruments=INSTRUMENT_CLASSES,
    )
    assert plan.template == "events_in_window"
    assert plan.params["start_s"] == 60.0


def test_plan_resolves_open_ended_after_window():
    plan = plan_query(
        "which tools appeared after 2 minutes",
        "vid1",
        known_instruments=INSTRUMENT_CLASSES,
        video_duration_s=600.0,
    )
    assert plan.params["end_s"] == 600.0


def test_hybrid_plan_populates_both_paths():
    plan = plan_query(
        "what was said while the clipper was out",
        "vid1",
        known_instruments=INSTRUMENT_CLASSES,
    )
    assert plan.uses_graph and plan.uses_vector
    assert plan.template == "narration_during_instrument"
    assert plan.vector_query and "clipper" in plan.vector_query


def test_vector_only_plan_has_no_template():
    plan = plan_query(
        "why did the surgeon mention inflammation",
        "vid1",
        known_instruments=INSTRUMENT_CLASSES,
    )
    assert plan.template is None
    assert plan.uses_vector
