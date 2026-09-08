"""Deciding how to answer a question.

The Ask-Anything endpoint has three retrieval paths and picking between them is
the whole game:

  GRAPH   structured, temporal, counting -> Cypher over events and phases
  VECTOR  open-ended semantic -> embedding search over narration
  HYBRID  needs both, merged before synthesis

An LLM router works but costs a call and a few hundred milliseconds on every
question, and it is non-deterministic on exactly the questions users repeat
most. So routing is rule-based first: signal words plus extracted entities
resolve the large majority of real questions deterministically, and the LLM is
only consulted when the rules are genuinely ambiguous.

The parameter extraction here is the other half — a strategy without a template
and bound parameters still leaves the LLM writing raw Cypher, which is the path
we most want to avoid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from svqa.types import RetrievalStrategy, RoutingDecision

# --------------------------------------------------------------- signal words

GRAPH_SIGNALS = {
    "when", "how many", "how long", "how much", "which instrument",
    "what instrument", "count", "duration", "timeline", "first", "last",
    "before", "after", "followed", "precede", "phase", "stage", "order",
    "sequence", "total", "longest", "shortest", "list the instrument",
    "at what point", "how often", "used", "appear",
    # Temporal conjunctions. "what was said *while* the clipper was out" reads
    # as pure narration, but the "while" clause is a constraint only the graph
    # can resolve — without these, co-occurrence questions route to vector-only
    # and silently lose the time window.
    "while", "during", "at the same time", "simultaneously",
}

VECTOR_SIGNALS = {
    "said", "say", "says", "mention", "mentioned", "discuss", "discussed",
    "explain", "explained", "describe", "described", "why", "narration",
    "commentary", "talk about", "talked about", "note", "noted", "comment",
    "concern", "warned", "advice", "recommend",
}

# Questions that are open-ended enough that graph rows alone would answer badly.
SUMMARY_SIGNALS = {"summarise", "summarize", "summary", "overview", "walk me through"}

_TIME_UNITS = {
    "second": 1.0, "seconds": 1.0, "sec": 1.0, "secs": 1.0, "s": 1.0,
    "minute": 60.0, "minutes": 60.0, "min": 60.0, "mins": 60.0, "m": 60.0,
}


@dataclass(slots=True)
class QueryPlan:
    """A resolved plan the engine can execute without further interpretation."""

    strategy: RetrievalStrategy
    decision: RoutingDecision
    template: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    vector_query: str | None = None
    allow_generated_cypher: bool = False

    @property
    def uses_graph(self) -> bool:
        return self.strategy in (RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID)

    @property
    def uses_vector(self) -> bool:
        return self.strategy in (RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID)


# ------------------------------------------------------------------ extractors


def extract_instrument(question: str, known: tuple[str, ...]) -> str | None:
    """Find a known instrument label in the question.

    Matches on underscore-or-space variants and simple plurals, and prefers the
    longest match so "specimen bag" is not shadowed by "bag".
    """
    lowered = question.lower()
    matches: list[tuple[int, str]] = []
    for label in known:
        for variant in {label, label.replace("_", " "), label.replace("_", "-")}:
            pattern = rf"\b{re.escape(variant)}(?:s|es)?\b"
            if re.search(pattern, lowered):
                matches.append((len(variant), label))
                break
    if not matches:
        return None
    return max(matches)[1]


def _to_seconds(value: float, unit: str | None) -> float:
    return value * _TIME_UNITS.get((unit or "s").lower(), 1.0)


def extract_time_window(question: str) -> tuple[float, float] | None:
    """Parse a time window from natural language.

    Handles the forms that actually turn up:
        "between 2 and 4 minutes"      -> (120, 240)
        "in the first 90 seconds"      -> (0, 90)
        "after 5 minutes"              -> (300, inf)
        "at 3:20"                      -> (200, 200+window)
        "the last 2 minutes"           -> None (needs duration; caller resolves)
    """
    lowered = question.lower()

    # between X and Y <unit>
    match = re.search(
        r"between\s+(\d+(?:\.\d+)?)\s*(\w+)?\s+and\s+(\d+(?:\.\d+)?)\s*(\w+)?",
        lowered,
    )
    if match:
        start_val, start_unit, end_val, end_unit = match.groups()
        unit = end_unit if end_unit in _TIME_UNITS else start_unit
        return (
            _to_seconds(float(start_val), start_unit if start_unit in _TIME_UNITS else unit),
            _to_seconds(float(end_val), unit),
        )

    # first/last N <unit>
    match = re.search(r"first\s+(\d+(?:\.\d+)?)\s*(\w+)", lowered)
    if match:
        value, unit = match.groups()
        return (0.0, _to_seconds(float(value), unit))

    # mm:ss timestamp
    match = re.search(r"\b(\d{1,2}):([0-5]\d)\b", lowered)
    if match:
        minutes, seconds = int(match.group(1)), int(match.group(2))
        point = minutes * 60 + seconds
        return (max(0.0, point - 15.0), point + 15.0)

    # after N <unit>
    match = re.search(r"after\s+(\d+(?:\.\d+)?)\s*(\w+)", lowered)
    if match:
        value, unit = match.groups()
        if unit in _TIME_UNITS:
            return (_to_seconds(float(value), unit), float("inf"))

    # around/at N <unit>
    match = re.search(r"(?:around|at|near)\s+(\d+(?:\.\d+)?)\s*(\w+)", lowered)
    if match:
        value, unit = match.groups()
        if unit in _TIME_UNITS:
            point = _to_seconds(float(value), unit)
            return (max(0.0, point - 15.0), point + 15.0)

    return None


def extract_phase(question: str, known_phases: tuple[str, ...]) -> str | None:
    lowered = question.lower()
    for phase in sorted(known_phases, key=len, reverse=True):
        if phase.replace("_", " ") in lowered or phase in lowered:
            return phase
    return None


def _matched(question: str, signals: set[str]) -> tuple[str, ...]:
    lowered = question.lower()
    return tuple(sorted(signal for signal in signals if signal in lowered))


# -------------------------------------------------------------------- routing


def classify(question: str) -> RoutingDecision:
    """Rule-based strategy selection.

    A confidence below ~0.6 is the engine's cue to ask the LLM router instead.
    """
    graph_hits = _matched(question, GRAPH_SIGNALS)
    vector_hits = _matched(question, VECTOR_SIGNALS)
    summary_hits = _matched(question, SUMMARY_SIGNALS)

    if summary_hits:
        return RoutingDecision(
            strategy=RetrievalStrategy.HYBRID,
            reason="summary request needs structured timeline plus narration",
            confidence=0.8,
            matched_signals=summary_hits + graph_hits + vector_hits,
        )

    if graph_hits and vector_hits:
        return RoutingDecision(
            strategy=RetrievalStrategy.HYBRID,
            reason="question mixes a temporal/structural constraint with narration",
            confidence=0.85,
            matched_signals=graph_hits + vector_hits,
        )
    if graph_hits:
        return RoutingDecision(
            strategy=RetrievalStrategy.GRAPH,
            reason=f"structural signals: {', '.join(graph_hits)}",
            confidence=0.9,
            matched_signals=graph_hits,
        )
    if vector_hits:
        return RoutingDecision(
            strategy=RetrievalStrategy.VECTOR,
            reason=f"narration signals: {', '.join(vector_hits)}",
            confidence=0.85,
            matched_signals=vector_hits,
        )

    # No signal either way. Hybrid is the safe default: it costs one extra
    # retrieval and cannot miss the modality the answer was in.
    return RoutingDecision(
        strategy=RetrievalStrategy.HYBRID,
        reason="no decisive signal; defaulting to hybrid",
        confidence=0.4,
        matched_signals=(),
    )


def plan_query(
    question: str,
    video_id: str,
    *,
    known_instruments: tuple[str, ...] = (),
    known_phases: tuple[str, ...] = (),
    decision: RoutingDecision | None = None,
    default_limit: int = 5,
    video_duration_s: float | None = None,
) -> QueryPlan:
    """Resolve a question into a strategy, a template, and bound parameters.

    Template selection is ordered most-specific first, because several
    templates can technically serve a question and the narrowest one returns
    the fewest irrelevant rows for the synthesiser to wade through.
    """
    decision = decision or classify(question)
    lowered = question.lower()
    instrument = extract_instrument(question, known_instruments)
    phase = extract_phase(question, known_phases)
    window = extract_time_window(question)
    if window and window[1] == float("inf"):
        window = (window[0], video_duration_s or 1e9)

    params: dict[str, Any] = {"video_id": video_id}
    template: str | None = None

    if decision.strategy in (RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID):
        if instrument and any(
            word in lowered for word in ("after", "next", "followed", "then")
        ):
            template = "what_followed"
            params |= {"instrument": instrument, "limit": default_limit}
        elif instrument and any(
            word in lowered for word in ("said", "say", "mention", "narrat", "comment")
        ):
            template = "narration_during_instrument"
            params |= {"instrument": instrument, "limit": default_limit}
        elif phase:
            template = "instruments_in_phase"
            params |= {"phase": phase}
        elif window:
            template = "events_in_window"
            params |= {"start_s": window[0], "end_s": window[1]}
        elif instrument:
            template = "instrument_timeline"
            params |= {"instrument": instrument}
        elif any(word in lowered for word in ("phase", "stage", "step")):
            template = "phase_summary"
        elif any(
            word in lowered
            for word in ("instrument", "tool", "used", "appear", "how many")
        ):
            template = "instruments_in_video"
        else:
            template = "video_overview"

    vector_query: str | None = None
    if decision.strategy in (RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID):
        # Give the embedding the instrument name explicitly — narration often
        # names the anatomy rather than the tool, so the bare question can miss.
        vector_query = question if not instrument else f"{question} ({instrument})"

    return QueryPlan(
        strategy=decision.strategy,
        decision=decision,
        template=template,
        params=params,
        vector_query=vector_query,
        # Generated Cypher is a last resort, only when no template fit and the
        # question is clearly structural.
        allow_generated_cypher=(
            decision.strategy is RetrievalStrategy.GRAPH
            and template == "video_overview"
            and decision.confidence >= 0.8
        ),
    )
