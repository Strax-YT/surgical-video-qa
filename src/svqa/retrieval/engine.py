"""The Ask-Anything engine.

plan -> retrieve (graph and/or vector) -> merge -> synthesise -> cite.

The merge step is where hybrid retrieval either works or produces mush. Graph
rows and transcript chunks are not comparable by score, so they are not merged
by score: graph evidence is placed first because it is factual and exact, and
transcript evidence follows, truncated to a budget. The synthesiser then sees
structure before narration, which is the order the answer should reason in.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from svqa.graph.client import GraphStore
from svqa.graph.guard import UnsafeCypherError
from svqa.retrieval.llm import (
    CYPHER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    LLMClient,
    build_prompt,
)
from svqa.retrieval.planner import QueryPlan, plan_query
from svqa.retrieval.vector_store import VectorStore
from svqa.types import Answer, Evidence
from svqa.vision.detector import INSTRUMENT_CLASSES

logger = logging.getLogger(__name__)

MAX_GRAPH_EVIDENCE = 12
MAX_VECTOR_EVIDENCE = 5


class AskEngine:
    def __init__(
        self,
        graph: GraphStore,
        vectors: VectorStore,
        llm: LLMClient,
        *,
        known_instruments: tuple[str, ...] = INSTRUMENT_CLASSES,
        known_phases: tuple[str, ...] = (),
        vector_top_k: int = 5,
    ) -> None:
        self._graph = graph
        self._vectors = vectors
        self._llm = llm
        self._instruments = known_instruments
        self._phases = known_phases
        self._top_k = vector_top_k

    def ask(
        self,
        question: str,
        video_id: str,
        *,
        video_duration_s: float | None = None,
    ) -> Answer:
        started = time.perf_counter()
        plan = plan_query(
            question,
            video_id,
            known_instruments=self._instruments,
            known_phases=self._phases,
            default_limit=self._top_k,
            video_duration_s=video_duration_s,
        )
        logger.info(
            "routed %r -> %s (%s)",
            question, plan.strategy.value, plan.decision.reason,
        )

        evidence: list[Evidence] = []
        if plan.uses_graph:
            evidence.extend(self._retrieve_graph(plan, question, video_id))
        if plan.uses_vector:
            evidence.extend(self._retrieve_vector(plan, question, video_id))

        if not evidence:
            return Answer(
                question=question,
                text=(
                    "Nothing in the indexed events or narration for this video "
                    "matches that question."
                ),
                strategy=plan.strategy,
                evidence=[],
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        prompt = build_prompt(question, evidence, plan.strategy)
        try:
            text = self._llm.generate(prompt, system=SYSTEM_PROMPT).strip()
        except Exception:  # noqa: BLE001
            logger.exception("LLM generation failed; returning evidence only")
            text = ""
        if not text:
            text = _fallback_answer(evidence)

        return Answer(
            question=question,
            text=text,
            strategy=plan.strategy,
            evidence=evidence,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    # ------------------------------------------------------------- retrieval

    def _retrieve_graph(
        self, plan: QueryPlan, question: str, video_id: str
    ) -> list[Evidence]:
        rows: list[dict[str, Any]] = []
        source = plan.template or "generated"
        if plan.template:
            try:
                rows = self._graph.run_template(plan.template, plan.params)
            except (KeyError, NotImplementedError):
                logger.warning("template %s unavailable", plan.template, exc_info=True)

        # Only if the template found nothing and the plan permits it.
        if not rows and plan.allow_generated_cypher:
            rows = self._generated_cypher(question, video_id)
            source = "generated"

        return [
            Evidence(
                source="graph",
                content=_row_to_sentence(row),
                start_s=_first_float(row, ("start_s", "next_start_s", "after_s",
                                          "first_seen_s")),
                end_s=_first_float(row, ("end_s",)),
                metadata={"template": source, "row": row},
            )
            for row in rows[:MAX_GRAPH_EVIDENCE]
        ]

    def _generated_cypher(self, question: str, video_id: str) -> list[dict[str, Any]]:
        try:
            query = self._llm.generate(
                f"Question: {question}", system=CYPHER_SYSTEM_PROMPT
            ).strip()
        except Exception:  # noqa: BLE001
            logger.exception("cypher generation failed")
            return []
        if not query:
            return []
        query = query.removeprefix("```cypher").removeprefix("```").removesuffix("```")
        try:
            return self._graph.run_cypher(query, {"video_id": video_id})
        except UnsafeCypherError as exc:
            # Rejected queries are logged with the offending text: this is the
            # audit trail for the read-only guarantee.
            logger.warning("rejected generated cypher (%s): %s", exc, query)
            return []
        except NotImplementedError:
            return []
        except Exception:  # noqa: BLE001
            logger.exception("generated cypher failed to execute")
            return []

    def _retrieve_vector(
        self, plan: QueryPlan, question: str, video_id: str
    ) -> list[Evidence]:
        query = plan.vector_query or question
        try:
            hits = self._vectors.search(query, video_id=video_id, k=self._top_k)
        except Exception:  # noqa: BLE001
            logger.exception("vector search failed")
            return []
        return hits[:MAX_VECTOR_EVIDENCE]


# ------------------------------------------------------------------- helpers


def _row_to_sentence(row: dict[str, Any]) -> str:
    """Render a graph row as a short factual clause for the prompt.

    Graph rows go into the prompt as prose rather than JSON: models cite prose
    reliably and mangle nested JSON, and the row is preserved verbatim in
    `Evidence.metadata` for the API response anyway.
    """
    parts = []
    for key, value in row.items():
        if value is None:
            continue
        label = key.replace("_", " ")
        if isinstance(value, float):
            parts.append(f"{label}={value:.2f}")
        else:
            parts.append(f"{label}={value}")
    return ", ".join(parts)


def _first_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _fallback_answer(evidence: list[Evidence]) -> str:
    """Used when the LLM is unreachable or returns nothing.

    Degrading to cited raw evidence is the right failure mode here — the user
    still gets the retrieved facts and can see exactly where they came from.
    """
    lines = [f"[{i}] {item.content}" for i, item in enumerate(evidence[:5], start=1)]
    return (
        "Answer synthesis is unavailable, so here is the retrieved evidence:\n"
        + "\n".join(lines)
    )


def build_engine(
    settings: Any,
    graph: GraphStore,
    vectors: VectorStore,
    llm: LLMClient,
    known_phases: tuple[str, ...] = (),
) -> AskEngine:
    return AskEngine(
        graph,
        vectors,
        llm,
        known_phases=known_phases,
        vector_top_k=settings.vector_top_k,
    )
