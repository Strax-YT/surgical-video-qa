"""LLM access.

One narrow interface, two implementations. `StubLLM` is extractive rather than
generative — it composes an answer directly from the retrieved evidence. That
makes the whole QA path runnable and testable with no API key, and it doubles
as a grounding floor: if the stub's extractive answer is better than the real
model's, the prompt is losing information the retriever already found.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol, runtime_checkable

from svqa.types import Evidence, RetrievalStrategy

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about a surgical video using only the \
supplied evidence.

Rules:
- Use only the evidence blocks. If they do not contain the answer, say so \
plainly and stop.
- Cite the evidence you used by its number, like [1] or [2][4], immediately \
after the claim it supports.
- Timestamps are in seconds from the start of the video. Convert them to mm:ss \
when you mention a time.
- Do not offer clinical judgement, diagnosis, or an assessment of surgical \
technique. Report what the evidence shows.
- Be concise: two to four sentences unless the question asks for a walkthrough.
"""

CYPHER_SYSTEM_PROMPT = """You translate questions into a single read-only Neo4j \
Cypher query.

Schema:
  (:Video {video_id, filename, duration_s})
  (:Event {event_id, start_s, end_s, duration_s, frame_count, mean_confidence})
  (:Instrument {name})
  (:Phase {name, start_s, end_s, index})
  (:Segment {segment_id, start_s, end_s, text})
  (:Video)-[:HAS_EVENT]->(:Event), (:Video)-[:HAS_PHASE]->(:Phase),
  (:Video)-[:HAS_SEGMENT]->(:Segment)
  (:Event)-[:OF_INSTRUMENT]->(:Instrument), (:Event)-[:DURING]->(:Phase)
  (:Event)-[:MENTIONED_DURING {overlap_s}]->(:Segment)
  (:Event)-[:PRECEDES {gap_s}]->(:Event)

Rules:
- MATCH/RETURN only. Never CREATE, MERGE, SET, DELETE, REMOVE or CALL.
- Always filter by $video_id.
- Always include a LIMIT of 50 or less.
- Return the query only, with no explanation and no markdown fences.
"""


@runtime_checkable
class LLMClient(Protocol):
    def generate(self, prompt: str, *, system: str | None = None) -> str: ...

    @property
    def name(self) -> str: ...


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        *,
        timeout_s: float = 30.0,
        temperature: float = 0.1,
    ) -> None:
        if not api_key:
            raise ValueError("SVQA_GEMINI_API_KEY is not set")
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model_name = model
        self._model = genai.GenerativeModel(model)
        self._timeout_s = timeout_s
        # Low temperature: this is grounded extraction, not composition.
        self._config = {"temperature": temperature, "max_output_tokens": 800}

    @property
    def name(self) -> str:
        return f"gemini:{self._model_name}"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        full = f"{system}\n\n{prompt}" if system else prompt
        response = self._model.generate_content(
            full,
            generation_config=self._config,
            request_options={"timeout": self._timeout_s},
        )
        return (getattr(response, "text", "") or "").strip()


class StubLLM:
    """Extractive answerer. Deterministic, offline, always cited."""

    @property
    def name(self) -> str:
        return "stub:extractive"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        if system is CYPHER_SYSTEM_PROMPT or "Cypher" in (system or ""):
            # Refuse rather than emit a guessed query: the guard would reject a
            # malformed one anyway, and a wrong query is worse than no answer.
            return ""

        question = _extract_section(prompt, "Question:")
        evidence_block = _extract_section(prompt, "Evidence:")
        if not evidence_block.strip():
            return "The retrieved evidence does not cover that question."

        lines = [
            line.strip() for line in evidence_block.splitlines() if line.strip()
        ]
        cited = lines[:3]
        numbers = [
            match.group(1)
            for line in cited
            if (match := re.match(r"\[(\d+)\]", line))
        ]
        citation = "".join(f"[{n}]" for n in numbers) or "[1]"
        body = " ".join(re.sub(r"^\[\d+\]\s*", "", line) for line in cited)
        return (
            f"Based on the retrieved evidence{citation}: {body}"
            f"{'' if not question else ''}"
        ).strip()


def _extract_section(prompt: str, header: str) -> str:
    """Pull one labelled section out of the assembled prompt."""
    if header not in prompt:
        return ""
    after = prompt.split(header, 1)[1]
    for next_header in ("Question:", "Evidence:", "Retrieval strategy:"):
        if next_header != header and next_header in after:
            after = after.split(next_header, 1)[0]
    return after.strip()


def build_prompt(
    question: str,
    evidence: list[Evidence],
    strategy: RetrievalStrategy,
) -> str:
    """Assemble the grounded-answer prompt.

    Evidence is numbered so citations are checkable: `Answer.evidence[i]` maps
    to `[i+1]` in the text, which is what makes automated faithfulness scoring
    possible rather than vibes-based.
    """
    blocks = []
    for i, item in enumerate(evidence, start=1):
        window = (
            f" ({item.start_s:.1f}s-{item.end_s:.1f}s)"
            if item.start_s is not None and item.end_s is not None
            else ""
        )
        blocks.append(f"[{i}] ({item.source}{window}) {item.content}")

    return (
        f"Retrieval strategy: {strategy.value}\n\n"
        f"Evidence:\n" + ("\n".join(blocks) if blocks else "(none)") + "\n\n"
        f"Question: {question}\n"
    )


def build_llm(settings: Any) -> LLMClient:
    if settings.backend == "stub" or not settings.gemini_api_key:
        if settings.backend != "stub":
            logger.warning("no SVQA_GEMINI_API_KEY set; falling back to StubLLM")
        return StubLLM()
    return GeminiClient(
        settings.gemini_api_key,
        settings.llm_model,
        timeout_s=settings.llm_timeout_s,
    )
