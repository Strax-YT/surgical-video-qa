"""Ask-Anything endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from svqa.api.deps import Services, get_services
from svqa.api.schemas import AskRequest, AskResponse, EvidenceOut
from svqa.retrieval.planner import classify

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, services: Services = Depends(get_services)) -> AskResponse:
    answer = services.engine.ask(
        body.question,
        body.video_id,
        video_duration_s=services.durations.get(body.video_id),
    )
    return AskResponse(
        question=answer.question,
        answer=answer.text,
        strategy=answer.strategy.value,
        routing_reason=classify(body.question).reason,
        evidence=[
            EvidenceOut(
                source=item.source,
                content=item.content,
                start_s=item.start_s,
                end_s=item.end_s,
                citation=item.cite(),
                metadata=item.metadata,
            )
            for item in answer.evidence
        ],
        latency_ms=round(answer.latency_ms, 2),
    )


@router.get("/ask/route")
def route_only(question: str) -> dict:
    """Expose the router without retrieving anything.

    This is the endpoint the routing eval hits — asserting on strategy choice
    for a fixed question set is how you catch a signal-word change regressing
    the router, and it costs nothing to run in CI.
    """
    decision = classify(question)
    return {
        "question": question,
        "strategy": decision.strategy.value,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "matched_signals": list(decision.matched_signals),
    }
