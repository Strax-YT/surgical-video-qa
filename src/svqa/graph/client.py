"""Graph persistence.

`GraphStore` is a Protocol with two implementations:

  Neo4jGraphStore     the real thing, over the Bolt driver
  InMemoryGraphStore  the same query semantics in Python dicts

The in-memory store exists so the API, the retriever and the whole test suite
run with no database container. It implements the *template* queries only —
generated Cypher raises NotImplementedError there, which is the honest
behaviour and keeps the fake from quietly diverging from Neo4j's semantics.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

from svqa.graph import schema
from svqa.graph.events import align_transcript
from svqa.graph.guard import validate_read_only
from svqa.types import VideoIngestResult

logger = logging.getLogger(__name__)

Row = dict[str, Any]


@runtime_checkable
class GraphStore(Protocol):
    def ensure_schema(self) -> None: ...
    def ingest(self, result: VideoIngestResult, **meta: Any) -> dict[str, int]: ...
    def run_template(self, name: str, params: Row) -> list[Row]: ...
    def run_cypher(self, query: str, params: Row) -> list[Row]: ...
    def close(self) -> None: ...


def _event_id(video_id: str, index: int, label: str) -> str:
    return f"{video_id}:ev{index:04d}:{label}"


def _segment_id(video_id: str, index: int) -> str:
    return f"{video_id}:sg{index:04d}"


def _payloads(result: VideoIngestResult) -> tuple[list[Row], list[Row], list[Row], list[Row]]:
    """Flatten an ingest result into UNWIND-ready parameter lists."""
    events = [
        {
            "event_id": _event_id(result.video_id, i, e.label),
            "label": e.label,
            "start_s": round(e.start_s, 3),
            "end_s": round(e.end_s, 3),
            "duration_s": round(e.duration_s, 3),
            "frame_count": e.frame_count,
            "mean_confidence": round(e.mean_confidence, 4),
            "peak_confidence": round(e.peak_confidence, 4),
        }
        for i, e in enumerate(result.events)
    ]
    segments = [
        {
            "segment_id": _segment_id(result.video_id, i),
            "start_s": round(s.start_s, 3),
            "end_s": round(s.end_s, 3),
            "text": s.text,
            "speaker": s.speaker,
        }
        for i, s in enumerate(result.transcript)
    ]
    phases = [
        {
            "name": p.name,
            "start_s": round(p.start_s, 3),
            "end_s": round(p.end_s, 3),
            "index": p.index,
        }
        for p in result.phases
    ]
    links = [
        {
            "event_id": events[event_index]["event_id"],
            "segment_id": segments[segment_index]["segment_id"],
            "overlap_s": round(overlap, 3),
        }
        for event_index, segment_index, overlap in align_transcript(
            result.events, result.transcript
        )
    ]
    return events, segments, phases, links


class Neo4jGraphStore:
    """Bolt-backed store. Writes in one transaction per video."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str = "neo4j",
    ) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        logger.info("connected to neo4j at %s", uri)

    def ensure_schema(self) -> None:
        with self._driver.session(database=self._database) as session:
            for statement in (*schema.CONSTRAINTS, *schema.INDEXES):
                session.run(statement)
        logger.info("schema ensured")

    def ingest(self, result: VideoIngestResult, **meta: Any) -> dict[str, int]:
        events, segments, phases, links = _payloads(result)
        counts: dict[str, int] = {}

        with self._driver.session(database=self._database) as session:
            def _write(tx: Any) -> dict[str, int]:
                tx.run(
                    schema.MERGE_VIDEO,
                    video_id=result.video_id,
                    filename=meta.get("filename", result.video_id),
                    duration_s=round(result.duration_s, 3),
                    fps=meta.get("fps", 0.0),
                    frames_sampled=result.frames_sampled,
                    detector=meta.get("detector", "unknown"),
                    segmenter=meta.get("segmenter", "unknown"),
                )
                # Re-ingest is a replace, not an append.
                tx.run(schema.DELETE_VIDEO_CHILDREN, video_id=result.video_id)
                out = {
                    "events": tx.run(
                        schema.CREATE_EVENTS,
                        video_id=result.video_id, events=events,
                    ).single()["created"],
                    "segments": tx.run(
                        schema.CREATE_SEGMENTS,
                        video_id=result.video_id, segments=segments,
                    ).single()["created"],
                    "phases": tx.run(
                        schema.CREATE_PHASES,
                        video_id=result.video_id, phases=phases,
                    ).single()["created"],
                }
                out["event_phase_links"] = tx.run(
                    schema.LINK_EVENTS_TO_PHASES, video_id=result.video_id
                ).single()["linked"]
                out["event_segment_links"] = tx.run(
                    schema.LINK_EVENTS_TO_SEGMENTS, links=links
                ).single()["linked"]
                out["sequence_links"] = tx.run(
                    schema.LINK_EVENT_SEQUENCE, video_id=result.video_id
                ).single()["linked"]
                return out

            counts = session.execute_write(_write)

        logger.info("ingested %s: %s", result.video_id, counts)
        return counts

    def run_template(self, name: str, params: Row) -> list[Row]:
        query = schema.TEMPLATE_QUERIES.get(name)
        if query is None:
            raise KeyError(f"unknown template {name!r}")
        return self._read(query, params)

    def run_cypher(self, query: str, params: Row) -> list[Row]:
        guarded = validate_read_only(query)
        if guarded.notes:
            logger.info("cypher guard: %s", "; ".join(guarded.notes))
        return self._read(guarded.query, params)

    def _read(self, query: str, params: Row) -> list[Row]:
        # READ access mode is the second layer of the read-only guarantee: the
        # server rejects writes here regardless of what the validator allowed.
        with self._driver.session(
            database=self._database, default_access_mode="READ"
        ) as session:
            return [dict(record) for record in session.run(query, **params)]

    def close(self) -> None:
        self._driver.close()


class InMemoryGraphStore:
    """Dict-backed store implementing the template query semantics.

    Deliberately not a Cypher interpreter. Each template is reimplemented in
    Python, and tests/test_graph_store.py asserts both stores agree on shape
    for the templates that matter.
    """

    def __init__(self) -> None:
        self._videos: dict[str, Row] = {}
        self._events: dict[str, list[Row]] = defaultdict(list)
        self._segments: dict[str, list[Row]] = defaultdict(list)
        self._phases: dict[str, list[Row]] = defaultdict(list)
        self._links: dict[str, list[Row]] = defaultdict(list)

    def ensure_schema(self) -> None:
        return None

    def ingest(self, result: VideoIngestResult, **meta: Any) -> dict[str, int]:
        events, segments, phases, links = _payloads(result)
        video_id = result.video_id
        self._videos[video_id] = {
            "video_id": video_id,
            "filename": meta.get("filename", video_id),
            "duration_s": round(result.duration_s, 3),
            "fps": meta.get("fps", 0.0),
            "frames_sampled": result.frames_sampled,
            "detector": meta.get("detector", "unknown"),
            "segmenter": meta.get("segmenter", "unknown"),
        }
        self._events[video_id] = events
        self._segments[video_id] = segments
        self._phases[video_id] = phases
        self._links[video_id] = links

        # Match Neo4j's PRECEDES chain so "what followed" works here too.
        ordered = sorted(events, key=lambda e: (e["start_s"], e["event_id"]))
        for a, b in zip(ordered, ordered[1:], strict=False):
            a["next_event_id"] = b["event_id"]
            a["gap_s"] = round(b["start_s"] - a["end_s"], 3)

        return {
            "events": len(events),
            "segments": len(segments),
            "phases": len(phases),
            "event_phase_links": sum(
                1
                for e in events
                for p in phases
                if e["start_s"] < p["end_s"] and p["start_s"] < e["end_s"]
            ),
            "event_segment_links": len(links),
            "sequence_links": max(0, len(ordered) - 1),
        }

    def known_videos(self) -> list[str]:
        return sorted(self._videos)

    # ------------------------------------------------------------- templates

    def run_template(self, name: str, params: Row) -> list[Row]:
        if name not in schema.TEMPLATE_QUERIES:
            raise KeyError(f"unknown template {name!r}")
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            raise NotImplementedError(
                f"template {name!r} is not implemented by InMemoryGraphStore"
            )
        return handler(params)

    def run_cypher(self, query: str, params: Row) -> list[Row]:
        validate_read_only(query)  # still exercise the guard
        raise NotImplementedError(
            "InMemoryGraphStore cannot execute generated Cypher; "
            "run against Neo4j (docker compose up neo4j) for that path."
        )

    def _events_for(self, params: Row) -> list[Row]:
        return self._events.get(params.get("video_id", ""), [])

    def _t_instruments_in_video(self, params: Row) -> list[Row]:
        grouped: dict[str, list[Row]] = defaultdict(list)
        for event in self._events_for(params):
            grouped[event["label"]].append(event)
        rows = [
            {
                "instrument": label,
                "uses": len(events),
                "total_duration_s": round(sum(e["duration_s"] for e in events), 2),
                "first_seen_s": round(min(e["start_s"] for e in events), 2),
            }
            for label, events in grouped.items()
        ]
        return sorted(rows, key=lambda r: -r["total_duration_s"])

    def _t_events_in_window(self, params: Row) -> list[Row]:
        start, end = params["start_s"], params["end_s"]
        rows = [
            {
                "instrument": e["label"],
                "start_s": e["start_s"],
                "end_s": e["end_s"],
                "confidence": round(e["mean_confidence"], 3),
            }
            for e in self._events_for(params)
            if e["start_s"] < end and start < e["end_s"]
        ]
        return sorted(rows, key=lambda r: r["start_s"])

    def _t_instrument_timeline(self, params: Row) -> list[Row]:
        target = params["instrument"]
        rows = [
            {
                "start_s": e["start_s"],
                "end_s": e["end_s"],
                "duration_s": e["duration_s"],
                "frames": e["frame_count"],
            }
            for e in self._events_for(params)
            if e["label"] == target
        ]
        return sorted(rows, key=lambda r: r["start_s"])

    def _t_instruments_in_phase(self, params: Row) -> list[Row]:
        video_id = params.get("video_id", "")
        needle = str(params.get("phase", "")).lower()
        matched = [
            p for p in self._phases.get(video_id, [])
            if needle in p["name"].lower()
        ]
        rows: list[Row] = []
        for phase in sorted(matched, key=lambda p: p["start_s"]):
            grouped: dict[str, list[Row]] = defaultdict(list)
            for event in self._events.get(video_id, []):
                if event["start_s"] < phase["end_s"] and phase["start_s"] < event["end_s"]:
                    grouped[event["label"]].append(event)
            for label, events in sorted(
                grouped.items(),
                key=lambda kv: -sum(e["duration_s"] for e in kv[1]),
            ):
                rows.append(
                    {
                        "phase": phase["name"],
                        "instrument": label,
                        "uses": len(events),
                        "duration_s": round(
                            sum(e["duration_s"] for e in events), 2
                        ),
                    }
                )
        return rows

    def _t_narration_during_instrument(self, params: Row) -> list[Row]:
        video_id = params.get("video_id", "")
        target = params["instrument"]
        limit = int(params.get("limit", 5))
        wanted = {
            e["event_id"]
            for e in self._events.get(video_id, [])
            if e["label"] == target
        }
        by_id = {s["segment_id"]: s for s in self._segments.get(video_id, [])}
        rows: list[Row] = []
        for link in self._links.get(video_id, []):
            if link["event_id"] not in wanted:
                continue
            segment = by_id.get(link["segment_id"])
            if segment is None:
                continue
            rows.append(
                {
                    "start_s": segment["start_s"],
                    "end_s": segment["end_s"],
                    "text": segment["text"],
                    "overlap_s": round(link["overlap_s"], 2),
                }
            )
        rows.sort(key=lambda r: (-r["overlap_s"], r["start_s"]))
        return rows[:limit]

    def _t_what_followed(self, params: Row) -> list[Row]:
        video_id = params.get("video_id", "")
        target = params["instrument"]
        limit = int(params.get("limit", 5))
        by_id = {e["event_id"]: e for e in self._events.get(video_id, [])}
        rows: list[Row] = []
        for event in self._events.get(video_id, []):
            if event["label"] != target:
                continue
            nxt = by_id.get(event.get("next_event_id", ""))
            if nxt is None:
                continue
            rows.append(
                {
                    "after_s": event["start_s"],
                    "next_instrument": nxt["label"],
                    "next_start_s": nxt["start_s"],
                    "gap_s": event.get("gap_s", 0.0),
                }
            )
        rows.sort(key=lambda r: r["after_s"])
        return rows[:limit]

    def _t_phase_summary(self, params: Row) -> list[Row]:
        video_id = params.get("video_id", "")
        rows = []
        for phase in sorted(self._phases.get(video_id, []), key=lambda p: p["index"]):
            events = sum(
                1
                for e in self._events.get(video_id, [])
                if e["start_s"] < phase["end_s"] and phase["start_s"] < e["end_s"]
            )
            rows.append(
                {
                    "idx": phase["index"],
                    "phase": phase["name"],
                    "start_s": phase["start_s"],
                    "end_s": phase["end_s"],
                    "events": events,
                }
            )
        return rows

    def _t_video_overview(self, params: Row) -> list[Row]:
        video_id = params.get("video_id", "")
        video = self._videos.get(video_id)
        if video is None:
            return []
        return [
            {
                "filename": video["filename"],
                "duration_s": video["duration_s"],
                "events": len(self._events.get(video_id, [])),
                "segments": len(self._segments.get(video_id, [])),
                "phases": len(self._phases.get(video_id, [])),
            }
        ]

    def close(self) -> None:
        return None


def build_graph_store(settings: Any) -> GraphStore:
    """Pick a store based on the configured backend."""
    if settings.backend == "stub":
        logger.info("using InMemoryGraphStore (backend=stub)")
        return InMemoryGraphStore()
    return Neo4jGraphStore(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        database=settings.neo4j_database,
    )
