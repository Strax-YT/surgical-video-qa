"""Neo4j schema.

    (:Video {video_id, filename, duration_s, fps, ingested_at})
      -[:HAS_EVENT]->    (:Event {event_id, start_s, end_s, duration_s,
                                  frame_count, mean_confidence})
      -[:HAS_PHASE]->    (:Phase {name, start_s, end_s, index})
      -[:HAS_SEGMENT]->  (:Segment {segment_id, start_s, end_s, text})

    (:Event)-[:OF_INSTRUMENT]->  (:Instrument {name})
    (:Event)-[:DURING]->         (:Phase)
    (:Event)-[:MENTIONED_DURING {overlap_s}]-> (:Segment)
    (:Event)-[:PRECEDES {gap_s}]-> (:Event)

Why these choices:

  * `Instrument` is its own node rather than a property on `Event` so
    "which instruments appear across all videos" is a one-hop traversal and
    instrument-level metadata has somewhere to live.
  * `PRECEDES` between consecutive events is denormalised on write. Ordering by
    timestamp is easy in Cypher, but "what happened immediately after the
    clipper" is what people actually ask, and an explicit edge makes that a
    traversal instead of a sort.
  * `MENTIONED_DURING` carries `overlap_s` so the retriever can rank by how much
    of the utterance actually fell inside the event.
"""

from __future__ import annotations

# Executed once at startup; all are idempotent.
CONSTRAINTS = (
    "CREATE CONSTRAINT video_id IF NOT EXISTS "
    "FOR (v:Video) REQUIRE v.video_id IS UNIQUE",
    "CREATE CONSTRAINT event_id IF NOT EXISTS "
    "FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
    "CREATE CONSTRAINT segment_id IF NOT EXISTS "
    "FOR (s:Segment) REQUIRE s.segment_id IS UNIQUE",
    "CREATE CONSTRAINT instrument_name IF NOT EXISTS "
    "FOR (i:Instrument) REQUIRE i.name IS UNIQUE",
)

INDEXES = (
    # Range indexes: every temporal question filters on these.
    "CREATE INDEX event_start IF NOT EXISTS FOR (e:Event) ON (e.start_s)",
    "CREATE INDEX event_end IF NOT EXISTS FOR (e:Event) ON (e.end_s)",
    "CREATE INDEX segment_start IF NOT EXISTS FOR (s:Segment) ON (s.start_s)",
    "CREATE INDEX phase_name IF NOT EXISTS FOR (p:Phase) ON (p.name)",
    # Full-text over narration, for keyword questions that don't need embeddings.
    "CREATE FULLTEXT INDEX segment_text IF NOT EXISTS "
    "FOR (s:Segment) ON EACH [s.text]",
)

# ------------------------------------------------------------------- ingest

MERGE_VIDEO = """
MERGE (v:Video {video_id: $video_id})
SET v.filename    = $filename,
    v.duration_s  = $duration_s,
    v.fps         = $fps,
    v.frames_sampled = $frames_sampled,
    v.ingested_at = datetime(),
    v.detector    = $detector,
    v.segmenter   = $segmenter
RETURN v.video_id AS video_id
"""

# Wipe prior derived nodes so re-ingesting a video is idempotent rather than
# additive. The Video node itself is kept so its id stays stable.
DELETE_VIDEO_CHILDREN = """
MATCH (v:Video {video_id: $video_id})-[]->(n)
WHERE n:Event OR n:Segment OR n:Phase
DETACH DELETE n
"""

CREATE_EVENTS = """
MATCH (v:Video {video_id: $video_id})
UNWIND $events AS ev
MERGE (i:Instrument {name: ev.label})
CREATE (e:Event {
    event_id:        ev.event_id,
    start_s:         ev.start_s,
    end_s:           ev.end_s,
    duration_s:      ev.duration_s,
    frame_count:     ev.frame_count,
    mean_confidence: ev.mean_confidence,
    peak_confidence: ev.peak_confidence
})
CREATE (v)-[:HAS_EVENT]->(e)
CREATE (e)-[:OF_INSTRUMENT]->(i)
RETURN count(e) AS created
"""

CREATE_SEGMENTS = """
MATCH (v:Video {video_id: $video_id})
UNWIND $segments AS seg
CREATE (s:Segment {
    segment_id: seg.segment_id,
    start_s:    seg.start_s,
    end_s:      seg.end_s,
    text:       seg.text,
    speaker:    seg.speaker
})
CREATE (v)-[:HAS_SEGMENT]->(s)
RETURN count(s) AS created
"""

CREATE_PHASES = """
MATCH (v:Video {video_id: $video_id})
UNWIND $phases AS ph
CREATE (p:Phase {
    name:    ph.name,
    start_s: ph.start_s,
    end_s:   ph.end_s,
    index:   ph.index
})
CREATE (v)-[:HAS_PHASE]->(p)
RETURN count(p) AS created
"""

LINK_EVENTS_TO_PHASES = """
MATCH (v:Video {video_id: $video_id})-[:HAS_EVENT]->(e:Event)
MATCH (v)-[:HAS_PHASE]->(p:Phase)
WHERE e.start_s < p.end_s AND p.start_s < e.end_s
MERGE (e)-[:DURING]->(p)
RETURN count(*) AS linked
"""

LINK_EVENTS_TO_SEGMENTS = """
UNWIND $links AS link
MATCH (e:Event {event_id: link.event_id})
MATCH (s:Segment {segment_id: link.segment_id})
MERGE (e)-[m:MENTIONED_DURING]->(s)
SET m.overlap_s = link.overlap_s
RETURN count(m) AS linked
"""

# Chain consecutive events so "what came next" is a hop, not a sort.
LINK_EVENT_SEQUENCE = """
MATCH (v:Video {video_id: $video_id})-[:HAS_EVENT]->(e:Event)
WITH e ORDER BY e.start_s, e.event_id
WITH collect(e) AS events
UNWIND range(0, size(events) - 2) AS idx
WITH events[idx] AS a, events[idx + 1] AS b
MERGE (a)-[r:PRECEDES]->(b)
SET r.gap_s = b.start_s - a.end_s
RETURN count(r) AS linked
"""

# ------------------------------------------------------- read-only templates
# Parameterised templates for the common question shapes. The router prefers
# these over generated Cypher — a template that covers the question is always
# safer and faster than asking an LLM to write SQL-like text.

TEMPLATE_QUERIES: dict[str, str] = {
    "instruments_in_video": """
        MATCH (v:Video {video_id: $video_id})-[:HAS_EVENT]->(e:Event)
              -[:OF_INSTRUMENT]->(i:Instrument)
        RETURN i.name AS instrument,
               count(e) AS uses,
               round(sum(e.duration_s), 2) AS total_duration_s,
               round(min(e.start_s), 2) AS first_seen_s
        ORDER BY total_duration_s DESC
    """,
    "events_in_window": """
        MATCH (v:Video {video_id: $video_id})-[:HAS_EVENT]->(e:Event)
              -[:OF_INSTRUMENT]->(i:Instrument)
        WHERE e.start_s < $end_s AND $start_s < e.end_s
        RETURN i.name AS instrument, e.start_s AS start_s, e.end_s AS end_s,
               round(e.mean_confidence, 3) AS confidence
        ORDER BY e.start_s
    """,
    "instrument_timeline": """
        MATCH (v:Video {video_id: $video_id})-[:HAS_EVENT]->(e:Event)
              -[:OF_INSTRUMENT]->(i:Instrument {name: $instrument})
        RETURN e.start_s AS start_s, e.end_s AS end_s,
               e.duration_s AS duration_s, e.frame_count AS frames
        ORDER BY e.start_s
    """,
    "instruments_in_phase": """
        MATCH (v:Video {video_id: $video_id})-[:HAS_PHASE]->(p:Phase)
        WHERE toLower(p.name) CONTAINS toLower($phase)
        MATCH (e:Event)-[:DURING]->(p)
        MATCH (e)-[:OF_INSTRUMENT]->(i:Instrument)
        RETURN p.name AS phase, i.name AS instrument,
               count(e) AS uses, round(sum(e.duration_s), 2) AS duration_s
        ORDER BY p.start_s, duration_s DESC
    """,
    "narration_during_instrument": """
        MATCH (v:Video {video_id: $video_id})-[:HAS_EVENT]->(e:Event)
              -[:OF_INSTRUMENT]->(i:Instrument {name: $instrument})
        MATCH (e)-[m:MENTIONED_DURING]->(s:Segment)
        RETURN s.start_s AS start_s, s.end_s AS end_s, s.text AS text,
               round(m.overlap_s, 2) AS overlap_s
        ORDER BY m.overlap_s DESC, s.start_s
        LIMIT $limit
    """,
    "what_followed": """
        MATCH (v:Video {video_id: $video_id})-[:HAS_EVENT]->(e:Event)
              -[:OF_INSTRUMENT]->(:Instrument {name: $instrument})
        MATCH (e)-[r:PRECEDES]->(next:Event)-[:OF_INSTRUMENT]->(ni:Instrument)
        RETURN e.start_s AS after_s, ni.name AS next_instrument,
               next.start_s AS next_start_s, round(r.gap_s, 2) AS gap_s
        ORDER BY e.start_s
        LIMIT $limit
    """,
    "phase_summary": """
        MATCH (v:Video {video_id: $video_id})-[:HAS_PHASE]->(p:Phase)
        OPTIONAL MATCH (e:Event)-[:DURING]->(p)
        RETURN p.index AS idx, p.name AS phase, p.start_s AS start_s,
               p.end_s AS end_s, count(e) AS events
        ORDER BY p.index
    """,
    "video_overview": """
        MATCH (v:Video {video_id: $video_id})
        OPTIONAL MATCH (v)-[:HAS_EVENT]->(e:Event)
        OPTIONAL MATCH (v)-[:HAS_SEGMENT]->(s:Segment)
        OPTIONAL MATCH (v)-[:HAS_PHASE]->(p:Phase)
        RETURN v.filename AS filename, v.duration_s AS duration_s,
               count(DISTINCT e) AS events, count(DISTINCT s) AS segments,
               count(DISTINCT p) AS phases
    """,
}

# Default phase rules: (phase name, instruments that must be co-present).
# Tuned for laparoscopic cholecystectomy; override per procedure in config.
DEFAULT_PHASE_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    ("preparation", frozenset({"grasper"})),
    ("dissection", frozenset({"grasper", "hook"})),
    ("coagulation", frozenset({"bipolar"})),
    ("clipping", frozenset({"clipper"})),
    ("cutting", frozenset({"scissors"})),
    ("clipping_and_cutting", frozenset({"clipper", "scissors"})),
    ("irrigation", frozenset({"irrigator"})),
    ("extraction", frozenset({"specimen_bag"})),
)
