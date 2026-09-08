"""The read-only Cypher guard."""

from __future__ import annotations

import pytest

from svqa.graph.guard import UnsafeCypherError, validate_read_only


def test_plain_read_query_passes():
    result = validate_read_only(
        "MATCH (v:Video {video_id: $video_id}) RETURN v LIMIT 10"
    )
    assert result.applied_limit == 10


@pytest.mark.parametrize(
    "query",
    [
        "CREATE (n:Evil) RETURN n",
        "MATCH (n) DETACH DELETE n",
        "MATCH (n:Video) SET n.name = 'x' RETURN n",
        "MATCH (n) REMOVE n:Video RETURN n",
        "DROP CONSTRAINT video_id",
        "MATCH (n) CALL dbms.shutdown()",
        "LOAD CSV FROM 'file:///etc/passwd' AS row RETURN row",
        "MATCH (n:Video) MERGE (m:Video {video_id: 'x'}) RETURN m",
    ],
)
def test_write_clauses_are_rejected(query: str):
    with pytest.raises(UnsafeCypherError):
        validate_read_only(query)


def test_second_statement_cannot_smuggle_a_write():
    with pytest.raises(UnsafeCypherError):
        validate_read_only(
            "MATCH (v:Video) RETURN v LIMIT 1; MATCH (n) DETACH DELETE n"
        )


def test_keyword_inside_a_string_literal_is_allowed():
    """A transcript quote containing 'delete' must not trip the guard."""
    result = validate_read_only(
        "MATCH (s:Segment) WHERE s.text CONTAINS 'delete that clip' "
        "RETURN s.text LIMIT 5"
    )
    assert "delete that clip" in result.query


def test_missing_limit_is_added():
    result = validate_read_only("MATCH (v:Video) RETURN v")
    assert "LIMIT 200" in result.query
    assert result.notes


def test_oversized_limit_is_clamped():
    result = validate_read_only("MATCH (v:Video) RETURN v LIMIT 99999")
    assert result.applied_limit == 200
    assert "LIMIT 200" in result.query


def test_empty_query_rejected():
    with pytest.raises(UnsafeCypherError):
        validate_read_only("   ")


def test_non_read_opener_rejected():
    with pytest.raises(UnsafeCypherError):
        validate_read_only("RETURN 1 UNION CREATE (n) RETURN n")
