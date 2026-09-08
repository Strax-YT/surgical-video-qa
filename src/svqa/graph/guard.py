"""Read-only guard for generated Cypher.

Any text-to-Cypher feature hands an LLM a database connection, and the prompt
that reaches it may contain text lifted from a transcript. So generated Cypher
is treated as untrusted input and validated in code before execution, rather
than trusted because the system prompt asked nicely for read-only queries.

Two independent layers, because either alone is bypassable:

1.  This validator: statement must start with a read clause, must contain no
    write or admin clause, must be a single statement, and must carry a LIMIT.
2.  A Neo4j session opened in `READ` access mode against a least-privilege
    user. The database refuses the write even if this validator is fooled.

Layer 2 is the real control; layer 1 gives a clear error and a cheap audit log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Clauses that mutate data, schema, or roles.
FORBIDDEN_CLAUSES = (
    "create", "merge", "delete", "detach", "set", "remove", "drop",
    "foreach", "load csv", "call db.create", "call apoc.create",
    "call apoc.merge", "call apoc.periodic", "call dbms", "grant", "deny",
    "revoke", "alter", "start database", "stop database", "terminate",
    "use system",
)

ALLOWED_OPENERS = ("match", "with", "unwind", "return", "call {", "profile match",
                   "explain match", "optional match")

MAX_LIMIT = 200
_LIMIT_RE = re.compile(r"\blimit\s+(\d+)\b", re.IGNORECASE)
_STRING_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


class UnsafeCypherError(ValueError):
    """Generated Cypher failed validation and was not executed."""


@dataclass(frozen=True, slots=True)
class GuardResult:
    query: str
    applied_limit: int | None = None
    notes: tuple[str, ...] = ()


def _strip_literals(query: str) -> str:
    """Remove comments and string literals before keyword scanning.

    Without this, a transcript quote containing the word "delete" would trip
    the validator, and a keyword hidden inside a string would slip past it.
    """
    without_comments = _COMMENT_RE.sub(" ", query)
    return _STRING_RE.sub("''", without_comments)


def validate_read_only(query: str, *, max_limit: int = MAX_LIMIT) -> GuardResult:
    """Validate and normalise a generated read query.

    Raises:
        UnsafeCypherError: if the statement is empty, multi-statement, opens
            with something other than a read clause, or contains a write
            clause outside a string literal.
    """
    if not query or not query.strip():
        raise UnsafeCypherError("empty query")

    stripped = query.strip().rstrip(";").strip()
    scannable = _strip_literals(stripped)
    notes: list[str] = []

    # Single statement only — a trailing statement after a semicolon is the
    # classic way to smuggle a write past a prefix check.
    if ";" in scannable.rstrip(";"):
        raise UnsafeCypherError("multiple statements are not allowed")

    lowered = scannable.lower().lstrip("(").lstrip()
    if not any(lowered.startswith(opener) for opener in ALLOWED_OPENERS):
        raise UnsafeCypherError(
            f"query must start with a read clause, got: {stripped[:60]!r}"
        )

    for clause in FORBIDDEN_CLAUSES:
        if re.search(rf"(?<![\w.]){re.escape(clause)}(?![\w])", lowered):
            raise UnsafeCypherError(f"forbidden clause in generated Cypher: {clause!r}")

    # A missing or oversized LIMIT is a resource problem, not a safety one, so
    # it is corrected rather than rejected.
    applied_limit: int | None = None
    match = _LIMIT_RE.search(scannable)
    if match is None:
        stripped = f"{stripped}\nLIMIT {max_limit}"
        applied_limit = max_limit
        notes.append(f"added LIMIT {max_limit}")
    else:
        requested = int(match.group(1))
        if requested > max_limit:
            stripped = _LIMIT_RE.sub(f"LIMIT {max_limit}", stripped, count=1)
            applied_limit = max_limit
            notes.append(f"clamped LIMIT {requested} -> {max_limit}")
        else:
            applied_limit = requested

    return GuardResult(query=stripped, applied_limit=applied_limit, notes=tuple(notes))
