"""Semantic search over narration.

The graph answers "when" and "how many". This answers "what was said about it",
which is the half of the question space that structure cannot reach.

Two implementations behind one Protocol. `ChromaVectorStore` is the real path.
`InMemoryVectorStore` is a lexical TF-IDF cosine index with no dependencies —
it is not a semantic index and does not pretend to be, but it returns sensibly
ranked results for the demo and makes the retrieval path testable without
downloading a 90 MB embedding model in CI.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from svqa.types import Evidence, TranscriptSegment

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Words too common in OR narration to carry retrieval signal.
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "had", "has", "have", "here", "i", "if", "in", "into", "is", "it", "its", "just", "like", "me", "my", "no", "not", "of", "on", "or", "our", "so", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "too", "was", "we", "were", "what", "when", "where", "which", "who", "will", "with", "you", "your", "ok", "okay", "right", "now", "see", "look", "going", "get", "got", "let"]
)


# Longest first, so "clipping" loses "ing" before "s" is considered.
_SUFFIXES = ("ings", "ing", "ers", "er", "ed", "es", "s")


def stem(token: str) -> str:
    """Crude suffix stripping.

    Not linguistically principled, and deliberately so — its whole job is to
    collapse the instrument-name variants that appear in narration but not in
    the class labels: "clipper" and "clipping" both reduce to "clipp",
    "grasper" and "grasping" to "grasp". A real embedding index handles this
    semantically and does not need it; this keeps the dependency-free fallback
    from missing the obvious matches.
    """
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [
        stem(token)
        for token in _TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


@runtime_checkable
class VectorStore(Protocol):
    def index_segments(
        self, video_id: str, segments: Sequence[TranscriptSegment]
    ) -> int: ...

    def search(
        self, query: str, *, video_id: str | None = None, k: int = 5
    ) -> list[Evidence]: ...

    def delete_video(self, video_id: str) -> None: ...


class ChromaVectorStore:
    """Persistent Chroma collection, embedded with sentence-transformers."""

    def __init__(
        self,
        path: str | Path,
        collection: str = "transcript_segments",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        import chromadb
        from chromadb.utils import embedding_functions

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            embedding_function=self._embedder,
            # Cosine, not the L2 default: transcript chunks vary a lot in
            # length and L2 quietly favours the short ones.
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("chroma ready at %s (collection=%s)", path, collection)

    def index_segments(
        self, video_id: str, segments: Sequence[TranscriptSegment]
    ) -> int:
        if not segments:
            return 0
        self.delete_video(video_id)  # re-index replaces
        ids, documents, metadatas = [], [], []
        for i, segment in enumerate(segments):
            if not segment.text.strip():
                continue
            ids.append(f"{video_id}:sg{i:04d}")
            documents.append(segment.text)
            metadatas.append(
                {
                    "video_id": video_id,
                    "start_s": float(segment.start_s),
                    "end_s": float(segment.end_s),
                    "speaker": segment.speaker or "",
                }
            )
        if not ids:
            return 0
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def search(
        self, query: str, *, video_id: str | None = None, k: int = 5
    ) -> list[Evidence]:
        where = {"video_id": video_id} if video_id else None
        result = self._collection.query(
            query_texts=[query], n_results=k, where=where
        )
        return _chroma_to_evidence(result)

    def delete_video(self, video_id: str) -> None:
        try:
            self._collection.delete(where={"video_id": video_id})
        except Exception:  # noqa: BLE001 - deleting a missing video is fine
            logger.debug("delete_video no-op for %s", video_id, exc_info=True)


def _chroma_to_evidence(result: dict[str, Any]) -> list[Evidence]:
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    evidence: list[Evidence] = []
    for document, metadata, distance in zip(
        documents, metadatas, distances, strict=False
    ):
        metadata = metadata or {}
        evidence.append(
            Evidence(
                source="transcript",
                content=document,
                start_s=metadata.get("start_s"),
                end_s=metadata.get("end_s"),
                metadata={
                    "score": round(1.0 - float(distance), 4),
                    "speaker": metadata.get("speaker", ""),
                },
            )
        )
    return evidence


class InMemoryVectorStore:
    """Lexical TF-IDF cosine index. No model download, fully deterministic."""

    def __init__(self) -> None:
        self._docs: dict[str, list[tuple[TranscriptSegment, Counter[str]]]] = {}
        self._df: dict[str, Counter[str]] = {}

    def index_segments(
        self, video_id: str, segments: Sequence[TranscriptSegment]
    ) -> int:
        entries: list[tuple[TranscriptSegment, Counter[str]]] = []
        document_frequency: Counter[str] = Counter()
        for segment in segments:
            tokens = tokenize(segment.text)
            if not tokens:
                continue
            counts = Counter(tokens)
            entries.append((segment, counts))
            document_frequency.update(set(counts))
        self._docs[video_id] = entries
        self._df[video_id] = document_frequency
        return len(entries)

    def search(
        self, query: str, *, video_id: str | None = None, k: int = 5
    ) -> list[Evidence]:
        video_ids = [video_id] if video_id else list(self._docs)
        query_tokens = Counter(tokenize(query))
        if not query_tokens:
            return []

        scored: list[tuple[float, TranscriptSegment]] = []
        for vid in video_ids:
            entries = self._docs.get(vid, [])
            if not entries:
                continue
            total_docs = len(entries)
            document_frequency = self._df.get(vid, Counter())
            query_vector = {
                token: count * self._idf(token, document_frequency, total_docs)
                for token, count in query_tokens.items()
            }
            query_norm = math.sqrt(sum(v * v for v in query_vector.values())) or 1.0

            for segment, counts in entries:
                doc_vector = {
                    token: count * self._idf(token, document_frequency, total_docs)
                    for token, count in counts.items()
                }
                doc_norm = math.sqrt(sum(v * v for v in doc_vector.values())) or 1.0
                overlap = set(query_vector) & set(doc_vector)
                if not overlap:
                    continue
                dot = sum(query_vector[t] * doc_vector[t] for t in overlap)
                scored.append((dot / (query_norm * doc_norm), segment))

        scored.sort(key=lambda pair: (-pair[0], pair[1].start_s))
        return [
            Evidence(
                source="transcript",
                content=segment.text,
                start_s=segment.start_s,
                end_s=segment.end_s,
                metadata={"score": round(score, 4), "speaker": segment.speaker or ""},
            )
            for score, segment in scored[:k]
        ]

    @staticmethod
    def _idf(token: str, document_frequency: Counter[str], total_docs: int) -> float:
        # Smoothed IDF; keeps a term that appears in every segment from
        # dominating purely because narration repeats it.
        return math.log((total_docs + 1) / (document_frequency.get(token, 0) + 1)) + 1.0

    def delete_video(self, video_id: str) -> None:
        self._docs.pop(video_id, None)
        self._df.pop(video_id, None)


def build_vector_store(settings: Any) -> VectorStore:
    if settings.backend == "stub":
        logger.info("using InMemoryVectorStore (backend=stub)")
        return InMemoryVectorStore()
    return ChromaVectorStore(
        settings.chroma_path,
        collection=settings.chroma_collection,
        embedding_model=settings.embedding_model,
    )
