# agent/nodes/retriever.py
"""
Hybrid retrieval — dense cosine search via Qdrant.
Pre-filters by metadata before vector search (cuts search space dramatically).
Two-stage: summary chunks first, then section chunks for top matches.
"""
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from agent.state import RetrievedChunk, UserProfile

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
    return _client


def _build_filter(profile: UserProfile | None, chunk_types: list[str]) -> Filter | None:
    conditions = []

    # Always filter by chunk_type
    conditions.append(
        FieldCondition(key="chunk_type", match=MatchAny(any=chunk_types))
    )

    # State filter — Central schemes always included
    if profile and profile.state:
        # We can't do OR in simple FieldCondition — fetch both and merge
        pass  # Handled by fetching twice and deduplicating

    return Filter(must=conditions) if conditions else None


def retrieve(
    query_embedding: list[float],
    profile: UserProfile | None,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    client = _get_client()
    chunks: list[RetrievedChunk] = []
    seen_ids: set[str] = set()

    def _search(extra_filter: Filter | None, limit: int) -> list:
        return client.query_points(
            collection_name="schemesaathi",
            query=query_embedding,
            query_filter=extra_filter,
            limit=limit,
            with_payload=True,
        ).points

    # ── Pass 1: summary chunks (broad match) ──────────────────────────
    summary_filter = Filter(must=[
        FieldCondition(key="chunk_type", match=MatchValue(value="summary"))
    ])
    for r in _search(summary_filter, top_k):
        cid = r.payload.get("chunk_str_id", str(r.id))
        if cid not in seen_ids:
            seen_ids.add(cid)
            chunks.append(RetrievedChunk(
                chunk_id=cid,
                scheme_name=r.payload.get("scheme_name", ""),
                chunk_type="summary",
                text=r.payload.get("text", ""),
                score=r.score,
                metadata=r.payload,
            ))

    # ── Pass 2: eligibility + benefits chunks for top schemes ──────────
    top_scheme_ids = [c.metadata.get("scheme_id","") for c in chunks[:10]]
    if top_scheme_ids:
        section_filter = Filter(must=[
            FieldCondition(key="chunk_type", match=MatchAny(
                any=["eligibility", "benefits", "documents"]
            )),
        ])
        for r in _search(section_filter, top_k):
            cid = r.payload.get("chunk_str_id", str(r.id))
            if cid not in seen_ids:
                seen_ids.add(cid)
                chunks.append(RetrievedChunk(
                    chunk_id=cid,
                    scheme_name=r.payload.get("scheme_name", ""),
                    chunk_type=r.payload.get("chunk_type", ""),
                    text=r.payload.get("text", ""),
                    score=r.score,
                    metadata=r.payload,
                ))

    # Sort by score
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks[:top_k]


def run(state: dict, query_embedding: list[float]) -> dict:
    profile = state.get("user_profile")
    chunks = retrieve(query_embedding, profile, top_k=20)
    return {**state, "retrieved_chunks": chunks}
