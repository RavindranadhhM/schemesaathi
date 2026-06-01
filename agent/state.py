# agent/state.py
from typing import TypedDict, Optional
from dataclasses import dataclass, field


@dataclass
class UserProfile:
    state: Optional[str] = None
    income_inr: Optional[int] = None
    caste: Optional[str] = None        # general / obc / sc / st
    gender: Optional[str] = None       # male / female / other
    age: Optional[int] = None
    occupation: Optional[str] = None   # farmer / student / unemployed / etc
    raw_text: str = ""                 # original unparsed profile text


@dataclass
class RetrievedChunk:
    chunk_id: str
    scheme_name: str
    chunk_type: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────
    raw_query: str
    language: str                          # "en" or "hi"
    user_profile: Optional[UserProfile]

    # ── Pipeline flags ─────────────────────────────────────
    cache_hit: bool
    is_in_scope: bool
    latency_tier: str                      # "fast" | "standard" | "deep"

    # ── Retrieval ──────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    grader_scores: list[float]             # relevance score per chunk

    # ── Output ─────────────────────────────────────────────
    response: str
    matched_schemes: list[dict]
    citations: list[dict]
    validation_passed: bool
    error: Optional[str]

    # ── Memory ─────────────────────────────────────────────
    session_history: list[dict]            # [{role, content}]
    session_summary: str                   # compressed history
