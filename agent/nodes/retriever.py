import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from agent.state import RetrievedChunk

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30,
        )
    return _client


def retrieve(query_embedding: list[float], profile, top_k: int = 20) -> list[RetrievedChunk]:
    client = _get_client()
    chunks = []
    seen_ids = set()

    def _search(f, limit):
        return client.query_points(
            collection_name="schemesaathi",
            query=query_embedding,
            query_filter=f,
            limit=limit,
            with_payload=True,
        ).points

    # Pass 1: summary chunks
    sf = Filter(must=[FieldCondition(key="chunk_type", match=MatchValue(value="summary"))])
    for r in _search(sf, top_k):
        cid = r.payload.get("chunk_str_id", str(r.id))
        if cid not in seen_ids:
            seen_ids.add(cid)
            chunks.append(RetrievedChunk(
                chunk_id=cid, scheme_name=r.payload.get("scheme_name",""),
                chunk_type="summary", text=r.payload.get("text",""),
                score=r.score, metadata=r.payload,
            ))

    # Pass 2: eligibility + benefits section chunks
    ef = Filter(must=[FieldCondition(key="chunk_type",
                match=MatchAny(any=["eligibility","benefits","documents"]))])
    for r in _search(ef, top_k):
        cid = r.payload.get("chunk_str_id", str(r.id))
        if cid not in seen_ids:
            seen_ids.add(cid)
            chunks.append(RetrievedChunk(
                chunk_id=cid, scheme_name=r.payload.get("scheme_name",""),
                chunk_type=r.payload.get("chunk_type",""),
                text=r.payload.get("text",""), score=r.score, metadata=r.payload,
            ))

    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks[:top_k]


# Known scheme keywords mapped to name fragments for boosted retrieval
SCHEME_NAME_KEYWORDS = {
    "pm kisan":       "PM Kisan",
        "pradhan mantri kisan": "PM Kisan",
        "pm-kisan":       "PM Kisan",
        "kisan samman nidhi": "PM Kisan",
    "kisan samman":   "PM Kisan",
    "ayushman":       "Ayushman",
    "pm-jay":         "Ayushman",
    "ujjwala":        "Ujjwala",
    "fasal bima":     "Fasal Bima",
    "pmfby":          "Fasal Bima",
    "kaushal vikas":  "Kaushal Vikas",
    "pmkvy":          "Kaushal Vikas",
    "awas yojana":    "Awas",
        "housing scheme":  "Housing",
        "housing for bpl": "Housing",
        "2bhk":            "Double Bedroom",
        "double bedroom":  "Double Bedroom",
    "matru vandana":  "Matru Vandana",
    "mudra":          "MUDRA",
    "jal jeevan":     "Jal Jeevan",
    "swachh bharat":  "Swachh Bharat",
    "beti bachao":    "Beti Bachao",
    "sukanya":        "Sukanya",
    "atal pension":   "Atal Pension",
}


def _boost_by_profile(chunks: list, profile) -> list:
    """Re-score chunks based on profile match. Central schemes always kept."""
    if not profile or not profile.state:
        return chunks

    user_state = profile.state.lower()
    boosted = []
    for c in chunks:
        score = c.score
        level = c.metadata.get("level", "").lower()
        scheme_name = c.scheme_name.lower()

        if level == "central":
            score += 0.05          # slight boost — applies to everyone
        elif user_state in scheme_name or user_state in str(c.metadata.get("eligible_states",[])).lower():
            score += 0.15          # strong boost for matching state
        elif level == "state":
            score -= 0.10          # penalise other-state schemes

        c.score = min(1.0, max(0.0, score))
        boosted.append(c)

    return sorted(boosted, key=lambda x: x.score, reverse=True)


def run(state: dict, query_embedding: list[float]) -> dict:
    profile = state.get("user_profile")
    query   = state["raw_query"].lower()

    # Boost known scheme names by putting them first
    name_boost = []
    for kw, name_fragment in SCHEME_NAME_KEYWORDS.items():
        if kw in query:
            # Use semantic search but filter to matching scheme names
            # This avoids expensive scroll — just does a targeted vector search
            semantic = retrieve(query_embedding, profile, top_k=20)
            name_boost = [
                c for c in semantic
                if name_fragment.lower() in c.scheme_name.lower()
            ]
            # Boost their scores
            for c in name_boost:
                c.score = min(1.0, c.score + 0.3)
            break

    semantic_chunks = retrieve(query_embedding, profile, top_k=20)

    # Merge: name matches first, then semantic
    seen = set()
    merged = []
    for c in name_boost:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            merged.append(c)
    for c in semantic_chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            merged.append(c)

    merged.sort(key=lambda c: c.score, reverse=True)
    # Apply profile-based re-scoring
    merged = _boost_by_profile(merged, state.get("user_profile"))
    return {**state, "retrieved_chunks": merged[:25]}
