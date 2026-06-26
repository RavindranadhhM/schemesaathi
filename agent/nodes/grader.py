
RELEVANCE_THRESHOLD = 0.55

def run(state: dict) -> dict:
    chunks = state.get("retrieved_chunks", [])
    kept = [c for c in chunks if c.score >= RELEVANCE_THRESHOLD]
    if not kept:
        kept = sorted(chunks, key=lambda c: c.score, reverse=True)[:3]
    kept = sorted(kept, key=lambda c: c.score, reverse=True)[:8]
    return {**state, "reranked_chunks": kept, "grader_scores": [c.score for c in kept],
            "latency_tier": "standard"}
