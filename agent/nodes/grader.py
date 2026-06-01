import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from google import genai
from google.genai import types
from agent.state import RetrievedChunk
from agent.prompts import GRADER

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
RELEVANCE_THRESHOLD = 0.5

def grade_chunk(query: str, chunk: RetrievedChunk) -> float:
    try:
        resp = _client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            contents=GRADER.format(query=query, chunk=chunk.text[:400]),
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
        )
        return float(resp.text.strip())
    except Exception:
        return chunk.score

def run(state: dict) -> dict:
    query  = state["raw_query"]
    chunks = state.get("retrieved_chunks", [])
    to_grade, ungraded = chunks[:10], chunks[10:]
    scores, kept = [], []
    for chunk in to_grade:
        score = grade_chunk(query, chunk)
        scores.append(score)
        if score >= RELEVANCE_THRESHOLD:
            kept.append(chunk)
    kept.extend(ungraded)
    if not kept:
        kept = sorted(chunks, key=lambda c: c.score, reverse=True)[:3]
    return {**state, "reranked_chunks": kept[:10], "grader_scores": scores,
            "latency_tier": "standard" if len(kept) >= 3 else "deep"}
