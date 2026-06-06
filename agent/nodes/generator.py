import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from groq import Groq
from agent.state import RetrievedChunk, UserProfile
from agent.prompts import GENERATOR

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

def _format_profile(profile) -> str:
    if not profile: return "Not provided"
    parts = []
    if profile.state:      parts.append(f"State: {profile.state}")
    if profile.age:        parts.append(f"Age: {profile.age}")
    if profile.gender:     parts.append(f"Gender: {profile.gender}")
    if profile.caste:      parts.append(f"Category: {profile.caste.upper()}")
    if profile.income_inr: parts.append(f"Annual income: \u20b9{profile.income_inr:,}")
    if profile.occupation: parts.append(f"Occupation: {profile.occupation}")
    return ", ".join(parts) if parts else "Not provided"

def _format_context(chunks) -> str:
    by_scheme = {}
    for c in chunks:
        by_scheme.setdefault(c.scheme_name, []).append(f"[{c.chunk_type}] {c.text[:500]}")
    return "\n\n".join(
        f"--- {name} ---\n" + "\n".join(texts)
        for name, texts in list(by_scheme.items())[:8]
    )

def _matched(chunks) -> list:
    seen, schemes = set(), []
    for c in chunks:
        if c.scheme_name not in seen:
            seen.add(c.scheme_name)
            schemes.append({"scheme_name": c.scheme_name, "slug": c.metadata.get("slug",""),
                "level": c.metadata.get("level",""), "category": c.metadata.get("category",[]),
                "source_url": c.metadata.get("source_url","")})
    return schemes[:8]

def run(state: dict) -> dict:
    chunks  = state.get("reranked_chunks", [])
    profile = state.get("user_profile")
    query   = state["raw_query"]
    if not chunks:
        return {**state, "response": "I could not find relevant schemes. Please provide your state, age, income and occupation.",
                "matched_schemes": [], "validation_passed": True}
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": GENERATOR.format(
                profile=_format_profile(profile),
                history=state.get("session_summary") or "No prior conversation",
                context=_format_context(chunks),
                query=query,
            )}],
            temperature=0.3,
            max_tokens=1024,
        )
        response_text = resp.choices[0].message.content.strip()
    except Exception as e:
        response_text = f"Error generating response: {e}"
    return {**state, "response": response_text, "matched_schemes": _matched(chunks),
            "citations": [{"scheme_name": c.scheme_name, "chunk_type": c.chunk_type,
                           "score": round(c.score,3)} for c in chunks[:5]],
            "validation_passed": True}
