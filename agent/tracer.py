# agent/tracer.py
"""
Langfuse tracing — wraps every agent run with a trace.
Logs: query, profile, retrieved schemes, response, latency, RAGAS scores.
"""
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

_langfuse = None

def get_langfuse():
    global _langfuse
    if _langfuse is None:
        try:
            from langfuse import Langfuse
            _langfuse = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
        except Exception:
            _langfuse = None
    return _langfuse


def trace_query(query: str, result: dict, latency_ms: float):
    """Log a complete query run to Langfuse."""
    lf = get_langfuse()
    if not lf:
        return

    try:
        lf.trace(
            name="schemesaathi-query",
            input={"query": query},
            output={"response": result.get("response", "")[:500]},
            metadata={
                "cache_hit":      result.get("cache_hit", False),
                "latency_tier":   result.get("latency_tier", "standard"),
                "latency_ms":     round(latency_ms),
                "schemes_matched": len(result.get("matched_schemes", [])),
                "chunks_retrieved": len(result.get("retrieved_chunks", [])),
                "profile": {
                    "state":      getattr(result.get("user_profile"), "state", None),
                    "caste":      getattr(result.get("user_profile"), "caste", None),
                    "occupation": getattr(result.get("user_profile"), "occupation", None),
                },
                "top_schemes": [
                    s["scheme_name"] for s in result.get("matched_schemes", [])[:3]
                ],
            },
        )
        lf.flush()
    except Exception:
        pass  # Tracing is never fatal
