import json, os
import numpy as np
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
import redis

_redis_client = None

def _get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=False,
        )
    return _redis_client

def cosine_similarity(a, b) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

def cache_lookup(query_embedding, threshold=0.92):
    try:
        r = _get_redis()
        for key in r.keys("cache:*"):
            data = r.get(key)
            if not data: continue
            entry = json.loads(data)
            if cosine_similarity(query_embedding, entry["embedding"]) >= threshold:
                return entry["response"]
    except Exception:
        pass
    return None

def cache_store(query_embedding, response, ttl=86400):
    try:
        r = _get_redis()
        key = f"cache:{hash(tuple(query_embedding[:10]))}"
        r.setex(key, ttl, json.dumps({"embedding": query_embedding, "response": response}))
    except Exception:
        pass

def run(state: dict, query_embedding) -> dict:
    cached = cache_lookup(query_embedding)
    if cached:
        return {**state, "cache_hit": True, "response": cached.get("response",""),
                "matched_schemes": cached.get("matched_schemes",[]), "latency_tier": "fast"}
    return {**state, "cache_hit": False}
