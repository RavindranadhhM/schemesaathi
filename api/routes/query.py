# api/routes/query.py
import uuid
from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from api.models import QueryRequest, QueryResponse
from agent.graph import run_query

router  = APIRouter()
limiter = Limiter(key_func=get_remote_address)

_sessions: dict[str, dict] = {}

@router.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query_schemes(request: Request, body: QueryRequest):
    session_id    = body.session_id or str(uuid.uuid4())
    session_state = _sessions.get(session_id, {})

    result = run_query(body.query, session_state)

    _sessions[session_id] = {
        "session_history": result.get("session_history", []),
        "session_summary": result.get("session_summary", ""),
        "user_profile":    result.get("user_profile"),
    }

    return QueryResponse(
        response        = result["response"],
        refined_output  = result.get("refined_output"),
        disclaimer      = result.get("disclaimer", "Verify eligibility at myscheme.gov.in before applying."),
        matched_schemes = result.get("matched_schemes", []),
        citations       = result.get("citations", []),
        cache_hit       = result.get("cache_hit", False),
        latency_tier    = result.get("latency_tier", "standard"),
        session_id      = session_id,
    )
