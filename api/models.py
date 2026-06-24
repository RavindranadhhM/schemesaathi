from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    session_id: Optional[str] = None

class SchemeResult(BaseModel):
    scheme_name: str
    slug: str = ""
    level: str = ""
    source_url: str = ""

class QueryResponse(BaseModel):
    response: str
    refined_output: Optional[list[dict]] = None
    disclaimer: str = ""
    matched_schemes: list[SchemeResult] = []
    citations: list[dict] = []
    cache_hit: bool = False
    latency_tier: str = "standard"
    session_id: Optional[str] = None
