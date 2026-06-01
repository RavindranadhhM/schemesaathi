import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from google import genai
from google.genai import types
from agent.prompts import SCOPE_GATE, OUT_OF_SCOPE

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def run(state: dict) -> dict:
    query = state["raw_query"]
    scheme_keywords = ["scheme","yojana","benefit","eligible","eligibility","apply",
        "application","government","sarkar","subsidy","pension","scholarship",
        "documents","welfare","relief","pm ","pradhan mantri","central","state scheme"]
    if any(kw in query.lower() for kw in scheme_keywords):
        return {**state, "is_in_scope": True}
    try:
        resp = _client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            contents=SCOPE_GATE.format(query=query),
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
        )
        in_scope = resp.text.strip().upper().startswith("YES")
    except Exception:
        in_scope = True
    if not in_scope:
        return {**state, "is_in_scope": False, "response": OUT_OF_SCOPE,
                "matched_schemes": [], "validation_passed": True}
    return {**state, "is_in_scope": True}
