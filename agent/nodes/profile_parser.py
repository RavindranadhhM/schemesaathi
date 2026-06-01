import json, os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from google import genai
from google.genai import types
from agent.state import UserProfile
from agent.prompts import PROFILE_PARSER

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def parse_profile(text: str) -> UserProfile:
    if not text or len(text.strip()) < 5:
        return UserProfile(raw_text=text)
    try:
        resp = _client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            contents=PROFILE_PARSER.format(message=text),
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=200),
        )
        raw = resp.text.strip().replace("```json","").replace("```","").strip()
        data = json.loads(raw)
        return UserProfile(
            state=data.get("state"), income_inr=data.get("income_inr"),
            caste=data.get("caste"), gender=data.get("gender"),
            age=data.get("age"), occupation=data.get("occupation"), raw_text=text,
        )
    except Exception:
        return UserProfile(raw_text=text)

def run(state: dict) -> dict:
    profile_text = state["raw_query"]
    if state.get("session_summary"):
        profile_text = state["session_summary"] + "\n" + profile_text
    profile = parse_profile(profile_text)
    existing = state.get("user_profile")
    if existing:
        if not profile.state:      profile.state      = existing.state
        if not profile.income_inr: profile.income_inr = existing.income_inr
        if not profile.caste:      profile.caste      = existing.caste
        if not profile.gender:     profile.gender     = existing.gender
        if not profile.age:        profile.age        = existing.age
        if not profile.occupation: profile.occupation = existing.occupation
    return {**state, "user_profile": profile}
