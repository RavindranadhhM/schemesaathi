import json, os, re
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from groq import Groq
from agent.state import UserProfile
from agent.prompts import PROFILE_PARSER

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

INDIAN_STATES = [
    "andhra pradesh","arunachal pradesh","assam","bihar","chhattisgarh","goa",
    "gujarat","haryana","himachal pradesh","jharkhand","karnataka","kerala",
    "madhya pradesh","maharashtra","manipur","meghalaya","mizoram","nagaland",
    "odisha","punjab","rajasthan","sikkim","tamil nadu","telangana","tripura",
    "uttar pradesh","uttarakhand","west bengal","delhi","jammu","kashmir",
    "puducherry","chandigarh","ladakh","andaman",
]

def _keyword_parse(text: str) -> UserProfile:
    lower = text.lower()
    profile = UserProfile(raw_text=text)
    for state in INDIAN_STATES:
        if state in lower:
            profile.state = state.title()
            break
    if any(w in lower for w in ["sc ", "scheduled caste", "dalit"]):
        profile.caste = "sc"
    elif any(w in lower for w in ["st ", "scheduled tribe", "tribal", "adivasi"]):
        profile.caste = "st"
    elif "obc" in lower or "backward class" in lower:
        profile.caste = "obc"
    elif "general" in lower or "open category" in lower:
        profile.caste = "general"
    if any(w in lower for w in ["woman","women","female","widow","girl","mother"]):
        profile.gender = "female"
    elif any(w in lower for w in ["man","male","farmer","boy","father"]):
        profile.gender = "male"
    age_match = re.search(r"\b(\d{1,2})\s*(?:years?\s*old|yr|age)", lower)
    if age_match:
        profile.age = int(age_match.group(1))
    income_match = re.search(r"(?:income|earn|salary)[^\d]*(\d+(?:,\d+)*)", lower)
    if income_match:
        profile.income_inr = int(income_match.group(1).replace(",",""))
    if any(w in lower for w in ["farmer","kisan","agriculture","farming"]):
        profile.occupation = "farmer"
    elif any(w in lower for w in ["student","studying","school","college"]):
        profile.occupation = "student"
    elif any(w in lower for w in ["unemployed","no job","jobless"]):
        profile.occupation = "unemployed"
    elif any(w in lower for w in ["self employed","business","entrepreneur"]):
        profile.occupation = "self_employed"
    elif any(w in lower for w in ["salaried","job","employed","working"]):
        profile.occupation = "salaried"
    return profile

def parse_profile(text: str) -> UserProfile:
    profile = _keyword_parse(text)
    has_info = any([profile.state, profile.caste, profile.gender,
                    profile.age, profile.income_inr, profile.occupation])
    if has_info:
        return profile
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROFILE_PARSER.format(message=text)}],
            temperature=0.0, max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        data = json.loads(raw)
        return UserProfile(
            state=data.get("state"), income_inr=data.get("income_inr"),
            caste=data.get("caste"), gender=data.get("gender"),
            age=data.get("age"), occupation=data.get("occupation"), raw_text=text,
        )
    except Exception:
        return profile

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
