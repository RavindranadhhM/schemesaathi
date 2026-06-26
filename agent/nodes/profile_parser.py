import json
import os
import re
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from groq import Groq
from agent.state import UserProfile
from agent.prompts import PROFILE_PARSER

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

CITY_TO_STATE = {
    "bangalore": "Karnataka", "bengaluru": "Karnataka",
    "mysore": "Karnataka", "hubli": "Karnataka", "mangalore": "Karnataka",
    "mumbai": "Maharashtra", "pune": "Maharashtra", "nagpur": "Maharashtra",
    "nashik": "Maharashtra", "aurangabad": "Maharashtra",
    "delhi": "Delhi", "new delhi": "Delhi", "noida": "Uttar Pradesh",
    "gurgaon": "Haryana", "gurugram": "Haryana", "faridabad": "Haryana",
    "chennai": "Tamil Nadu", "coimbatore": "Tamil Nadu", "madurai": "Tamil Nadu",
    "salem": "Tamil Nadu", "tiruchirappalli": "Tamil Nadu",
    "hyderabad": "Telangana", "warangal": "Telangana", "nizamabad": "Telangana",
    "kolkata": "West Bengal", "howrah": "West Bengal", "durgapur": "West Bengal",
    "ahmedabad": "Gujarat", "surat": "Gujarat", "vadodara": "Gujarat",
    "rajkot": "Gujarat", "gandhinagar": "Gujarat",
    "jaipur": "Rajasthan", "jodhpur": "Rajasthan", "udaipur": "Rajasthan",
    "kota": "Rajasthan", "ajmer": "Rajasthan",
    "lucknow": "Uttar Pradesh", "kanpur": "Uttar Pradesh", "varanasi": "Uttar Pradesh",
    "agra": "Uttar Pradesh", "allahabad": "Uttar Pradesh", "prayagraj": "Uttar Pradesh",
    "patna": "Bihar", "gaya": "Bihar", "muzaffarpur": "Bihar",
    "bhopal": "Madhya Pradesh", "indore": "Madhya Pradesh", "jabalpur": "Madhya Pradesh",
    "bhubaneswar": "Odisha", "cuttack": "Odisha", "rourkela": "Odisha",
    "chandigarh": "Punjab", "amritsar": "Punjab", "ludhiana": "Punjab",
    "jalandhar": "Punjab", "patiala": "Punjab",
    "kochi": "Kerala", "thiruvananthapuram": "Kerala", "kozhikode": "Kerala",
    "thrissur": "Kerala", "kollam": "Kerala",
    "guwahati": "Assam", "dibrugarh": "Assam", "silchar": "Assam",
    "ranchi": "Jharkhand", "jamshedpur": "Jharkhand", "dhanbad": "Jharkhand",
    "raipur": "Chhattisgarh", "bilaspur": "Chhattisgarh", "durg": "Chhattisgarh",
    "dehradun": "Uttarakhand", "haridwar": "Uttarakhand", "rishikesh": "Uttarakhand",
    "shimla": "Himachal Pradesh", "manali": "Himachal Pradesh", "dharamshala": "Himachal Pradesh",
    "panaji": "Goa", "margao": "Goa", "vasco": "Goa",
    "imphal": "Manipur", "shillong": "Meghalaya", "aizawl": "Mizoram",
    "agartala": "Tripura", "gangtok": "Sikkim", "itanagar": "Arunachal Pradesh",
    "kohima": "Nagaland", "dispur": "Assam",
    "srinagar": "Jammu and Kashmir", "jammu": "Jammu and Kashmir",
    "leh": "Ladakh", "kargil": "Ladakh",
    "pondicherry": "Puducherry", "puducherry": "Puducherry",
    "port blair": "Andaman and Nicobar",
    "visakhapatnam": "Andhra Pradesh", "vijayawada": "Andhra Pradesh",
    "guntur": "Andhra Pradesh", "tirupati": "Andhra Pradesh",
}

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

    # State — check cities first, then full state names
    for city, state in CITY_TO_STATE.items():
        if city in lower:
            profile.state = state
            break
    if not profile.state:
        for state in INDIAN_STATES:
            if state in lower:
                profile.state = state.title()
                break

    # Caste
    if any(w in lower for w in ["sc ", " sc,", "sc/", "scheduled caste", "dalit"]):
        profile.caste = "sc"
    elif any(w in lower for w in ["st ", " st,", "st/", "scheduled tribe", "tribal", "adivasi"]):
        profile.caste = "st"
    elif "obc" in lower or "backward class" in lower:
        profile.caste = "obc"
    elif "general" in lower or "open category" in lower:
        profile.caste = "general"

    # Gender
    if any(w in lower for w in ["woman","women","female","widow","girl","mother","wife","sister"]):
        profile.gender = "female"
    elif any(w in lower for w in [" man "," male ","farmer","boy","father","husband","brother"]):
        profile.gender = "male"

    # Age
    age_match = re.search(r"\b(\d{1,2})\s*(?:years?\s*old|yr|yrs|age)", lower)
    if age_match:
        profile.age = int(age_match.group(1))

    # Income
    income_match = re.search(r"(?:income|earn|salary|earning)[^\d]*(\d+(?:,\d+)*(?:\.\d+)?)", lower)
    if income_match:
        profile.income_inr = int(income_match.group(1).replace(",",""))

    # Occupation
    if any(w in lower for w in ["farmer","kisan","agriculture","farming","cultivat","grower"]):
        profile.occupation = "farmer"
    elif any(w in lower for w in ["student","studying","school","college","btech","b.tech","engineering","graduation","undergraduate","pursuing"]):
        profile.occupation = "student"
    elif any(w in lower for w in ["unemployed","no job","jobless","looking for work","seeking job"]):
        profile.occupation = "unemployed"
    elif any(w in lower for w in ["self employed","business","entrepreneur","shop","shopkeeper"]):
        profile.occupation = "self_employed"
    elif any(w in lower for w in ["salaried","job","employed","working","service","employee"]):
        profile.occupation = "salaried"
    elif any(w in lower for w in ["widow","widower"]):
        profile.occupation = "widow"

    return profile


def parse_profile(text: str) -> UserProfile:
    profile = _keyword_parse(text)
    has_info = any([profile.state, profile.caste, profile.gender,
                    profile.age, profile.income_inr, profile.occupation])
    if has_info:
        return profile
    # Fallback to Groq for complex queries
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":PROFILE_PARSER.format(message=text)}],
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
        if not profile.state:
            profile.state = existing.state
        if not profile.income_inr:
            profile.income_inr = existing.income_inr
        if not profile.caste:
            profile.caste = existing.caste
        if not profile.gender:
            profile.gender = existing.gender
        if not profile.age:
            profile.age = existing.age
        if not profile.occupation:
            profile.occupation = existing.occupation
    return {**state, "user_profile": profile}
