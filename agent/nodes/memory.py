import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from google import genai
from google.genai import types
from agent.prompts import SUMMARIZER

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MAX_RAW_TURNS = 5

def compress_history(history: list[dict]) -> str:
    if not history: return ""
    try:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}" for m in history[-10:]
        )
        resp = _client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            contents=SUMMARIZER.format(history=history_text),
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=150),
        )
        return resp.text.strip()
    except Exception:
        return ""

def update(state: dict) -> dict:
    history = list(state.get("session_history", []))
    history.append({"role": "user",      "content": state["raw_query"]})
    history.append({"role": "assistant", "content": state.get("response","")})
    summary = state.get("session_summary","")
    if len(history) > MAX_RAW_TURNS * 2:
        summary = compress_history(history)
        history = history[-(MAX_RAW_TURNS * 2):]
    return {**state, "session_history": history, "session_summary": summary}
