import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from groq import Groq
from agent.prompts import SUMMARIZER

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"  # Fast small model for summarisation
MAX_RAW_TURNS = 5

def compress_history(history: list[dict]) -> str:
    if not history: return ""
    try:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}" for m in history[-10:]
        )
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": SUMMARIZER.format(history=history_text)}],
            temperature=0.0, max_tokens=150,
        )
        return resp.choices[0].message.content.strip()
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
