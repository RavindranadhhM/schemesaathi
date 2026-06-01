# ingestor/cleaner.py
"""
Light cleaner — this dataset is already clean.
Just normalises whitespace and removes residual HTML artifacts.
"""
import re
import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

def clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML tags if any
    text = re.sub(r"\s{2,}", " ", text)             # collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)          # max 2 newlines
    return text.strip()

def clean_all() -> list[dict]:
    with open(PROCESSED_DIR / "schemes_parsed.json", encoding="utf-8") as f:
        schemes = json.load(f)

    cleaned = []
    for s in schemes:
        cleaned.append({
            "scheme_name": clean(s["scheme_name"]),
            "slug":        s.get("slug",""),
            "details":     clean(s.get("details","")),
            "benefits":    clean(s.get("benefits","")),
            "eligibility": clean(s.get("eligibility","")),
            "application": clean(s.get("application","")),
            "documents":   clean(s.get("documents","")),
            "level":       s.get("level",""),
            "category":    s.get("category",""),
            "tags":        s.get("tags",""),
        })

    out = PROCESSED_DIR / "schemes_clean.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"Cleaned {len(cleaned)} schemes → {out}")
    for col in ["details","benefits","eligibility","documents","application"]:
        filled = sum(1 for s in cleaned if len(s.get(col,"")) > 50)
        print(f"  {col:12}: {filled}/{len(cleaned)}")
    return cleaned

if __name__ == "__main__":
    clean_all()
