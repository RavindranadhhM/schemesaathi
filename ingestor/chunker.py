# ingestor/chunker.py
"""
Hierarchical chunker — 3 levels per scheme:
  summary     ~200 tokens   parent chunk, broad retrieval
  section     ~400 tokens   per-field chunks with chunk_type metadata
  fact        ~80 tokens    sentences containing numbers/amounts

3,400 schemes × ~5 chunks = ~17,000 chunks expected.
"""
import re
import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def _slug_id(name: str, suffix: str = "") -> str:
    s = name.lower()
    for ch in ' /(),.&\'"':
        s = s.replace(ch, "-")
    s = "-".join(p for p in s.split("-") if p)[:70]
    return f"{s}-{suffix}" if suffix else s


def _facts(text: str) -> list[str]:
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        sent = sent.strip()
        if re.search(r"\d+\s*(?:lakh|crore|rupee|rs\.?|₹|inr|year|month|%|percent|age)", sent, re.I):
            if 20 < len(sent) < 300:
                out.append(sent)
    return out[:4]


def _meta(s: dict, chunk_type: str, chunk_id: str, parent_id) -> dict:
    # Parse tags into a list
    tags = [t.strip() for t in s.get("tags","").split(",") if t.strip()]
    # Parse categories
    cats = [c.strip() for c in s.get("category","").split(",") if c.strip()]

    return {
        "scheme_id":          _slug_id(s["scheme_name"]),
        "scheme_name":        s["scheme_name"],
        "slug":               s.get("slug",""),
        "chunk_type":         chunk_type,
        "chunk_id":           chunk_id,
        "parent_id":          parent_id,
        # Pre-filled from CSV — no LLM enrichment needed for these
        "level":              s.get("level",""),          # Central / State
        "category":           cats,
        "tags":               tags,
        # Placeholders for Gemini enricher (run later, optional)
        "ministry":           "",
        "beneficiary_types":  [],
        "eligible_states":    ["all"],
        "gender":             ["all"],
        "age_min":            None,
        "age_max":            None,
        "income_ceiling_inr": None,
        "caste_eligibility":  ["all"],
        "benefit_amount":     None,
        "documents_required": [],
        "application_mode":   ["online"],
        "source_url":         f"https://myscheme.gov.in/schemes/{s.get('slug','')}",
        "embedding_model":    "BAAI/bge-m3",
        "last_verified":      "2025-01",
        "language":           "en",
    }


def chunk_scheme(s: dict) -> list[dict]:
    chunks = []
    name      = s["scheme_name"]
    summary_id = _slug_id(name, "summary")

    # ── SUMMARY ───────────────────────────────────────────────────────
    overview      = s.get("details","")[:400]
    elig_preview  = s.get("eligibility","")[:150]
    benefit_preview = s.get("benefits","")[:150]
    summary_text = (
        f"Scheme: {name}\n"
        f"Category: {s.get('category','')} | Level: {s.get('level','')} | Tags: {s.get('tags','')}\n"
        f"Overview: {overview}\n"
        f"Benefits preview: {benefit_preview}\n"
        f"Eligibility preview: {elig_preview}"
    ).strip()

    chunks.append({
        "id":       summary_id,
        "text":     summary_text,
        "metadata": _meta(s, "summary", summary_id, None),
    })

    # ── SECTION CHUNKS ────────────────────────────────────────────────
    sections = [
        ("eligibility", "Eligibility Criteria",  s.get("eligibility","")),
        ("benefits",    "Benefits",               s.get("benefits","")),
        ("documents",   "Documents Required",     s.get("documents","")),
        ("application", "How To Apply",           s.get("application","")),
        ("details",     "Scheme Details",         s.get("details","")),
    ]

    for field, label, content in sections:
        if len(content) < 30:
            continue
        chunk_id = _slug_id(name, field)
        text = f"Scheme: {name}\n{label}:\n{content[:1000]}"
        chunks.append({
            "id":       chunk_id,
            "text":     text,
            "metadata": _meta(s, field, chunk_id, summary_id),
        })

        # ── FACT CHUNKS ───────────────────────────────────────────────
        for j, fact in enumerate(_facts(content)):
            fid = _slug_id(name, f"fact-{field}-{j}")
            chunks.append({
                "id":       fid,
                "text":     f"Scheme: {name}\nFact: {fact}",
                "metadata": _meta(s, "fact", fid, chunk_id),
            })

    return chunks


def chunk_all(save: bool = True) -> list[dict]:
    clean = PROCESSED_DIR / "schemes_clean.json"
    src   = clean if clean.exists() else PROCESSED_DIR / "schemes_parsed.json"
    print(f"Source: {src.name}")

    with open(src, encoding="utf-8") as f:
        schemes = json.load(f)

    all_chunks = []
    for s in schemes:
        all_chunks.extend(chunk_scheme(s))

    by_type: dict[str,int] = {}
    for c in all_chunks:
        t = c["metadata"]["chunk_type"]
        by_type[t] = by_type.get(t, 0) + 1

    print(f"Total chunks : {len(all_chunks)} from {len(schemes)} schemes")
    for t, n in sorted(by_type.items()):
        print(f"  {t:12}: {n}")

    if save:
        out = PROCESSED_DIR / "chunks.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved → {out}")

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all()
    # Show one clean sample of each type
    for t in ["summary","eligibility","benefits","fact"]:
        c = next((x for x in chunks if x["metadata"]["chunk_type"] == t), None)
        if c:
            print(f"\n── {t.upper()} SAMPLE ──")
            print(c["text"][:300])
