# agent/nodes/refinery.py
import re


def _clean(text: str) -> str:
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"\(From [^)]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(Scheme: [^)]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(comes from [^)]+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(Eligibility Criteria|Eligibility|Benefits?|Note|Action)[:\-]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"(\w)(in )([A-Z])", r"\1 in \3", text)  # fix "Stationsin Karnataka"
    return text.strip().rstrip(".,")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _match(s: str, patterns: list[str]) -> bool:
    return any(re.search(p, s, re.IGNORECASE) for p in patterns)


BENEFIT_P = [
    r"rs\.?\s*\d", r"₹", r"lakh", r"crore", r"provide[sd]?",
    r"benefit", r"assist", r"support", r"loan", r"subsidy",
    r"pension", r"scholarship", r"stipend", r"grant", r"financial",
    r"amount", r"facilitating", r"exposure",
]
ELIG_P = [
    r"eligib", r"criteria", r"must be", r"\bage\b", r"\bincome\b",
    r"bpl", r"sc.{0,10}category", r"st.{0,10}category", r"obc",
    r"registered", r"applicant", r"who can", r"land holding",
    r"land record", r"annual income", r"below poverty", r"sc/st",
]
ACTION_P = [
    r"apply", r"visit", r"register", r"submit",
    r"online", r"portal", r"website", r"csc", r"department",
]


def _extract_fields(body: str) -> tuple[str, str, str]:
    # First try to find labelled sections: "- Benefits:", "- Eligibility Criteria:"
    benefit     = _extract_labelled(body, r"-\s*Benefits?[:\-]\s*(.+?)(?=\s*-\s*(?:Eligibility|How to|Apply|$))")
    eligibility = _extract_labelled(body, r"-\s*Eligibility(?:\s*Criteria)?[:\-]\s*(.+?)(?=\s*-\s*\w|\s*Please|\s*Note|$)")
    action      = _extract_labelled(body, r"-\s*(?:How to Apply|Application Process|Apply)[:\-]\s*([^-]+?)(?=\s*-\s*\w|$)")

    # Fallback to sentence-level extraction
    sents = _sentences(body)
    used = set()

    if not benefit:
        for i, s in enumerate(sents):
            if _match(s, BENEFIT_P):
                benefit = _clean(s)
                used.add(i)
                break

    if not eligibility:
        for i, s in enumerate(sents):
            if i in used:
                continue
            if _match(s, ELIG_P):
                eligibility = _clean(s)
                used.add(i)
                break

    if not action:
        for i, s in enumerate(sents):
            if i in used:
                continue
            if _match(s, ACTION_P):
                action = _clean(s)
                break

    if not benefit and sents:
        benefit = _clean(sents[0])

    return _clean(benefit), _clean(eligibility), _clean(action)


def _extract_labelled(text: str, pattern: str) -> str:
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _parse_blocks(text: str) -> list[dict]:
    # Normalise: collapse multiple spaces but preserve numbered item boundaries
    # Handle both newline-separated AND space-collapsed formats
    # Insert newline before "  1." "  2." etc
    text = re.sub(r"\s{2,}(\d+\.)", r"\n\1", text)

    raw_blocks = re.split(r"\n(?=\d+\.)", text)
    if len(raw_blocks) <= 1:
        raw_blocks = re.split(r"(?m)^(?=\d+\.)", text)

    blocks = []
    for block in raw_blocks:
        block = block.strip()
        if not block or not re.match(r"\d+\.", block):
            continue

        block = re.sub(r"^\d+\.\s*", "", block).strip()

        bold = re.match(r"\*\*([^*]+)\*\*[:\-]?\s*(.*)", block, re.DOTALL)
        if bold:
            name = bold.group(1).strip()
            body = bold.group(2).strip()
        else:
            lines = block.split("\n", 1)
            name = re.sub(r"[:\-].*", "", lines[0]).strip()
            body = lines[1].strip() if len(lines) > 1 else lines[0]

        name = _clean(name)
        if not name or len(name) < 4:
            continue

        benefit, eligibility, action = _extract_fields(body)

        blocks.append({
            "name":        name,
            "benefit":     benefit,
            "eligibility": eligibility,
            "action":      action,
            "source_url":  "https://myscheme.gov.in",
        })

    return blocks


def _extract_disclaimer(text: str) -> str:
    for pattern in [
        r"Please verify [^\n]+\.",
        r"Note that [^\n]+\.",
        r"Please note [^\n]+\.",
        r"verify eligibility [^\n]+\.",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return "Verify eligibility at myscheme.gov.in before applying."


def refine(state: dict) -> dict:
    raw = state.get("response", "")
    matched = state.get("matched_schemes", [])

    url_map = {
        s["scheme_name"].lower()[:40]: s.get("source_url", "")
        for s in matched
    }

    blocks = _parse_blocks(raw)

    for block in blocks:
        name_key = block["name"].lower()[:40]
        for key, url in url_map.items():
            if name_key[:25] in key or key[:25] in name_key:
                block["source_url"] = url
                break

    return {
        **state,
        "refined_output": blocks if blocks else None,
        "disclaimer":     _extract_disclaimer(raw),
    }
