from agent.prompts import OUT_OF_SCOPE

SCHEME_KEYWORDS = [
    "scheme","yojana","benefit","eligible","eligibility","apply","application",
    "government","sarkar","subsidy","pension","scholarship","documents","welfare",
    "relief","pm ","pradhan mantri","central","state","ministry","grant","allowance",
    "ration","bpl","apl","obc","sc","st","farmer","kisan","student","widow",
    "disability","housing","health","insurance","loan","training","skill",
    "employment","food","education","girl","women","child","senior","aged",
]

def run(state: dict) -> dict:
    query = state["raw_query"].lower()
    if any(kw in query for kw in SCHEME_KEYWORDS):
        return {**state, "is_in_scope": True}
    if len(query.split()) <= 4:
        return {**state, "is_in_scope": True}
    return {
        **state,
        "is_in_scope":       False,
        "response":          OUT_OF_SCOPE,
        "matched_schemes":   [],
        "validation_passed": True,
    }
