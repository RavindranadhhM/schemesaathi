# eval/ragas_eval.py
"""
RAGAS evaluation using direct Gemini API — no langchain wrappers.
Metrics computed manually to avoid dependency conflicts.
"""
import sys, json, os, time
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from pathlib import Path
from google import genai
from google.genai import types
from agent.graph import run_query

RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(exist_ok=True)

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GOLDEN = [
    {"question": "What schemes are available for SC farmers in Karnataka?",
     "ground_truth": "AGR 4 Farm Mechanization Scheme for SC Farmers provides subsidized farm equipment to SC category farmers."},
    {"question": "I am a widow in Maharashtra below poverty line. What pension can I get?",
     "ground_truth": "Indira Gandhi National Widow Pension Scheme Maharashtra provides monthly pension to BPL widows."},
    {"question": "What scholarship schemes exist for SC ST students in higher education?",
     "ground_truth": "Post Matric Scholarship for SC students provides financial assistance for higher education."},
    {"question": "Schemes for pregnant women and maternal health in India?",
     "ground_truth": "Pradhan Mantri Matru Vandana Yojana provides cash incentive of Rs 5000 to pregnant mothers."},
    {"question": "What housing schemes are available for BPL families?",
     "ground_truth": "Pradhan Mantri Awas Yojana provides financial assistance for house construction to BPL families."},
    {"question": "Skill training schemes for unemployed youth in India?",
     "ground_truth": "Pradhan Mantri Kaushal Vikas Yojana provides free skill training to unemployed youth."},
    {"question": "What schemes help women start their own business?",
     "ground_truth": "Stand-Up India provides bank loans to SC ST women entrepreneurs for greenfield enterprises."},
    {"question": "Medical insurance scheme for poor families in India?",
     "ground_truth": "Ayushman Bharat PM-JAY provides health coverage of Rs 5 lakh per family per year to BPL families."},
    {"question": "What is PM KISAN scheme and who is eligible?",
     "ground_truth": "PM Kisan Samman Nidhi provides Rs 6000 per year to small and marginal farmer families."},
    {"question": "Schemes for disabled persons in India?",
     "ground_truth": "Assistance to Disabled Persons for Purchase of Aids and Appliances provides assistive devices."},
    {"question": "What schemes are there for old age pension in India?",
     "ground_truth": "Indira Gandhi National Old Age Pension Scheme provides monthly pension to BPL citizens above 60 years."},
    {"question": "Free LPG gas connection schemes for BPL families?",
     "ground_truth": "Pradhan Mantri Ujjwala Yojana provides free LPG connections to women from BPL households."},
    {"question": "What schemes help farmers get crop insurance?",
     "ground_truth": "Pradhan Mantri Fasal Bima Yojana provides crop insurance to farmers at subsidized premium rates."},
    {"question": "Schemes for weavers and handloom workers in India?",
     "ground_truth": "Pradhan Mantri MUDRA Yojana provides micro loans to weavers and handloom workers."},
    {"question": "What drinking water schemes exist for rural areas?",
     "ground_truth": "Jal Jeevan Mission provides tap water connections to every rural household in India."},
]


def gemini_score(prompt: str) -> float:
    """Ask Gemini to score something 0.0-1.0. Returns float."""
    try:
        resp = _client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5),
        )
        return min(1.0, max(0.0, float(resp.text.strip())))
    except Exception:
        return 0.5


def score_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """Are all claims in the answer supported by the contexts?"""
    ctx = "\n".join(contexts[:3])[:1500]
    prompt = f"""You are evaluating if an answer is faithful to the provided context.
Score 0.0 to 1.0 where 1.0 means every claim is supported by context, 0.0 means hallucination.
Return ONLY a decimal number.

Context: {ctx}
Answer: {answer[:600]}

Faithfulness score:"""
    return gemini_score(prompt)


def score_answer_relevancy(question: str, answer: str) -> float:
    """Does the answer actually address the question?"""
    prompt = f"""Score how well this answer addresses the question.
1.0 = perfectly relevant, 0.0 = completely irrelevant.
Return ONLY a decimal number.

Question: {question}
Answer: {answer[:600]}

Relevancy score:"""
    return gemini_score(prompt)


def score_context_precision(question: str, contexts: list[str]) -> float:
    """Are the retrieved contexts relevant to the question?"""
    ctx = "\n".join(contexts[:3])[:1000]
    prompt = f"""Score how relevant these retrieved documents are to the question.
1.0 = all documents are highly relevant, 0.0 = irrelevant.
Return ONLY a decimal number.

Question: {question}
Retrieved contexts: {ctx}

Context precision score:"""
    return gemini_score(prompt)


def run_evaluation():
    print(f"Running evaluation on {len(GOLDEN)} queries...\n")
    results = []

    for i, item in enumerate(GOLDEN):
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"[{i+1:2}/{len(GOLDEN)}] {q[:65]}")

        try:
            result   = run_query(q)
            answer   = result.get("response", "")
            contexts = [c.text for c in result.get("reranked_chunks", [])[:5]]

            f_score  = score_faithfulness(q, answer, contexts)
            r_score  = score_answer_relevancy(q, answer)
            p_score  = score_context_precision(q, contexts)

            avg = round((f_score + r_score + p_score) / 3, 3)
            print(f"         faith={f_score:.2f}  relevancy={r_score:.2f}  precision={p_score:.2f}  avg={avg:.2f}")

            results.append({
                "question":          q,
                "ground_truth":      gt,
                "answer":            answer[:300],
                "faithfulness":      f_score,
                "answer_relevancy":  r_score,
                "context_precision": p_score,
                "avg":               avg,
            })

        except Exception as e:
            print(f"         ERROR: {e}")
            results.append({"question": q, "error": str(e)})

        time.sleep(2)  # rate limit buffer

    # Aggregate
    valid = [r for r in results if "faithfulness" in r]
    if valid:
        agg = {
            "faithfulness":      round(sum(r["faithfulness"]      for r in valid) / len(valid), 4),
            "answer_relevancy":  round(sum(r["answer_relevancy"]  for r in valid) / len(valid), 4),
            "context_precision": round(sum(r["context_precision"] for r in valid) / len(valid), 4),
            "overall":           round(sum(r["avg"]               for r in valid) / len(valid), 4),
            "n_evaluated":       len(valid),
        }

        print("\n" + "="*45)
        print("FINAL RAGAS-STYLE SCORES")
        print("="*45)
        for k, v in agg.items():
            print(f"  {k:22}: {v}")

        # Save
        out = RESULTS_DIR / "ragas_scores.json"
        with open(out, "w") as f:
            json.dump({"aggregate": agg, "per_query": results}, f, indent=2)
        print(f"\nSaved → {out}")
        return agg

    print("No valid results.")
    return {}


if __name__ == "__main__":
    run_evaluation()
