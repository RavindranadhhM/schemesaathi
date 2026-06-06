
import sys, json, time
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from pathlib import Path
from agent.graph import run_query

GOLDEN = [
    "What schemes are available for SC farmers in Karnataka?",
    "I am a widow in Maharashtra below poverty line. What pension can I get?",
    "What scholarship schemes exist for SC ST students in higher education?",
    "Schemes for pregnant women and maternal health in India?",
    "What housing schemes are available for BPL families?",
    "Skill training schemes for unemployed youth in India?",
    "What schemes help women start their own business?",
    "Medical insurance scheme for poor families in India?",
    "What is PM KISAN scheme and who is eligible?",
    "Schemes for disabled persons in India?",
    "What schemes are there for old age pension in India?",
    "Free LPG gas connection schemes for BPL families?",
    "What schemes help farmers get crop insurance?",
    "Schemes for weavers and handloom workers in India?",
    "What drinking water schemes exist for rural areas?",
]

GROUND_TRUTHS = [
    "AGR 4 Farm Mechanization Scheme provides subsidized farm equipment to SC farmers.",
    "Indira Gandhi National Widow Pension Scheme provides monthly pension to BPL widows.",
    "Post Matric Scholarship for SC students provides financial assistance for higher education.",
    "Pradhan Mantri Matru Vandana Yojana provides cash incentive of Rs 5000 to pregnant mothers.",
    "Pradhan Mantri Awas Yojana provides financial assistance for house construction to BPL families.",
    "Pradhan Mantri Kaushal Vikas Yojana provides free skill training to unemployed youth.",
    "Stand-Up India provides bank loans to SC ST women entrepreneurs for greenfield enterprises.",
    "Ayushman Bharat PM-JAY provides health coverage of Rs 5 lakh per family per year.",
    "PM Kisan Samman Nidhi provides Rs 6000 per year to small and marginal farmer families.",
    "Assistance to Disabled Persons scheme provides assistive devices to persons with disabilities.",
    "Indira Gandhi National Old Age Pension Scheme provides monthly pension to BPL citizens above 60.",
    "Pradhan Mantri Ujjwala Yojana provides free LPG connections to women from BPL households.",
    "Pradhan Mantri Fasal Bima Yojana provides crop insurance at subsidized premium rates.",
    "Pradhan Mantri MUDRA Yojana provides micro loans to weavers and handloom workers.",
    "Jal Jeevan Mission provides tap water connections to every rural household in India.",
]

Path("eval/results").mkdir(exist_ok=True)
out_path = Path("eval/results/agent_responses.json")

# Load existing
existing = {}
if out_path.exists():
    for r in json.load(open(out_path)):
        existing[r["question"]] = r
print(f"Already have {len(existing)}/15 responses")

responses = list(existing.values())

for i, (q, gt) in enumerate(zip(GOLDEN, GROUND_TRUTHS)):
    if q in existing:
        print(f"[{i+1:2}/15] SKIP: {q[:55]}")
        continue
    print(f"[{i+1:2}/15] {q[:55]}")
    
    # Retry up to 3 times
    for attempt in range(3):
        try:
            result = run_query(q)
            entry = {
                "question":     q,
                "ground_truth": gt,
                "answer":       result.get("response",""),
                "contexts":     [c.text for c in result.get("reranked_chunks",[])[:5]],
                "matched":      [s["scheme_name"] for s in result.get("matched_schemes",[])[:3]],
            }
            responses.append(entry)
            existing[q] = entry
            print(f"        matched: {entry['matched'][:2]}")
            # Save after each success
            with open(out_path,"w") as f:
                json.dump(responses, f, indent=2)
            break
        except Exception as e:
            print(f"        Attempt {attempt+1} failed: {str(e)[:60]}")
            if attempt < 2:
                time.sleep(5 * (attempt+1))  # 5s, 10s backoff
            else:
                print(f"        Giving up on this query")
    
    time.sleep(3)

print(f"\nFinal: {len(responses)}/15 responses saved")
