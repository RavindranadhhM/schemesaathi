# ingestor/metadata_enricher.py
"""
Uses Gemini 2.5 Flash-Lite to extract structured metadata from parsed scheme text.
Runs ONCE offline. Results saved to disk. Never runs again unless you delete the cache.

Rate limiting: Flash-Lite gives 15 RPM, 1000 RPD free.
At 1927 schemes, this takes ~130 minutes at 15 RPM.
We batch in groups and add sleep to stay under limits.
"""
import json
import time
import asyncio
from pathlib import Path
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ENRICHED_PATH = PROCESSED_DIR / "schemes_enriched.json"

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17")

EXTRACTION_PROMPT = """You are a structured data extractor for Indian government schemes.
Extract the following fields from the scheme information below.
Respond ONLY with valid JSON. No explanation, no markdown, no backticks.

Scheme Name: {scheme_name}
Eligibility Text: {eligibility}
Benefits Text: {benefits}
Documents Text: {documents}

Extract these fields:
{{
  "ministry": "string - ministry name, or 'Unknown'",
  "category": ["list", "of", "categories from: agriculture, education, health, housing, finance, women, sc_st_obc, skill, pension, insurance, scholarship, other"],
  "benefit_type": "one of: cash_transfer, scholarship, loan, subsidy, insurance, pension, food_grain, housing, skill_training, healthcare, other",
  "beneficiary_types": ["list from: farmer, student, woman, sc, st, obc, minority, disabled, senior_citizen, bpl, unemployed, entrepreneur, other"],
  "eligible_states": ["all"] or ["State1", "State2"] if state-specific,
  "gender": ["all"] or subset of ["male", "female", "transgender"],
  "age_min": null or integer,
  "age_max": null or integer,
  "income_ceiling_inr": null or integer (annual, in INR),
  "caste_eligibility": ["all"] or subset of ["general", "obc", "sc", "st"],
  "benefit_amount": "string like '6000/year' or '2 lakh' or null",
  "documents_required": ["aadhaar", "bank_passbook", etc],
  "application_mode": ["online"] or ["offline"] or ["online", "csc"] etc
}}"""


def extract_metadata(scheme: dict) -> Optional[dict]:
    """Extract structured metadata for one scheme using Gemini Flash-Lite."""
    prompt = EXTRACTION_PROMPT.format(
        scheme_name=scheme.get("scheme_name", ""),
        eligibility=scheme.get("Eligibility Of This Scheme", "")[:500],
        benefits=scheme.get("Benefits Of This Scheme", "")[:300],
        documents=scheme.get("Documents Required", "")[:300],
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,           # Deterministic extraction
                max_output_tokens=512,
            )
        )
        text = response.text.strip()

        # Strip any accidental markdown fences
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    except json.JSONDecodeError:
        print(f"    JSON parse failed for: {scheme['scheme_name'][:50]}")
        return None
    except Exception as e:
        print(f"    Error for {scheme['scheme_name'][:50]}: {e}")
        return None


def enrich_all(resume_from: int = 0):
    """
    Enrich all parsed schemes with structured metadata.
    Saves incrementally — safe to interrupt and resume.

    Args:
        resume_from: index to resume from if interrupted
    """
    # Load parsed schemes
    with open(PROCESSED_DIR / "schemes_parsed.json", encoding="utf-8") as f:
        schemes = json.load(f)

    # Load existing enriched if resuming
    enriched = []
    if ENRICHED_PATH.exists() and resume_from > 0:
        with open(ENRICHED_PATH, encoding="utf-8") as f:
            enriched = json.load(f)
        print(f"Resuming from index {resume_from}, already have {len(enriched)} enriched")

    total = len(schemes)
    print(f"Enriching {total - resume_from} schemes with Gemini Flash-Lite...")
    print("Rate limit: 15 RPM → ~130 min for full dataset")
    print("Safe to Ctrl+C and resume with resume_from=<last index>\n")

    for i, scheme in enumerate(schemes[resume_from:], start=resume_from):
        print(f"  [{i+1}/{total}] {scheme['scheme_name'][:60]}")

        metadata = extract_metadata(scheme)

        enriched_scheme = {
            **scheme,
            "extracted_metadata": metadata or {},
            "enriched": metadata is not None,
        }
        enriched.append(enriched_scheme)

        # Save every 10 schemes (incremental checkpoint)
        if (i + 1) % 10 == 0:
            with open(ENRICHED_PATH, "w", encoding="utf-8") as f:
                json.dump(enriched, f, ensure_ascii=False, indent=2)
            print(f"    Checkpoint saved ({i+1} done)")

        # Rate limiting: 15 RPM = 1 request per 4 seconds
        # Add small buffer to be safe
        time.sleep(4.1)

    # Final save
    with open(ENRICHED_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    success = sum(1 for s in enriched if s.get("enriched"))
    print(f"\nDone. {success}/{total} schemes enriched successfully.")
    print(f"Saved → {ENRICHED_PATH}")


if __name__ == "__main__":
    # Test on 3 schemes first before running full job
    import sys
    if "--test" in sys.argv:
        with open(PROCESSED_DIR / "schemes_parsed.json", encoding="utf-8") as f:
            schemes = json.load(f)
        for scheme in schemes[:3]:
            print(f"\nScheme: {scheme['scheme_name']}")
            meta = extract_metadata(scheme)
            print(json.dumps(meta, indent=2))
            time.sleep(4)
    else:
        # Full run — takes ~130 min
        # To resume after interruption: enrich_all(resume_from=200)
        enrich_all(resume_from=0)