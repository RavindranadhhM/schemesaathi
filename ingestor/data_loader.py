# ingestor/data_loader.py
import csv
import json
from pathlib import Path

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_and_parse(save: bool = True) -> list[dict]:
    path = RAW_DIR / "updated_data.csv"
    schemes = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scheme = {
                "scheme_name":  row["scheme_name"].strip().strip('"'),
                "slug":         row.get("slug", "").strip(),
                "details":      row.get("details", "").strip(),
                "benefits":     row.get("benefits", "").strip(),
                "eligibility":  row.get("eligibility", "").strip(),
                "application":  row.get("application", "").strip(),
                "documents":    row.get("documents", "").strip(),
                "level":        row.get("level", "").strip(),       # Central / State
                "category":     row.get("schemeCategory", "").strip(),
                "tags":         row.get("tags", "").strip(),
            }
            # Skip if no meaningful content at all
            if len(scheme["details"]) > 20 or len(scheme["benefits"]) > 20:
                schemes.append(scheme)

    print(f"Loaded {len(schemes)} schemes from CSV")
    for col in ["details","benefits","eligibility","documents","application"]:
        filled = sum(1 for s in schemes if len(s.get(col,"")) > 50)
        print(f"  {col:12}: {filled}/{len(schemes)}")

    if save:
        out = PROCESSED_DIR / "schemes_parsed.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(schemes, f, ensure_ascii=False, indent=2)
        print(f"Saved → {out}")

    return schemes


if __name__ == "__main__":
    schemes = load_and_parse()
    print(f"\nSample:")
    s = schemes[0]
    for k in ["scheme_name","level","category","tags","details","benefits","eligibility"]:
        print(f"  {k}: {str(s.get(k,''))[:120]}")
