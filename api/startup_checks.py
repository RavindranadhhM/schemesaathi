# api/startup_checks.py
"""
Startup validation — checks all required secrets are present.
Fails fast with clear error if anything is missing.
"""
import os
import sys

REQUIRED = {
    "GROQ_API_KEY":    "Get from console.groq.com",
    "QDRANT_URL":      "Get from cloud.qdrant.io",
    "QDRANT_API_KEY":  "Get from cloud.qdrant.io",
}

def validate_secrets():
    missing = []
    for key, hint in REQUIRED.items():
        val = os.getenv(key, "").strip()
        if not val or val.startswith('"') or val.startswith("'"):
            missing.append(f"  {key}: {hint}")

    if missing:
        print("STARTUP ERROR — Missing or malformed secrets:")
        for m in missing:
            print(m)
        sys.exit(1)

    print("✓ All secrets validated")
