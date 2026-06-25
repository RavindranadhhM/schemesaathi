# eval/drift_monitor.py
"""
Weekly embedding drift check.
Re-embeds 50 random chunks and compares to stored vectors.
If average cosine similarity < 0.95, triggers alert.

Run via GitHub Actions every Monday, or manually.
"""
import os, sys, json, random
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
import numpy as np
from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel
from pathlib import Path

THRESHOLD = 0.95
SAMPLE_SIZE = 50

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def run_drift_check():
    print("Loading BGE-M3...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, device="cpu")

    print("Connecting to Qdrant...")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    # Sample random points from Qdrant
    results = client.scroll(
        collection_name="schemesaathi",
        limit=SAMPLE_SIZE,
        with_payload=True,
        with_vectors=True,
    )[0]

    print(f"Sampled {len(results)} chunks")
    similarities = []

    for point in results:
        text = point.payload.get("text", "")
        if not text or len(text) < 20:
            continue

        stored_vec = point.vector
        new_out = model.encode(
            [text[:512]],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        new_vec = new_out["dense_vecs"][0].tolist()
        sim = cosine(stored_vec, new_vec)
        similarities.append(sim)

    avg_sim = float(np.mean(similarities))
    min_sim = float(np.min(similarities))

    print(f"\nDrift check results:")
    print(f"  Samples:    {len(similarities)}")
    print(f"  Avg cosine: {avg_sim:.4f}")
    print(f"  Min cosine: {min_sim:.4f}")
    print(f"  Threshold:  {THRESHOLD}")

    if avg_sim < THRESHOLD:
        print(f"\n⚠ DRIFT DETECTED — avg similarity {avg_sim:.4f} < {THRESHOLD}")
        print("  Recommendation: re-run ingestor/embedder.py")
        sys.exit(1)
    else:
        print(f"\n✓ No drift detected — embeddings are stable")

    # Save results
    out = Path("eval/results/drift_check.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "avg_cosine": avg_sim,
            "min_cosine": min_sim,
            "samples": len(similarities),
            "drift_detected": avg_sim < THRESHOLD,
        }, f, indent=2)

if __name__ == "__main__":
    run_drift_check()
