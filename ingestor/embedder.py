# ingestor/embedder.py
"""
Embeds all 27,262 chunks using BGE-M3 on Apple Silicon MPS.
Expected time: ~25-35 min on M1 8GB with MPS.
Checkpoints every 20 batches — safe to interrupt and resume.
"""
import json
import time
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CHUNKS_PATH   = PROCESSED_DIR / "chunks.json"
EMBEDDED_PATH = PROCESSED_DIR / "chunks_embedded.json"


def load_model():
    from FlagEmbedding import BGEM3FlagModel
    print("Loading BGE-M3 (downloads ~2.3GB on first run)...")
    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,
        device="mps",
    )
    print("Model ready.")
    return model


def embed_all(batch_size: int = 24):
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        all_chunks = json.load(f)
    print(f"Total chunks to embed: {len(all_chunks)}")

    # Resume from checkpoint
    done_ids: set[str] = set()
    embedded: list[dict] = []
    if EMBEDDED_PATH.exists():
        with open(EMBEDDED_PATH, encoding="utf-8") as f:
            embedded = json.load(f)
        done_ids = {c["id"] for c in embedded if "embedding" in c}
        print(f"Resuming — already done: {len(done_ids)}")

    remaining = [c for c in all_chunks if c["id"] not in done_ids]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All chunks already embedded.")
        return embedded

    model = load_model()
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    start = time.time()

    for batch_num, i in enumerate(range(0, len(remaining), batch_size), 1):
        batch  = remaining[i : i + batch_size]
        texts  = [c["text"] for c in batch]

        try:
            out = model.encode(
                texts,
                batch_size=batch_size,
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vectors = out["dense_vecs"].tolist()

            for chunk, vec in zip(batch, vectors):
                embedded.append({**chunk, "embedding": vec})

        except Exception as e:
            print(f"  Batch {batch_num} error: {e} — skipping")
            time.sleep(2)
            continue

        # Checkpoint + progress
        if batch_num % 20 == 0 or batch_num == total_batches:
            with open(EMBEDDED_PATH, "w", encoding="utf-8") as f:
                json.dump(embedded, f)
            elapsed  = time.time() - start
            rate     = (batch_num * batch_size) / elapsed
            remaining_sec = (len(remaining) - batch_num * batch_size) / max(rate, 1)
            print(
                f"  [{batch_num}/{total_batches}] "
                f"{len(embedded)} embedded | "
                f"{elapsed/60:.1f}m elapsed | "
                f"~{remaining_sec/60:.1f}m left"
            )

    # Final save
    with open(EMBEDDED_PATH, "w", encoding="utf-8") as f:
        json.dump(embedded, f)

    total_time = time.time() - start
    print(f"\nDone. {len(embedded)} chunks embedded in {total_time/60:.1f} min.")
    if embedded:
        print(f"Vector dimensions: {len(embedded[0]['embedding'])}")
    return embedded


if __name__ == "__main__":
    embed_all(batch_size=24)
