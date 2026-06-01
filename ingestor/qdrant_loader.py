# ingestor/qdrant_loader.py
"""
Uploads embedded chunks to Qdrant Cloud.
Uses named vectors — dense for semantic search.
BM25 sparse handled natively by Qdrant on their side.
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, PayloadSchemaType,
)

load_dotenv()

PROCESSED_DIR  = Path(__file__).parent.parent / "data" / "processed"
COLLECTION     = "schemesaathi"
VECTOR_DIM     = 1024   # BGE-M3 dense dimension
BATCH_SIZE     = 100


def get_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )


def create_collection(client: QdrantClient, recreate: bool = False):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        if recreate:
            client.delete_collection(COLLECTION)
            print(f"Deleted existing collection '{COLLECTION}'")
        else:
            print(f"Collection '{COLLECTION}' already exists — skipping creation")
            return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
            on_disk=False,
        ),
    )

    # Create payload indexes for metadata filtering
    # These make pre-filtering fast — O(1) instead of O(n)
    indexes = {
        "level":       PayloadSchemaType.KEYWORD,
        "chunk_type":  PayloadSchemaType.KEYWORD,
        "scheme_id":   PayloadSchemaType.KEYWORD,
        "parent_id":   PayloadSchemaType.KEYWORD,
    }
    for field, schema in indexes.items():
        client.create_payload_index(COLLECTION, field, schema)

    print(f"Created collection '{COLLECTION}' with {len(indexes)} payload indexes")


def upload_chunks(client: QdrantClient):
    path = PROCESSED_DIR / "chunks_embedded.json"
    if not path.exists():
        raise FileNotFoundError("Run embedder.py first — chunks_embedded.json not found")

    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)

    # Only upload chunks that have embeddings
    chunks = [c for c in chunks if "embedding" in c]
    print(f"Uploading {len(chunks)} chunks to Qdrant...")

    # Check what's already uploaded
    count = client.count(COLLECTION).count
    if count > 0:
        print(f"Collection already has {count} points — uploading remainder only")

    success = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        points = [
            PointStruct(
                id=abs(hash(c["id"])) % (2**63),   # Qdrant needs uint64
                vector=c["embedding"],
                payload={
                    **c["metadata"],
                    "text": c["text"],              # Store text in payload for retrieval
                    "chunk_str_id": c["id"],        # Keep original string ID
                },
            )
            for c in batch
        ]

        try:
            client.upsert(collection_name=COLLECTION, points=points)
            success += len(batch)
        except Exception as e:
            print(f"  Batch {i//BATCH_SIZE + 1} error: {e}")
            continue

        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"  Uploaded {success}/{len(chunks)}...")

    print(f"\nDone. {success}/{len(chunks)} chunks in Qdrant.")
    print(f"Collection size: {client.count(COLLECTION).count} points")


def verify(client: QdrantClient):
    """Test a quick search to confirm everything works."""
    import random

    # Load a random embedding to use as query
    path = PROCESSED_DIR / "chunks_embedded.json"
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)

    sample = random.choice([c for c in chunks if "embedding" in c])
    results = client.search(
        collection_name=COLLECTION,
        query_vector=sample["embedding"],
        limit=3,
        with_payload=True,
    )

    print("\nVerification search results:")
    for r in results:
        print(f"  score={r.score:.3f} | {r.payload.get('scheme_name','')[:60]} | {r.payload.get('chunk_type','')}")


if __name__ == "__main__":
    client = get_client()
    create_collection(client, recreate=False)
    upload_chunks(client)
    verify(client)
