"""
build_db.py — Run this ONCE locally before launching the Streamlit app.
It reads news.csv, deduplicates by URL, and stores articles in a 
persistent ChromaDB collection with OpenAI embeddings.
"""

import csv
import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# ── Configuration ────────────────────────────────────────────────────
CSV_PATH = "HW7_data/news.csv"
DB_DIR = "HW7_data/news_chroma_db"
COLLECTION_NAME = "news_articles"

# Reads key from environment or .streamlit/secrets.toml isn't available here,
# so set OPENAI_API_KEY in your shell before running:
#   export OPENAI_API_KEY="sk-..."
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("Set OPENAI_API_KEY environment variable before running.")


def main():
    # ── Read & deduplicate ───────────────────────────────────────────
    seen_urls = set()
    articles = []

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row["URL"].strip()
            if url in seen_urls:
                continue
            seen_urls.add(url)
            articles.append(row)

    print(f"Loaded {len(articles)} unique articles (from CSV with duplicates removed)")

    # ── Build ChromaDB ───────────────────────────────────────────────
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )

    client = chromadb.PersistentClient(path=DB_DIR)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    # ── Batch insert (ChromaDB max batch = 5461) ────────────────────
    BATCH_SIZE = 200
    for i in range(0, len(articles), BATCH_SIZE):
        batch = articles[i : i + BATCH_SIZE]
        collection.add(
            ids=[f"article_{i + j}" for j in range(len(batch))],
            documents=[row["Document"] for row in batch],
            metadatas=[
                {
                    "company": row["company_name"].strip(),
                    "date": row["Date"],
                    "url": row["URL"].strip(),
                }
                for row in batch
            ],
        )
        print(f"  Inserted batch {i // BATCH_SIZE + 1} ({len(batch)} articles)")

    print(f"\nDone! ChromaDB persisted to ./{DB_DIR}/")
    print(f"Collection '{COLLECTION_NAME}' has {collection.count()} documents.")


if __name__ == "__main__":
    main()