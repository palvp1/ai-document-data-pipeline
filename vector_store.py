import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent

SILVER_FILE = (
    BASE_DIR / "data/silver/clean_invoices.parquet"
)

VECTOR_DIR = BASE_DIR / "data/vector_store"
REPORT_DIR = BASE_DIR / "reports"

VECTOR_FILE = VECTOR_DIR / "invoice_embeddings.npy"
METADATA_FILE = VECTOR_DIR / "invoice_metadata.parquet"

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"

BATCH_SIZE = 10


def build_document_text(record):

    return f"""
Invoice ID: {record['invoice_id']}
Vendor: {record['vendor']}
Invoice Date: {record['invoice_date']}
Due Date: {record['due_date']}
Category: {record['category']}
Subtotal: {record['subtotal']}
Tax: {record['tax']}
Total Amount: {record['total_amount']}
Currency: {record['currency']}
Payment Status: {record['status']}
Source Document: {record['file_name']}
"""


def generate_embeddings(texts):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts
        },
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    vectors = np.asarray(
        result["embeddings"],
        dtype=np.float32
    )

    if vectors.ndim != 2:
        raise ValueError(
            "Invalid embedding dimensions"
        )

    if len(vectors) != len(texts):
        raise ValueError(
            "Embedding count mismatch"
        )

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Invalid embedding values"
        )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    if np.any(norms == 0):
        raise ValueError(
            "Zero-length embedding detected"
        )

    return vectors / norms


def run_vector_store():

    start_time = time.perf_counter()

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    invoices = pd.read_parquet(SILVER_FILE)

    if invoices.empty:
        raise ValueError(
            "No validated invoices available"
        )

    records = invoices.to_dict("records")

    documents = []

    for record in records:

        documents.append({
            "document_id": record["document_id"],
            "invoice_id": record["invoice_id"],
            "vendor": record["vendor"],
            "category": record["category"],
            "status": record["status"],
            "total_amount": record["total_amount"],
            "file_name": record["file_name"],
            "source_path": record["source_path"],
            "file_hash": record["file_hash"],
            "document_text": build_document_text(
                record
            )
        })

    metadata = pd.DataFrame(documents)

    total_documents = len(metadata)

    total_batches = (
        total_documents + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print("=" * 60)
    print("VECTOR EMBEDDING PIPELINE STARTED")
    print("=" * 60)

    print(f"Embedding model : {EMBEDDING_MODEL}")
    print(f"Total documents : {total_documents}")
    print(f"Total batches   : {total_batches}")

    all_vectors = []

    for index in range(
        0, total_documents, BATCH_SIZE
    ):

        batch = metadata.iloc[
            index:index + BATCH_SIZE
        ]

        batch_number = (
            index // BATCH_SIZE + 1
        )

        print(
            f"\nProcessing embedding batch "
            f"{batch_number}/{total_batches}...",
            flush=True
        )

        batch_start = time.perf_counter()

        texts = [
            "search_document: " + text
            for text in batch["document_text"]
        ]

        vectors = generate_embeddings(texts)

        all_vectors.append(vectors)

        duration = round(
            time.perf_counter() - batch_start,
            2
        )

        print(
            f"Generated {len(vectors)} embeddings "
            f"in {duration} seconds",
            flush=True
        )

    embeddings = np.vstack(all_vectors)

    if len(embeddings) != total_documents:

        raise ValueError(
            "Vector reconciliation failed"
        )

    np.save(
        VECTOR_FILE,
        embeddings
    )

    metadata.to_parquet(
        METADATA_FILE,
        index=False
    )

    duration = round(
        time.perf_counter() - start_time,
        2
    )

    report = {
        "stage": "Vector Embedding",
        "status": "SUCCESS",
        "embedding_model": EMBEDDING_MODEL,
        "total_documents": total_documents,
        "total_embeddings": len(embeddings),
        "embedding_dimensions": int(
            embeddings.shape[1]
        ),
        "total_batches": total_batches,
        "vector_storage_bytes": (
            VECTOR_FILE.stat().st_size
        ),
        "reconciliation": "PASS",
        "execution_time_seconds": duration,
        "executed_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    report_path = (
        REPORT_DIR / "vector_report.json"
    )

    with open(
        report_path, "w", encoding="utf-8"
    ) as file:

        json.dump(report, file, indent=4)

    print("\n" + "=" * 60)
    print("VECTOR EMBEDDING PIPELINE COMPLETED")
    print("=" * 60)

    print(f"Documents indexed    : {total_documents}")
    print(f"Embeddings generated : {len(embeddings)}")
    print(f"Embedding dimensions : {embeddings.shape[1]}")
    print(f"Embedding model      : {EMBEDDING_MODEL}")
    print(f"Reconciliation       : PASS")
    print(f"Execution time       : {duration} seconds")
    print(f"Vector file          : {VECTOR_FILE}")
    print(f"Metadata file        : {METADATA_FILE}")
    print(f"Report saved         : {report_path}")
    print("=" * 60)

    return report


if __name__ == "__main__":

    run_vector_store()