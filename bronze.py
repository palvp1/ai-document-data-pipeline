import hashlib
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import pymupdf


BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data/raw"
BRONZE_DIR = BASE_DIR / "data/bronze"
REPORT_DIR = BASE_DIR / "reports"

BRONZE_FILE = BRONZE_DIR / "bronze_documents.parquet"
MANIFEST_FILE = BRONZE_DIR / "ingestion_manifest.json"

SUPPORTED_FORMATS = {".pdf", ".json", ".csv"}


def calculate_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(lambda: file.read(65536), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def extract_text(file_path):

    extension = file_path.suffix.lower()

    if extension == ".pdf":

        with pymupdf.open(str(file_path)) as document:

            text = "\n".join(
                page.get_text() for page in document
            )

            return text, len(document)

    elif extension in [".json", ".csv"]:

        text = file_path.read_text(encoding="utf-8")

        return text, 0

    raise ValueError(f"Unsupported format: {extension}")


def load_existing_data():

    if BRONZE_FILE.exists():
        return pd.read_parquet(BRONZE_FILE)

    return pd.DataFrame()


def run_bronze():

    start_time = time.perf_counter()

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    existing_data = load_existing_data()

    manifest = {}

    if not existing_data.empty:

        manifest = dict(
            zip(
                existing_data["source_path"],
                existing_data["file_hash"]
            )
        )

    files = sorted(
        file_path
        for file_path in RAW_DIR.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_FORMATS
    )

    new_records = []
    errors = []

    skipped = 0
    new_documents = 0
    modified_documents = 0

    format_counts = {
        "pdf": 0,
        "json": 0,
        "csv": 0
    }

    for file_path in files:

        relative_path = file_path.relative_to(
            BASE_DIR
        ).as_posix()

        file_format = file_path.suffix.lower()[1:]

        format_counts[file_format] += 1

        try:

            file_hash = calculate_hash(file_path)

            if manifest.get(relative_path) == file_hash:

                skipped += 1
                continue

            text, page_count = extract_text(file_path)

            if not text.strip():
                raise ValueError("Document contains no text")

            record = {
                "document_id": file_path.stem,
                "file_name": file_path.name,
                "file_format": file_format,
                "source_path": relative_path,
                "file_hash": file_hash,
                "file_size_bytes": file_path.stat().st_size,
                "page_count": page_count,
                "raw_text": text,
                "text_length": len(text),
                "ingested_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }

            new_records.append(record)

            if relative_path in manifest:
                modified_documents += 1
            else:
                new_documents += 1

            print(
                f"INGESTED: {file_path.name}"
            )

        except Exception as error:

            errors.append({
                "file_name": file_path.name,
                "source_path": relative_path,
                "error": str(error)
            })

            print(
                f"FAILED: {file_path.name} - {error}"
            )

    if new_records:

        new_data = pd.DataFrame(new_records)

        bronze_data = pd.concat(
            [existing_data, new_data],
            ignore_index=True
        )

        bronze_data = bronze_data.drop_duplicates(
            subset=["source_path"],
            keep="last"
        )

    else:

        bronze_data = existing_data.copy()

    if not bronze_data.empty:

        bronze_data.to_parquet(
            BRONZE_FILE,
            index=False
        )

        manifest = dict(
            zip(
                bronze_data["source_path"],
                bronze_data["file_hash"]
            )
        )

    with open(
        MANIFEST_FILE, "w", encoding="utf-8"
    ) as file:

        json.dump(manifest, file, indent=4)

    duration = round(
        time.perf_counter() - start_time, 2
    )

    reconciliation_passed = (
        len(files)
        == new_documents
        + modified_documents
        + skipped
        + len(errors)
    )

    report = {
        "stage": "Bronze Ingestion",
        "status": (
            "SUCCESS"
            if not errors and reconciliation_passed
            else "FAILED"
        ),
        "total_discovered": len(files),
        "new_documents": new_documents,
        "modified_documents": modified_documents,
        "skipped_documents": skipped,
        "failed_documents": len(errors),
        "total_bronze_documents": len(bronze_data),
        "pdf_documents": format_counts["pdf"],
        "json_documents": format_counts["json"],
        "csv_documents": format_counts["csv"],
        "reconciliation": (
            "PASS" if reconciliation_passed else "FAIL"
        ),
        "execution_time_seconds": duration,
        "errors": errors,
        "executed_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    report_path = REPORT_DIR / "bronze_report.json"

    with open(
        report_path, "w", encoding="utf-8"
    ) as file:

        json.dump(report, file, indent=4)

    print("\n" + "=" * 55)
    print("BRONZE DOCUMENT INGESTION COMPLETED")
    print("=" * 55)
    print(f"Documents discovered : {len(files)}")
    print(f"PDF documents        : {format_counts['pdf']}")
    print(f"JSON documents       : {format_counts['json']}")
    print(f"CSV documents        : {format_counts['csv']}")
    print(f"New documents        : {new_documents}")
    print(f"Modified documents   : {modified_documents}")
    print(f"Skipped documents    : {skipped}")
    print(f"Failed documents     : {len(errors)}")
    print(f"Total Bronze records : {len(bronze_data)}")
    print(f"Reconciliation       : {report['reconciliation']}")
    print(f"Execution time       : {duration} seconds")
    print(f"Bronze file          : {BRONZE_FILE}")
    print(f"Report saved         : {report_path}")
    print("=" * 55)

    return report


if __name__ == "__main__":

    result = run_bronze()

    if result["status"] != "SUCCESS":
        raise SystemExit(1)