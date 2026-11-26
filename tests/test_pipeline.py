import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

BRONZE_FILE = (
    BASE_DIR / "data/bronze/bronze_documents.parquet"
)

SILVER_FILE = (
    BASE_DIR / "data/silver/clean_invoices.parquet"
)

QUARANTINE_FILE = (
    BASE_DIR / "data/quarantine/rejected_invoices.parquet"
)

DATABASE = (
    BASE_DIR / "data/gold/invoice_analytics.duckdb"
)

VECTOR_FILE = (
    BASE_DIR / "data/vector_store/invoice_embeddings.npy"
)

METADATA_FILE = (
    BASE_DIR / "data/vector_store/invoice_metadata.parquet"
)

REPORT_DIR = BASE_DIR / "reports"


# ==================================================
# TEST FIXTURES
# ==================================================

@pytest.fixture(scope="module")
def bronze():
    return pd.read_parquet(BRONZE_FILE)


@pytest.fixture(scope="module")
def silver():
    return pd.read_parquet(SILVER_FILE)


@pytest.fixture(scope="module")
def rejected():
    return pd.read_parquet(QUARANTINE_FILE)


@pytest.fixture(scope="module")
def embeddings():
    return np.load(VECTOR_FILE)


@pytest.fixture(scope="module")
def metadata():
    return pd.read_parquet(METADATA_FILE)


@pytest.fixture(scope="module")
def database():
    conn = duckdb.connect(
        str(DATABASE),
        read_only=True
    )

    yield conn

    conn.close()


def load_report(filename):
    with open(
        REPORT_DIR / filename,
        encoding="utf-8"
    ) as file:
        return json.load(file)


# ==================================================
# BRONZE LAYER TESTS
# ==================================================

def test_bronze_file_exists():
    assert BRONZE_FILE.exists()


def test_bronze_document_count(bronze):
    assert len(bronze) == 100


def test_bronze_no_duplicate_documents(bronze):
    assert bronze["document_id"].is_unique


def test_bronze_file_format_distribution(bronze):

    counts = bronze["file_format"].value_counts()

    assert counts["pdf"] == 60
    assert counts["json"] == 20
    assert counts["csv"] == 20


def test_bronze_document_content(bronze):

    assert bronze["raw_text"].notna().all()

    assert (
        bronze["raw_text"].str.strip().str.len() > 0
    ).all()


def test_bronze_file_hashes(bronze):

    assert bronze["file_hash"].str.fullmatch(
        r"[a-fA-F0-9]{64}"
    ).all()


# ==================================================
# SILVER LAYER TESTS
# ==================================================

def test_silver_file_exists():
    assert SILVER_FILE.exists()


def test_silver_valid_document_count(silver):
    assert len(silver) == 90


def test_silver_no_duplicate_invoices(silver):
    assert silver["invoice_id"].is_unique


def test_silver_required_fields(silver):

    required_fields = [
        "invoice_id",
        "vendor",
        "invoice_date",
        "due_date",
        "category",
        "subtotal",
        "tax",
        "total_amount",
        "currency",
        "status"
    ]

    assert set(required_fields).issubset(
        silver.columns
    )

    assert silver[required_fields].notna().all().all()


def test_silver_positive_amounts(silver):

    assert (silver["subtotal"] > 0).all()
    assert (silver["tax"] >= 0).all()
    assert (silver["total_amount"] > 0).all()


def test_silver_financial_calculation(silver):

    expected = (
        silver["subtotal"] + silver["tax"]
    )

    assert np.allclose(
        expected,
        silver["total_amount"],
        atol=0.010001,
        rtol=0
    )


def test_silver_valid_status(silver):

    valid_statuses = {
        "PAID",
        "PENDING",
        "OVERDUE"
    }

    assert set(
        silver["status"]
    ).issubset(valid_statuses)


def test_silver_valid_currency(silver):
    assert (silver["currency"] == "USD").all()


def test_silver_valid_dates(silver):

    invoice_dates = pd.to_datetime(
        silver["invoice_date"],
        errors="coerce"
    )

    due_dates = pd.to_datetime(
        silver["due_date"],
        errors="coerce"
    )

    assert invoice_dates.notna().all()
    assert due_dates.notna().all()
    assert (due_dates >= invoice_dates).all()


def test_silver_vendor_not_empty(silver):

    assert (
        silver["vendor"].str.strip().str.len() > 0
    ).all()


def test_silver_extraction_methods(silver):

    methods = set(silver["extraction_method"])

    assert methods == {
        "LLAMA3.1",
        "NATIVE_PARSER"
    }


# ==================================================
# QUARANTINE TESTS
# ==================================================

def test_quarantine_file_exists():
    assert QUARANTINE_FILE.exists()


def test_quarantine_document_count(rejected):
    assert len(rejected) == 10


def test_quarantine_rejection_reasons(rejected):

    assert rejected["rejection_reason"].notna().all()

    assert (
        rejected["rejection_reason"]
        .str.strip()
        .str.len() > 0
    ).all()


def test_quarantine_expected_invalid_documents(rejected):

    expected_ids = {
        "INV-00005",
        "INV-00015",
        "INV-00025",
        "INV-00035",
        "INV-00045",
        "INV-00055",
        "INV-00065",
        "INV-00075",
        "INV-00085",
        "INV-00095"
    }

    actual_ids = set(
        rejected["document_id"]
    )

    assert actual_ids == expected_ids


def test_clean_and_quarantine_no_overlap(
    silver,
    rejected
):

    clean_ids = set(silver["document_id"])

    rejected_ids = set(
        rejected["document_id"]
    )

    assert clean_ids.isdisjoint(rejected_ids)


def test_document_reconciliation(
    bronze,
    silver,
    rejected
):

    bronze_ids = set(bronze["document_id"])

    processed_ids = (
        set(silver["document_id"])
        | set(rejected["document_id"])
    )

    assert bronze_ids == processed_ids

    assert len(bronze) == (
        len(silver) + len(rejected)
    )


# ==================================================
# SOURCE ACCURACY TESTS
# ==================================================

def normalize_text(value):

    return " ".join(
        str(value).strip().split()
    )


def extract_pdf_source_fields(raw_text):

    fields = [
        "invoice_id",
        "vendor",
        "invoice_date",
        "due_date",
        "category",
        "subtotal",
        "tax",
        "total_amount",
        "currency",
        "status"
    ]

    labels = [
        field.replace("_", " ").title()
        for field in fields
    ]

    extracted = {}

    for field, label in zip(fields, labels):

        other_labels = [
            re.escape(item)
            for item in labels
            if item != label
        ]

        boundaries = "|".join(other_labels)

        pattern = (
            rf"(?im)^[ \t]*{re.escape(label)}"
            rf"[ \t]*:?[ \t]*"
            rf"(.*?)"
            rf"(?=^[ \t]*(?:{boundaries})"
            rf"[ \t]*:?[ \t]*(?:\s|$)"
            rf"|^[ \t]*Thank you"
            rf"|\Z)"
        )

        match = re.search(
            pattern,
            raw_text,
            flags=re.DOTALL
        )

        if match:

            extracted[field] = normalize_text(
                match.group(1)
            )

    return extracted


@pytest.fixture(scope="module")
def pdf_comparison(bronze, silver):

    pdfs = silver[
        silver["file_format"] == "pdf"
    ]

    raw_documents = bronze.set_index(
        "document_id"
    )["raw_text"]

    comparisons = []

    for _, row in pdfs.iterrows():

        document_id = row["document_id"]

        source = extract_pdf_source_fields(
            raw_documents.loc[document_id]
        )

        comparisons.append((row, source))

    return comparisons


def test_ai_pdf_source_fields_available(pdf_comparison):

    assert len(pdf_comparison) == 54

    for row, source in pdf_comparison:

        assert len(source) == 10, (
            f"Unable to parse all source fields "
            f"for {row['document_id']}: {source}"
        )


def test_ai_pdf_text_accuracy(pdf_comparison):

    fields = [
        "invoice_id",
        "vendor",
        "category",
        "currency",
        "status"
    ]

    mismatches = []

    for row, source in pdf_comparison:

        for field in fields:

            expected = normalize_text(
                source.get(field, "")
            )

            actual = normalize_text(
                row[field]
            )

            if expected != actual:

                mismatches.append(
                    f"{row['document_id']} | "
                    f"{field} | "
                    f"Expected: {expected} | "
                    f"Actual: {actual}"
                )

    assert not mismatches, "\n".join(mismatches)


def test_ai_pdf_financial_accuracy(pdf_comparison):

    fields = [
        "subtotal",
        "tax",
        "total_amount"
    ]

    mismatches = []

    for row, source in pdf_comparison:

        for field in fields:

            raw_value = source.get(field, "")

            try:

                expected = float(
                    raw_value
                    .replace("$", "")
                    .replace(",", "")
                )

            except ValueError:

                mismatches.append(
                    f"{row['document_id']} | "
                    f"Unable to parse {field}: "
                    f"{raw_value}"
                )

                continue

            actual = float(row[field])

            if not np.isclose(
                expected,
                actual,
                atol=0.001,
                rtol=0
            ):

                mismatches.append(
                    f"{row['document_id']} | "
                    f"{field} | "
                    f"Expected: {expected} | "
                    f"Actual: {actual}"
                )

    assert not mismatches, "\n".join(mismatches)


def test_ai_pdf_date_accuracy(pdf_comparison):

    mismatches = []

    for row, source in pdf_comparison:

        for field in [
            "invoice_date",
            "due_date"
        ]:

            expected = pd.to_datetime(
                source.get(field),
                errors="coerce"
            )

            actual = pd.to_datetime(
                row[field],
                errors="coerce"
            )

            if (
                pd.isna(expected)
                or pd.isna(actual)
                or expected != actual
            ):

                mismatches.append(
                    f"{row['document_id']} | "
                    f"{field} | "
                    f"Expected: {expected} | "
                    f"Actual: {actual}"
                )

    assert not mismatches, "\n".join(mismatches)


# ==================================================
# GOLD WAREHOUSE TESTS
# ==================================================

def test_gold_database_exists():
    assert DATABASE.exists()


def test_gold_required_tables(database):

    tables = {
        row[0]
        for row in database.execute(
            "SHOW TABLES"
        ).fetchall()
    }

    expected_tables = {
        "invoices",
        "rejected_invoices",
        "vendor_summary",
        "category_summary",
        "status_summary",
        "monthly_summary",
        "format_summary",
        "quality_summary",
        "pipeline_kpis"
    }

    assert expected_tables.issubset(tables)


def test_gold_invoice_count(database):

    count = database.execute(
        "SELECT COUNT(*) FROM invoices"
    ).fetchone()[0]

    assert count == 90


def test_gold_quarantine_count(database):

    count = database.execute(
        "SELECT COUNT(*) FROM rejected_invoices"
    ).fetchone()[0]

    assert count == 10


def test_gold_financial_reconciliation(
    database,
    silver
):

    warehouse_total = database.execute(
        "SELECT SUM(total_amount) FROM invoices"
    ).fetchone()[0]

    silver_total = silver["total_amount"].sum()

    assert np.isclose(
        float(warehouse_total),
        float(silver_total),
        atol=0.01,
        rtol=0
    )


def test_gold_vendor_reconciliation(database):

    invoice_total = database.execute(
        "SELECT SUM(total_amount) FROM invoices"
    ).fetchone()[0]

    vendor_total = database.execute(
        "SELECT SUM(total_spend) FROM vendor_summary"
    ).fetchone()[0]

    assert np.isclose(
        float(invoice_total),
        float(vendor_total),
        atol=0.01,
        rtol=0
    )


def test_gold_status_reconciliation(database):

    invoice_total = database.execute(
        "SELECT SUM(total_amount) FROM invoices"
    ).fetchone()[0]

    status_total = database.execute(
        "SELECT SUM(total_amount) FROM status_summary"
    ).fetchone()[0]

    assert np.isclose(
        float(invoice_total),
        float(status_total),
        atol=0.01,
        rtol=0
    )


def test_gold_quality_score(database):

    score = database.execute(
        "SELECT data_quality_score FROM pipeline_kpis"
    ).fetchone()[0]

    assert float(score) == 90.0


# ==================================================
# VECTOR STORE TESTS
# ==================================================

def test_vector_file_exists():
    assert VECTOR_FILE.exists()


def test_vector_metadata_exists():
    assert METADATA_FILE.exists()


def test_vector_embedding_count(embeddings):
    assert embeddings.shape[0] == 90


def test_vector_embedding_dimensions(embeddings):
    assert embeddings.shape[1] == 768


def test_vector_finite_values(embeddings):
    assert np.isfinite(embeddings).all()


def test_vector_nonzero_embeddings(embeddings):

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    assert (norms > 0).all()


def test_vector_metadata_reconciliation(
    embeddings,
    metadata
):

    assert len(metadata) == len(embeddings)


def test_vector_no_duplicate_documents(metadata):

    assert metadata["invoice_id"].is_unique


def test_vector_only_valid_invoices(
    metadata,
    silver
):

    assert set(
        metadata["invoice_id"]
    ) == set(
        silver["invoice_id"]
    )


# ==================================================
# PIPELINE REPORT TESTS
# ==================================================

def test_bronze_report():

    report = load_report(
        "bronze_report.json"
    )

    assert report["status"] == "SUCCESS"
    assert report["failed_documents"] == 0
    assert report["reconciliation"] == "PASS"


def test_silver_report():

    report = load_report(
        "silver_report.json"
    )

    assert report["valid_documents"] == 90
    assert report["rejected_documents"] == 10
    assert report["extraction_failures"] == 0
    assert report["reconciliation"] == "PASS"


def test_gold_report():

    report = load_report(
        "gold_report.json"
    )

    assert all(
        result == "PASS"
        for result in report[
            "reconciliation"
        ].values()
    )

    assert len(report["reconciliation"]) == 4


def test_vector_report():

    report = load_report(
        "vector_report.json"
    )

    assert report["embedding_dimensions"] == 768
    assert report["reconciliation"] == "PASS"


def test_pipeline_success():

    report = load_report(
        "pipeline_report.json"
    )

    assert report["status"] == "SUCCESS"
    assert report["successful_stages"] == 4
    assert report["failed_stages"] == 0


# ==================================================
# RAG REPORT TEST
# ==================================================

def test_rag_citation_validation():

    report = load_report(
        "rag_report.json"
    )

    assert report["citation_validation"] == "PASS"

    assert not report["invalid_citations"]

    assert report["retrieved_documents"] > 0