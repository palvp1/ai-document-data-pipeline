import json
import time
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

SILVER_FILE = (
    BASE_DIR / "data/silver/clean_invoices.parquet"
)

QUARANTINE_FILE = (
    BASE_DIR / "data/quarantine/rejected_invoices.parquet"
)

GOLD_DIR = BASE_DIR / "data/gold"
REPORT_DIR = BASE_DIR / "reports"

DATABASE = GOLD_DIR / "invoice_analytics.duckdb"


def run_gold():

    start_time = time.perf_counter()

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    clean = pd.read_parquet(SILVER_FILE)
    rejected = pd.read_parquet(QUARANTINE_FILE)

    with open(
        REPORT_DIR / "silver_report.json",
        encoding="utf-8"
    ) as file:

        silver_report = json.load(file)

    quality = pd.DataFrame([
        {
            "quality_issue": issue,
            "affected_documents": count
        }
        for issue, count in
        silver_report["error_counts"].items()
    ], columns=[
        "quality_issue",
        "affected_documents"
    ])

    print("=" * 60)
    print("GOLD ANALYTICS WAREHOUSE STARTED")
    print("=" * 60)

    with duckdb.connect(str(DATABASE)) as conn:

        conn.register("clean_df", clean)
        conn.register("rejected_df", rejected)
        conn.register("quality_df", quality)

        conn.execute("BEGIN TRANSACTION")

        conn.execute("""
            CREATE OR REPLACE TABLE invoices AS
            SELECT
                * EXCLUDE (
                    invoice_date,
                    due_date,
                    subtotal,
                    tax,
                    total_amount
                ),
                CAST(invoice_date AS DATE)
                    AS invoice_date,
                CAST(due_date AS DATE)
                    AS due_date,
                CAST(subtotal AS DECIMAL(18,2))
                    AS subtotal,
                CAST(tax AS DECIMAL(18,2))
                    AS tax,
                CAST(total_amount AS DECIMAL(18,2))
                    AS total_amount
            FROM clean_df
        """)

        conn.execute("""
            CREATE OR REPLACE TABLE rejected_invoices AS
            SELECT * FROM rejected_df
        """)

        conn.execute("""
            CREATE OR REPLACE TABLE vendor_summary AS
            SELECT
                vendor,
                COUNT(*) AS invoice_count,
                SUM(total_amount) AS total_spend,
                ROUND(AVG(total_amount), 2)
                    AS average_invoice,
                SUM(
                    CASE WHEN status = 'PAID'
                    THEN total_amount ELSE 0 END
                ) AS paid_amount,
                SUM(
                    CASE WHEN status != 'PAID'
                    THEN total_amount ELSE 0 END
                ) AS outstanding_amount
            FROM invoices
            GROUP BY vendor
            ORDER BY total_spend DESC
        """)

        conn.execute("""
            CREATE OR REPLACE TABLE category_summary AS
            SELECT
                category,
                COUNT(*) AS invoice_count,
                SUM(total_amount) AS total_spend,
                ROUND(AVG(total_amount), 2)
                    AS average_invoice
            FROM invoices
            GROUP BY category
            ORDER BY total_spend DESC
        """)

        conn.execute("""
            CREATE OR REPLACE TABLE status_summary AS
            SELECT
                status,
                COUNT(*) AS invoice_count,
                SUM(total_amount) AS total_amount
            FROM invoices
            GROUP BY status
            ORDER BY total_amount DESC
        """)

        conn.execute("""
            CREATE OR REPLACE TABLE monthly_summary AS
            SELECT
                STRFTIME(invoice_date, '%Y-%m')
                    AS invoice_month,
                COUNT(*) AS invoice_count,
                SUM(total_amount) AS total_spend,
                SUM(tax) AS total_tax
            FROM invoices
            GROUP BY invoice_month
            ORDER BY invoice_month
        """)

        conn.execute("""
            CREATE OR REPLACE TABLE format_summary AS
            SELECT
                file_format,
                COUNT(*) AS total_documents,
                SUM(is_valid) AS valid_documents,
                COUNT(*) - SUM(is_valid)
                    AS rejected_documents,
                ROUND(
                    100.0 * SUM(is_valid) / COUNT(*),
                    2
                ) AS quality_score
            FROM (
                SELECT
                    file_format,
                    1 AS is_valid
                FROM invoices

                UNION ALL

                SELECT
                    file_format,
                    0 AS is_valid
                FROM rejected_invoices
            )
            GROUP BY file_format
            ORDER BY file_format
        """)

        conn.execute("""
            CREATE OR REPLACE TABLE quality_summary AS
            SELECT * FROM quality_df
            ORDER BY affected_documents DESC
        """)

        conn.execute("""
            CREATE OR REPLACE TABLE pipeline_kpis AS
            SELECT
                COUNT(*) AS valid_invoices,

                (
                    SELECT COUNT(*)
                    FROM rejected_invoices
                ) AS rejected_invoices,

                COUNT(*) + (
                    SELECT COUNT(*)
                    FROM rejected_invoices
                ) AS total_documents,

                ROUND(
                    100.0 * COUNT(*) /
                    NULLIF(
                        COUNT(*) + (
                            SELECT COUNT(*)
                            FROM rejected_invoices
                        ), 0
                    ), 2
                ) AS data_quality_score,

                COUNT(DISTINCT vendor)
                    AS total_vendors,

                SUM(total_amount)
                    AS total_invoice_value,

                SUM(tax)
                    AS total_tax,

                ROUND(AVG(total_amount), 2)
                    AS average_invoice_value,

                SUM(
                    CASE WHEN status = 'PAID'
                    THEN total_amount ELSE 0 END
                ) AS paid_amount,

                SUM(
                    CASE WHEN status = 'PENDING'
                    THEN total_amount ELSE 0 END
                ) AS pending_amount,

                SUM(
                    CASE WHEN status = 'OVERDUE'
                    THEN total_amount ELSE 0 END
                ) AS overdue_amount

            FROM invoices
        """)

        kpis = conn.execute("""
            SELECT * FROM pipeline_kpis
        """).fetchdf().iloc[0].to_dict()

        tables = [
            "invoices",
            "rejected_invoices",
            "vendor_summary",
            "category_summary",
            "status_summary",
            "monthly_summary",
            "format_summary",
            "quality_summary",
            "pipeline_kpis"
        ]

        valid_count = int(kpis["valid_invoices"])
        rejected_count = int(kpis["rejected_invoices"])

        total_value = float(
            kpis["total_invoice_value"]
        )

        paid_amount = float(kpis["paid_amount"])
        pending_amount = float(kpis["pending_amount"])
        overdue_amount = float(kpis["overdue_amount"])

        source_amount = float(
            clean["total_amount"].sum()
        )

        record_check = (
            valid_count == len(clean)
            and rejected_count == len(rejected)
            and valid_count + rejected_count
            == silver_report["total_documents"]
        )

        amount_check = (
            abs(source_amount - total_value) <= 0.01
        )

        status_check = (
            abs(
                total_value
                - paid_amount
                - pending_amount
                - overdue_amount
            ) <= 0.01
        )

        vendor_total = conn.execute("""
            SELECT SUM(total_spend)
            FROM vendor_summary
        """).fetchone()[0]

        vendor_check = (
            abs(float(vendor_total) - total_value)
            <= 0.01
        )

        reconciliation = {
            "record_reconciliation": (
                "PASS" if record_check else "FAIL"
            ),
            "amount_reconciliation": (
                "PASS" if amount_check else "FAIL"
            ),
            "status_reconciliation": (
                "PASS" if status_check else "FAIL"
            ),
            "vendor_reconciliation": (
                "PASS" if vendor_check else "FAIL"
            )
        }

        all_passed = all(
            value == "PASS"
            for value in reconciliation.values()
        )

        if not all_passed:

            conn.execute("ROLLBACK")

            raise ValueError(
                f"Gold reconciliation failed: "
                f"{reconciliation}"
            )

        conn.execute("COMMIT")

        for table in tables:

            output_file = (
                GOLD_DIR / f"{table}.parquet"
            )

            conn.execute(
                f"SELECT * FROM {table}"
            ).fetchdf().to_parquet(
                output_file,
                index=False
            )

            count = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

            print(
                f"CREATED: {table} "
                f"({count} records)"
            )

    duration = round(
        time.perf_counter() - start_time, 2
    )

    report = {
        "stage": "Gold Analytics",
        "status": "SUCCESS",
        "total_documents": valid_count + rejected_count,
        "valid_invoices": valid_count,
        "rejected_invoices": rejected_count,
        "data_quality_score": float(
            kpis["data_quality_score"]
        ),
        "total_vendors": int(
            kpis["total_vendors"]
        ),
        "total_invoice_value": total_value,
        "total_tax": float(kpis["total_tax"]),
        "average_invoice_value": float(
            kpis["average_invoice_value"]
        ),
        "paid_amount": paid_amount,
        "pending_amount": pending_amount,
        "overdue_amount": overdue_amount,
        "database_tables": len(tables),
        "tables": tables,
        "reconciliation": reconciliation,
        "execution_time_seconds": duration,
        "executed_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    report_path = REPORT_DIR / "gold_report.json"

    with open(
        report_path, "w", encoding="utf-8"
    ) as file:

        json.dump(report, file, indent=4)

    print("\n" + "=" * 60)
    print("GOLD ANALYTICS WAREHOUSE COMPLETED")
    print("=" * 60)

    print(f"Total documents      : {valid_count + rejected_count}")
    print(f"Valid invoices       : {valid_count}")
    print(f"Rejected invoices    : {rejected_count}")
    print(f"Total vendors        : {report['total_vendors']}")
    print(f"Total invoice value  : ${total_value:,.2f}")
    print(f"Paid amount          : ${paid_amount:,.2f}")
    print(f"Pending amount       : ${pending_amount:,.2f}")
    print(f"Overdue amount       : ${overdue_amount:,.2f}")
    print(f"Total tax            : ${report['total_tax']:,.2f}")
    print(f"Database tables      : {len(tables)}")

    print("\nRECONCILIATION RESULTS")

    for check, result in reconciliation.items():

        print(f"{check}: {result}")

    print(f"\nExecution time : {duration} seconds")
    print(f"Database       : {DATABASE}")
    print(f"Report         : {report_path}")
    print("=" * 60)

    return report


if __name__ == "__main__":

    run_gold()