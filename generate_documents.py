import csv
import json
import random
from pathlib import Path
from datetime import datetime, timedelta

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

random.seed(42)

VENDORS = [
    "Apex Technology Solutions",
    "Global Office Supplies",
    "Northstar Logistics",
    "Summit Cloud Services",
    "Vertex Consulting",
    "Blue Ocean Systems",
    "Pacific Industrial Supply",
    "Metro Business Services",
    "Pioneer Software",
    "Atlas Infrastructure"
]

CATEGORIES = [
    "Software",
    "Hardware",
    "Consulting",
    "Office Supplies",
    "Cloud Services",
    "Logistics"
]


def create_folders():
    for folder in [
        "data/raw/pdf",
        "data/raw/json",
        "data/raw/csv",
        "data/bronze",
        "data/silver",
        "data/gold",
        "data/quarantine",
        "data/vector_store",
        "reports",
        "tests",
        "screenshots"
    ]:
        (BASE_DIR / folder).mkdir(
            parents=True,
            exist_ok=True
        )


def generate_invoice(index):

    invoice_date = datetime(2026, 1, 1) + timedelta(
        days=random.randint(0, 300)
    )

    subtotal = round(random.uniform(100, 25000), 2)
    tax = round(subtotal * 0.07, 2)

    return {
        "invoice_id": f"INV-{index:05d}",
        "vendor": random.choice(VENDORS),
        "invoice_date": invoice_date.strftime("%Y-%m-%d"),
        "due_date": (
            invoice_date + timedelta(days=30)
        ).strftime("%Y-%m-%d"),
        "category": random.choice(CATEGORIES),
        "subtotal": subtotal,
        "tax": tax,
        "total_amount": round(subtotal + tax, 2),
        "currency": "USD",
        "status": random.choice(
            ["PAID", "PENDING", "OVERDUE"]
        )
    }


def introduce_errors(invoice, index):

    if index in [5, 15, 65]:
        invoice["vendor"] = ""

    elif index in [25, 75, 85]:
        invoice["total_amount"] = -500

    elif index in [35, 95]:
        invoice["invoice_date"] = "INVALID_DATE"

    elif index in [45, 55]:
        invoice["status"] = "UNKNOWN"

    return invoice


def save_pdf(invoice, path):

    pdf = canvas.Canvas(str(path), pagesize=letter)

    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(50, 740, "BUSINESS INVOICE")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        50, 710,
        "Synthetic document - Educational use only"
    )

    pdf.line(50, 690, 560, 690)

    y = 650

    for key, value in invoice.items():

        label = key.replace("_", " ").title()

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, f"{label}:")

        pdf.setFont("Helvetica", 12)
        pdf.drawString(220, y, str(value))

        y -= 35

    pdf.line(50, y, 560, y)

    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        50, y - 30,
        "Thank you for your business."
    )

    pdf.save()


def save_json(invoice, path):

    with open(path, "w", encoding="utf-8") as file:
        json.dump(invoice, file, indent=4)


def save_csv(invoice, path):

    with open(
        path, "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=invoice.keys()
        )

        writer.writeheader()
        writer.writerow(invoice)


def main():

    create_folders()

    counts = {
        "pdf": 0,
        "json": 0,
        "csv": 0
    }

    for index in range(1, 101):

        invoice = generate_invoice(index)
        invoice = introduce_errors(invoice, index)

        filename = invoice["invoice_id"]

        if index <= 60:

            save_pdf(
                invoice,
                RAW_DIR / "pdf" / f"{filename}.pdf"
            )

            counts["pdf"] += 1

        elif index <= 80:

            save_json(
                invoice,
                RAW_DIR / "json" / f"{filename}.json"
            )

            counts["json"] += 1

        else:

            save_csv(
                invoice,
                RAW_DIR / "csv" / f"{filename}.csv"
            )

            counts["csv"] += 1

    report = {
        "total_documents": 100,
        "pdf_documents": counts["pdf"],
        "json_documents": counts["json"],
        "csv_documents": counts["csv"],
        "intentional_invalid_documents": 10,
        "expected_valid_documents": 90,
        "generated_at": datetime.now().isoformat()
    }

    report_path = (
        BASE_DIR / "reports" / "generation_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("=" * 55)
    print("DOCUMENT GENERATION COMPLETED")
    print("=" * 55)
    print(f"PDF documents       : {counts['pdf']}")
    print(f"JSON documents      : {counts['json']}")
    print(f"CSV documents       : {counts['csv']}")
    print("Total documents     : 100")
    print("Intentional errors  : 10")
    print("Expected valid      : 90")
    print(f"Report saved        : {report_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()