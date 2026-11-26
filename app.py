import json
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from rag_engine import ask_question


BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"

DATABASE = (
    BASE_DIR / "data/gold/invoice_analytics.duckdb"
)

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# LIGHT ENTERPRISE THEME
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #f1f5f9;
    color: #0f172a;
}

[data-testid="stHeader"] {
    background-color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

[data-testid="stSidebar"] h1 {
    color: #1d4ed8 !important;
    font-weight: 800;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #334155;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 3px 10px rgba(15,23,42,0.05);
}

[data-testid="stMetricLabel"] p {
    color: #475569 !important;
    font-size: 14px;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #1d4ed8 !important;
    font-size: clamp(18px, 1.8vw, 28px);
    font-weight: 700;
}

h1, h2, h3 {
    color: #0f172a !important;
    font-weight: 700;
}

[data-testid="stCaptionContainer"] p {
    color: #475569 !important;
}

[data-testid="stPlotlyChart"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 3px 10px rgba(15,23,42,0.04);
}

[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}

.stButton > button[kind="primary"] {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stDownloadButton > button {
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
}

[data-testid="stExpander"] {
    background-color: #ffffff;
    border-radius: 10px;
}

hr {
    border-color: #e2e8f0 !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero-banner {
    background: linear-gradient(
        110deg, #1e3a8a, #2563eb
    );
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 24px;
    color: #ffffff;
}

.hero-banner h1 {
    color: #ffffff !important;
    font-size: 32px;
    margin: 0;
}

.hero-banner p {
    color: #dbeafe;
    font-size: 15px;
    margin-top: 10px;
    margin-bottom: 0;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# COLOR PALETTE
# ==================================================

COLORS = [
    "#2563eb",
    "#16a34a",
    "#d97706",
    "#dc2626",
    "#7c3aed",
    "#0891b2",
    "#64748b",
    "#db2777"
]

STATUS_COLORS = {
    "PAID": "#15803d",
    "PENDING": "#b45309",
    "OVERDUE": "#dc2626",
    "SUCCESS": "#15803d",
    "FAILED": "#dc2626",
    "SKIPPED": "#64748b"
}


# ==================================================
# DATA ACCESS
# ==================================================

def query(sql):

    with duckdb.connect(
        str(DATABASE),
        read_only=True
    ) as conn:

        return conn.execute(sql).fetchdf()


def load_report(filename):

    path = REPORT_DIR / filename

    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_history():

    path = REPORT_DIR / "pipeline_history.jsonl"

    if not path.exists():
        return pd.DataFrame()

    records = []

    with open(path, encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            try:

                report = json.loads(line)

                records.append({
                    "Run ID": report["run_id"],
                    "Status": report["status"],
                    "Started At": report["started_at"],
                    "Runtime": report[
                        "execution_time_seconds"
                    ],
                    "Successful Stages": report[
                        "successful_stages"
                    ],
                    "Failed Stages": report[
                        "failed_stages"
                    ]
                })

            except (
                json.JSONDecodeError,
                KeyError
            ):

                continue

    return pd.DataFrame(records)


def money(value):

    if pd.isna(value):
        return "$0.00"

    return f"${float(value):,.2f}"


def show_chart(fig):

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=370,
        margin=dict(
            l=25,
            r=25,
            t=65,
            b=30
        ),
        font=dict(
            family="Arial",
            size=13,
            color="#334155"
        ),
        title_font=dict(
            size=18,
            color="#0f172a"
        ),
        legend=dict(
            font=dict(
                color="#334155",
                size=12
            )
        ),
        colorway=COLORS
    )

    fig.update_xaxes(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        tickfont=dict(
            color="#475569",
            size=12
        ),
        title_font=dict(
            color="#334155"
        )
    )

    fig.update_yaxes(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        tickfont=dict(
            color="#475569",
            size=12
        ),
        title_font=dict(
            color="#334155"
        )
    )

    fig.update_traces(
        selector=dict(type="pie"),
        textfont=dict(
            color="#ffffff",
            size=13
        )
    )

    st.plotly_chart(
        fig,
        theme=None,
        use_container_width=True,
        config={
            "displaylogo": False
        }
    )


def page_header(title, description):

    st.title(title)
    st.caption(description)
    st.divider()


if not DATABASE.exists():

    st.error(
        "Database not found. "
        "Run python pipeline.py first."
    )

    st.stop()


kpis = query(
    "SELECT * FROM pipeline_kpis"
).iloc[0]

gold = load_report("gold_report.json")
bronze = load_report("bronze_report.json")
silver = load_report("silver_report.json")
vector = load_report("vector_report.json")
pipeline = load_report("pipeline_report.json")


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("📊 Document AI")

    st.caption(
        "AI-Powered Data Engineering Platform"
    )

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Executive Overview",
            "Invoice Explorer",
            "Financial Analytics",
            "Data Quality",
            "AI Document Assistant",
            "Pipeline Monitoring",
            "System Architecture"
        ]
    )

    st.divider()

    st.subheader("System Status")

    if pipeline.get("status") == "SUCCESS":

        st.success("Pipeline Operational")

    else:

        st.warning("Check Pipeline Status")

    st.metric(
        "Indexed Documents",
        vector.get("total_documents", 0)
    )

    st.metric(
        "Data Quality",
        f"{kpis['data_quality_score']:.1f}%"
    )

    st.divider()

    st.subheader("Local AI Models")

    st.markdown(
        "**Language Model**"
    )

    st.code(
        "llama3.1:8b",
        language="text"
    )

    st.markdown(
        "**Embedding Model**"
    )

    st.code(
        "nomic-embed-text",
        language="text"
    )

    st.caption(
        "Synthetic invoice dataset | Portfolio POC"
    )


# ==================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==================================================

if page == "Executive Overview":

    st.markdown("""
    <div class="hero-banner">
        <h1>AI Document Intelligence Platform</h1>
        <p>
            Intelligent document processing,
            financial analytics and AI-powered search
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Document Processing Overview")

    a, b, c, d = st.columns(4)

    a.metric(
        "Total Documents",
        int(kpis["total_documents"])
    )

    b.metric(
        "Valid Invoices",
        int(kpis["valid_invoices"])
    )

    c.metric(
        "Data Quality Score",
        f"{kpis['data_quality_score']:.1f}%"
    )

    d.metric(
        "Total Vendors",
        int(kpis["total_vendors"])
    )

    st.subheader("Financial Overview")

    a, b, c, d = st.columns(4)

    a.metric(
        "Total Invoice Value",
        money(kpis["total_invoice_value"])
    )

    b.metric(
        "Paid Amount",
        money(kpis["paid_amount"])
    )

    c.metric(
        "Pending Amount",
        money(kpis["pending_amount"])
    )

    d.metric(
        "Overdue Amount",
        money(kpis["overdue_amount"])
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        status = query(
            "SELECT * FROM status_summary"
        )

        fig = px.pie(
            status,
            names="status",
            values="total_amount",
            hole=0.65,
            title="Invoice Payment Distribution",
            color="status",
            color_discrete_map=STATUS_COLORS
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent"
        )

        fig.add_annotation(
            text=(
                f"<b>{int(kpis['valid_invoices'])}</b>"
                "<br>Invoices"
            ),
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=18,
                color="#0f172a"
            )
        )

        show_chart(fig)

    with right:

        monthly = query("""
            SELECT *
            FROM monthly_summary
            ORDER BY invoice_month
        """)

        fig = px.area(
            monthly,
            x="invoice_month",
            y="total_spend",
            markers=True,
            title="Monthly Invoice Value",
            color_discrete_sequence=[
                "#2563eb"
            ],
            labels={
                "invoice_month": "Month",
                "total_spend": "Invoice Value"
            }
        )

        fig.update_traces(
            line=dict(width=3),
            fillcolor="rgba(37,99,235,0.15)"
        )

        fig.update_yaxes(
            tickprefix="$"
        )

        show_chart(fig)

    left, right = st.columns(2)

    with left:

        vendors = query(
            "SELECT * FROM vendor_summary"
        )

        fig = px.bar(
            vendors,
            x="total_spend",
            y="vendor",
            orientation="h",
            color="total_spend",
            color_continuous_scale="Blues",
            title="Vendor Spending",
            labels={
                "total_spend": "Total Spend",
                "vendor": "Vendor"
            }
        )

        fig.update_yaxes(
            categoryorder="total ascending"
        )

        fig.update_xaxes(
            tickprefix="$"
        )

        fig.update_layout(
            coloraxis_showscale=False
        )

        show_chart(fig)

    with right:

        formats = query(
            "SELECT * FROM format_summary"
        )

        fig = px.bar(
            formats,
            x="file_format",
            y=[
                "valid_documents",
                "rejected_documents"
            ],
            title="Document Processing Results",
            barmode="stack",
            color_discrete_sequence=[
                "#16a34a",
                "#dc2626"
            ],
            labels={
                "file_format": "Document Format",
                "value": "Documents",
                "variable": "Result"
            }
        )

        show_chart(fig)


# ==================================================
# PAGE 2: INVOICE EXPLORER
# ==================================================

elif page == "Invoice Explorer":

    page_header(
        "Invoice Explorer",
        "Search, filter and export validated invoices."
    )

    invoices = query("""
        SELECT
            invoice_id,
            vendor,
            invoice_date,
            due_date,
            category,
            subtotal,
            tax,
            total_amount,
            currency,
            status,
            file_format,
            extraction_method,
            file_name
        FROM invoices
        ORDER BY invoice_id
    """)

    a, b, c = st.columns(3)

    vendors = a.multiselect(
        "Vendor",
        sorted(invoices["vendor"].unique())
    )

    categories = b.multiselect(
        "Category",
        sorted(invoices["category"].unique())
    )

    statuses = c.multiselect(
        "Payment Status",
        sorted(invoices["status"].unique())
    )

    search = st.text_input(
        "Search invoice ID or vendor",
        placeholder="Example: INV-00001"
    )

    filtered = invoices.copy()

    if vendors:

        filtered = filtered[
            filtered["vendor"].isin(vendors)
        ]

    if categories:

        filtered = filtered[
            filtered["category"].isin(categories)
        ]

    if statuses:

        filtered = filtered[
            filtered["status"].isin(statuses)
        ]

    if search:

        mask = (
            filtered["invoice_id"].str.contains(
                search,
                case=False,
                regex=False,
                na=False
            )
            |
            filtered["vendor"].str.contains(
                search,
                case=False,
                regex=False,
                na=False
            )
        )

        filtered = filtered[mask]

    a, b, c = st.columns(3)

    a.metric(
        "Matching Invoices",
        len(filtered)
    )

    b.metric(
        "Matching Invoice Value",
        money(filtered["total_amount"].sum())
    )

    c.metric(
        "Matching Vendors",
        filtered["vendor"].nunique()
    )

    st.subheader("Invoice Records")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        height=500,
        column_config={
            "invoice_id": "Invoice ID",
            "vendor": "Vendor",
            "invoice_date": "Invoice Date",
            "due_date": "Due Date",
            "category": "Category",
            "subtotal": st.column_config.NumberColumn(
                "Subtotal",
                format="$%.2f"
            ),
            "tax": st.column_config.NumberColumn(
                "Tax",
                format="$%.2f"
            ),
            "total_amount": st.column_config.NumberColumn(
                "Total Amount",
                format="$%.2f"
            ),
            "status": "Payment Status",
            "file_format": "File Format",
            "extraction_method": "Extraction Method",
            "file_name": "Source File"
        }
    )

    st.download_button(
        "⬇ Download Filtered Invoices",
        filtered.to_csv(index=False),
        file_name="filtered_invoices.csv",
        mime="text/csv"
    )


# ==================================================
# PAGE 3: FINANCIAL ANALYTICS
# ==================================================

elif page == "Financial Analytics":

    page_header(
        "Financial Analytics",
        "DuckDB-powered vendor, category and payment analysis."
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Total Invoice Value",
        money(kpis["total_invoice_value"])
    )

    b.metric(
        "Average Invoice",
        money(kpis["average_invoice_value"])
    )

    c.metric(
        "Total Tax",
        money(kpis["total_tax"])
    )

    d.metric(
        "Outstanding Amount",
        money(
            kpis["pending_amount"]
            + kpis["overdue_amount"]
        )
    )

    vendors = query(
        "SELECT * FROM vendor_summary"
    )

    categories = query(
        "SELECT * FROM category_summary"
    )

    monthly = query("""
        SELECT *
        FROM monthly_summary
        ORDER BY invoice_month
    """)

    left, right = st.columns(2)

    with left:

        fig = px.bar(
            vendors,
            x="vendor",
            y=[
                "paid_amount",
                "outstanding_amount"
            ],
            barmode="stack",
            title="Paid vs Outstanding by Vendor",
            color_discrete_sequence=[
                "#16a34a",
                "#dc2626"
            ],
            labels={
                "vendor": "Vendor",
                "value": "Amount",
                "variable": "Payment"
            }
        )

        fig.update_yaxes(
            tickprefix="$"
        )

        show_chart(fig)

    with right:

        fig = px.pie(
            categories,
            names="category",
            values="total_spend",
            hole=0.5,
            title="Spending by Category",
            color_discrete_sequence=COLORS
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent"
        )

        show_chart(fig)

    fig = px.line(
        monthly,
        x="invoice_month",
        y=[
            "total_spend",
            "total_tax"
        ],
        markers=True,
        title="Monthly Financial Trends",
        color_discrete_sequence=[
            "#2563eb",
            "#0891b2"
        ],
        labels={
            "invoice_month": "Month",
            "value": "Amount",
            "variable": "Metric"
        }
    )

    fig.update_traces(
        line=dict(width=3)
    )

    fig.update_yaxes(
        tickprefix="$"
    )

    show_chart(fig)

    st.subheader("Vendor Financial Summary")

    st.dataframe(
        vendors,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Category Summary")

    st.dataframe(
        categories,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇ Download Vendor Summary",
        vendors.to_csv(index=False),
        file_name="vendor_summary.csv",
        mime="text/csv"
    )


# ==================================================
# PAGE 4: DATA QUALITY
# ==================================================

elif page == "Data Quality":

    page_header(
        "Data Quality & Quarantine",
        "Monitor validation rules, rejected documents and reconciliation."
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Quality Score",
        f"{kpis['data_quality_score']:.1f}%"
    )

    b.metric(
        "Accepted Documents",
        int(kpis["valid_invoices"])
    )

    c.metric(
        "Rejected Documents",
        int(kpis["rejected_invoices"])
    )

    d.metric(
        "Extraction Failures",
        silver.get("extraction_failures", 0)
    )

    quality = query(
        "SELECT * FROM quality_summary"
    )

    formats = query(
        "SELECT * FROM format_summary"
    )

    left, right = st.columns(2)

    with left:

        fig = px.bar(
            quality,
            x="affected_documents",
            y="quality_issue",
            orientation="h",
            color="affected_documents",
            title="Data Quality Issues",
            color_continuous_scale="Reds",
            text="affected_documents",
            labels={
                "affected_documents": "Documents",
                "quality_issue": "Validation Issue"
            }
        )

        fig.update_layout(
            coloraxis_showscale=False
        )

        show_chart(fig)

    with right:

        fig = px.bar(
            formats,
            x="file_format",
            y="quality_score",
            color="file_format",
            title="Quality Score by File Format",
            color_discrete_sequence=[
                "#2563eb",
                "#0891b2",
                "#7c3aed"
            ],
            text="quality_score",
            labels={
                "file_format": "File Format",
                "quality_score": "Quality Score (%)"
            }
        )

        fig.update_yaxes(
            range=[0, 110]
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        show_chart(fig)

    st.info(
        "A rejected document may contain multiple "
        "validation issues. Quality score measures "
        "the proportion of accepted documents."
    )

    st.subheader("Quarantined Documents")

    rejected = query("""
        SELECT
            document_id,
            file_name,
            file_format,
            vendor,
            total_amount,
            status,
            rejection_reason
        FROM rejected_invoices
    """)

    st.dataframe(
        rejected,
        use_container_width=True,
        hide_index=True,
        height=380
    )

    st.download_button(
        "⬇ Download Quarantine Report",
        rejected.to_csv(index=False),
        file_name="quarantine_report.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("Financial Reconciliation")

    checks = gold.get(
        "reconciliation", {}
    )

    columns = st.columns(2)

    for index, (check, result) in enumerate(
        checks.items()
    ):

        label = check.replace(
            "_", " "
        ).title()

        with columns[index % 2]:

            if result == "PASS":

                st.success(
                    f"✓ {label}: PASS"
                )

            else:

                st.error(
                    f"{label}: FAIL"
                )


# ==================================================
# PAGE 5: AI DOCUMENT ASSISTANT
# ==================================================

elif page == "AI Document Assistant":

    page_header(
        "AI Document Assistant",
        "Retrieval-augmented generation powered by local AI models."
    )

    a, b, c = st.columns(3)

    a.metric(
        "Indexed Documents",
        vector.get("total_documents", 0)
    )

    b.metric(
        "Embedding Dimensions",
        vector.get("embedding_dimensions", 0)
    )

    c.metric(
        "Language Model",
        "Llama 3.1"
    )

    st.info(
        "The assistant searches validated invoices "
        "and generates answers from retrieved context. "
        "For complete financial totals, use "
        "Financial Analytics."
    )

    examples = [
        "",
        "Find overdue invoices related to cloud services.",
        "Find invoices related to software purchases.",
        "Show invoices from Atlas Infrastructure.",
        "Summarize pending invoice payments."
    ]

    def update_question():

        st.session_state["ai_question"] = (
            st.session_state["example_question"]
        )

    st.selectbox(
        "Example Questions",
        examples,
        key="example_question",
        on_change=update_question
    )

    question = st.text_area(
        "Ask a question about your documents",
        placeholder=(
            "Which retrieved cloud service "
            "invoices are overdue?"
        ),
        height=120,
        key="ai_question"
    )

    top_k = st.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button(
        "✨ Generate AI Answer",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "Searching documents and "
                    "generating AI answer. "
                    "Local inference may take a minute..."
                ):

                    st.session_state["rag_result"] = (
                        ask_question(
                            question,
                            top_k
                        )
                    )

            except Exception as error:

                st.session_state.pop(
                    "rag_result", None
                )

                st.error(
                    f"AI request failed: {error}"
                )

    result = st.session_state.get(
        "rag_result"
    )

    if result:

        st.divider()

        st.subheader("AI Generated Answer")

        st.caption(
            f"Question: {result['question']}"
        )

        if result["citation_validation"] == "FAIL":

            st.warning(
                "The answer contains unsupported "
                "invoice references. Verify it "
                "against the retrieved sources."
            )

        elif (
            result["citation_validation"]
            == "NO_CITATIONS"
        ):

            st.warning(
                "The answer does not contain "
                "invoice citations."
            )

        with st.container(border=True):

            st.markdown(
                result["answer"]
            )

        a, b, c = st.columns(3)

        a.metric(
            "Retrieved Documents",
            result["retrieved_documents"]
        )

        b.metric(
            "Citation Validation",
            result["citation_validation"]
        )

        c.metric(
            "Response Time",
            f"{result['execution_time_seconds']}s"
        )

        st.caption(
            "Citation validation checks referenced "
            "invoice IDs, not complete factual accuracy."
        )

        st.subheader("Retrieved Source Documents")

        source_table = pd.DataFrame([
            {
                "Invoice ID": source["invoice_id"],
                "Vendor": source["vendor"],
                "Category": source["category"],
                "Status": source["status"],
                "Amount": source["total_amount"],
                "Similarity": source[
                    "similarity_score"
                ]
            }
            for source in result["sources"]
        ])

        st.dataframe(
            source_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Amount": st.column_config.NumberColumn(
                    "Amount",
                    format="$%.2f"
                ),
                "Similarity": st.column_config.NumberColumn(
                    "Similarity",
                    format="%.4f"
                )
            }
        )

        for source in result["sources"]:

            label = (
                f"{source['invoice_id']} | "
                f"{source['vendor']} | "
                f"Similarity: "
                f"{source['similarity_score']:.4f}"
            )

            with st.expander(label):

                st.text(
                    source["document_text"]
                )

                st.caption(
                    f"Source: {source['source_path']}"
                )

                st.progress(
                    max(
                        0.0,
                        min(
                            1.0,
                            source["similarity_score"]
                        )
                    )
                )

        st.download_button(
            "⬇ Download AI Analysis",
            json.dumps(
                result,
                indent=4
            ),
            file_name="ai_analysis.json",
            mime="application/json"
        )


# ==================================================
# PAGE 6: PIPELINE MONITORING
# ==================================================

elif page == "Pipeline Monitoring":

    page_header(
        "Pipeline Monitoring",
        "Track execution performance, caching, logs and pipeline history."
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Pipeline Status",
        pipeline.get(
            "status", "UNKNOWN"
        )
    )

    b.metric(
        "Successful Stages",
        pipeline.get(
            "successful_stages", 0
        )
    )

    c.metric(
        "Failed Stages",
        pipeline.get(
            "failed_stages", 0
        )
    )

    d.metric(
        "Total Runtime",
        f"{pipeline.get('execution_time_seconds', 0)}s"
    )

    st.subheader(
        "Latest Pipeline Execution"
    )

    stages = pd.DataFrame(
        pipeline.get("stages", [])
    )

    if not stages.empty:

        st.dataframe(
            stages[
                [
                    "stage",
                    "status",
                    "execution_time_seconds"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            stages,
            x="stage",
            y="execution_time_seconds",
            color="status",
            color_discrete_map=STATUS_COLORS,
            text="execution_time_seconds",
            title="Stage Execution Performance",
            labels={
                "stage": "Pipeline Stage",
                "execution_time_seconds": "Seconds",
                "status": "Status"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.2f}s",
            textposition="outside"
        )

        show_chart(fig)

    st.divider()

    st.subheader(
        "Incremental Processing & AI Cache"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "New Documents",
        bronze.get(
            "new_documents", 0
        )
    )

    b.metric(
        "Skipped Documents",
        bronze.get(
            "skipped_documents", 0
        )
    )

    c.metric(
        "Cached AI Documents",
        silver.get(
            "cached_documents", 0
        )
    )

    d.metric(
        "New AI Extractions",
        silver.get(
            "ai_processed", 0
        )
    )

    cached = silver.get(
        "cached_documents", 0
    )

    processed = silver.get(
        "ai_processed", 0
    )

    total_pdf = cached + processed

    cache_rate = (
        cached / total_pdf * 100
        if total_pdf else 0
    )

    st.markdown(
        f"**AI Cache Hit Rate: {cache_rate:.1f}%**"
    )

    st.progress(
        min(
            1.0,
            cache_rate / 100
        )
    )

    st.divider()

    st.subheader(
        "Pipeline Execution History"
    )

    history = load_history()

    if not history.empty:

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        fig = px.line(
            history,
            x="Started At",
            y="Runtime",
            markers=True,
            title="Pipeline Runtime History",
            color_discrete_sequence=[
                "#2563eb"
            ]
        )

        fig.update_traces(
            line=dict(width=3)
        )

        show_chart(fig)

    else:

        st.info(
            "No pipeline execution history available."
        )

    st.subheader("Pipeline Logs")

    log_file = (
        BASE_DIR / "logs/pipeline.log"
    )

    if log_file.exists():

        lines = log_file.read_text(
            encoding="utf-8"
        ).splitlines()

        st.code(
            "\n".join(lines[-40:]),
            language="text"
        )

        st.download_button(
            "⬇ Download Pipeline Logs",
            log_file.read_bytes(),
            file_name="pipeline.log",
            mime="text/plain"
        )

    else:

        st.info(
            "No pipeline logs available."
        )


# ==================================================
# PAGE 7: SYSTEM ARCHITECTURE
# ==================================================

elif page == "System Architecture":

    page_header(
        "System Architecture",
        "End-to-end architecture of the AI Document Data Pipeline."
    )

    st.graphviz_chart("""
    digraph Pipeline {

        rankdir=TB;

        bgcolor="#ffffff";

        pad=0.4;

        nodesep=0.5;

        ranksep=0.6;

        node [
            shape=box,
            style="rounded,filled",
            fillcolor="#eff6ff",
            color="#2563eb",
            fontcolor="#1e293b",
            fontname="Arial",
            fontsize=13,
            penwidth=1.5,
            margin=0.2
        ];

        edge [
            color="#64748b",
            fontcolor="#334155",
            fontname="Arial",
            fontsize=11
        ];

        Raw [
            label="Raw Documents\\nPDF | JSON | CSV",
            fillcolor="#f1f5f9",
            color="#64748b"
        ];

        Bronze [
            label="Bronze Layer\\nIncremental Ingestion",
            fillcolor="#fff7ed",
            color="#c2410c"
        ];

        Silver [
            label="Silver Layer\\nLlama AI Extraction",
            fillcolor="#f1f5f9",
            color="#64748b"
        ];

        Quality [
            label="Data Quality Validation",
            fillcolor="#eff6ff",
            color="#2563eb"
        ];

        Quarantine [
            label="Quarantine\\nRejected Documents",
            fillcolor="#fef2f2",
            color="#dc2626"
        ];

        Gold [
            label="Gold Layer\\nDuckDB Analytics",
            fillcolor="#fefce8",
            color="#ca8a04"
        ];

        Vector [
            label="Vector Store\\n768-D Embeddings",
            fillcolor="#f5f3ff",
            color="#7c3aed"
        ];

        RAG [
            label="RAG Engine\\nSemantic Search + LLM",
            fillcolor="#ecfeff",
            color="#0891b2"
        ];

        UI [
            label="Streamlit Dashboard",
            fillcolor="#f0fdf4",
            color="#16a34a"
        ];

        Raw -> Bronze;

        Bronze -> Silver;

        Silver -> Quality;

        Quality -> Quarantine [
            label="Rejected"
        ];

        Quality -> Gold [
            label="Validated"
        ];

        Quality -> Vector [
            label="Validated"
        ];

        Vector -> RAG;

        Gold -> UI;

        Quarantine -> UI;

        RAG -> UI;
    }
    """)

    st.divider()

    st.subheader("Technology Stack")

    stack = pd.DataFrame([
        ["Programming", "Python"],
        ["Document Processing", "PyMuPDF"],
        ["Data Processing", "Pandas, NumPy"],
        ["Storage", "Apache Parquet"],
        ["Analytics Warehouse", "DuckDB"],
        ["Local LLM", "Llama 3.1 8B"],
        ["Embedding Model", "Nomic Embed Text"],
        ["Vector Storage", "NumPy"],
        ["Retrieval", "Cosine Similarity"],
        ["AI Framework", "Custom RAG"],
        ["Model Serving", "Ollama"],
        ["Orchestration", "Python"],
        ["Visualization", "Streamlit, Plotly"],
        ["Monitoring", "JSON Reports, Logging"],
        ["Testing Framework", "Pytest"]
    ], columns=[
        "Component",
        "Technology"
    ])

    st.dataframe(
        stack,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Engineering Features")

    features = [
        "Medallion architecture",
        "SHA-256 content-based ingestion",
        "Incremental document processing",
        "Local LLM document extraction",
        "Content-based AI inference caching",
        "Data quality validation",
        "Quarantine management",
        "Financial reconciliation",
        "Vector embeddings",
        "Retrieval-augmented generation",
        "Invoice citation validation",
        "Dependency-aware orchestration",
        "Centralized logging",
        "Pipeline execution history"
    ]

    left, right = st.columns(2)

    for index, feature in enumerate(features):

        with left if index % 2 == 0 else right:

            st.markdown(
                f"✓ {feature}"
            )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "AI Document Intelligence Platform | "
    "Data Engineering & Generative AI Portfolio Project | "
    "Synthetic Data"
)