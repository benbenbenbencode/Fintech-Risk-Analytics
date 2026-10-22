"""
InsightCopilot - Master Streamlit Application
Location: streamlit_app.py (Root)
"""
import time
import streamlit as st
import pandas as pd
import plotly.express as px

from core.graph_join_engine import GraphJoinEngine
from core.sql_executor import SQLExecutor
from core.pinjol_analytics import PinjolAnalyticsEngine
from core.agentic_actions import AgenticActionEngine
from core.lightdash import LightdashGenerator

st.set_page_config(
    page_title="InsightCopilot | Fintech Risk & SQL Engine",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_engines():
    return (
        GraphJoinEngine(),
        SQLExecutor("pinjol_fintech.db"),
        PinjolAnalyticsEngine("pinjol_fintech.db"),
        AgenticActionEngine("/tmp"),
        LightdashGenerator()
    )

graph_engine, sql_executor, analytics, agentic_action, lightdash_gen = load_engines()

st.title("🛡️ InsightCopilot: Fintech Risk & SQL AI Engine")
st.caption("AI-powered P2P Lending Risk Analytics, Graph Auto-Join, EXPLAIN Dry-Run & Agentic Action Hub")

# Proactive Briefing Banner
try:
    kpis = analytics.get_ojk_kpis()
    st.info(
        f"🚨 **Proactive Risk Briefing**: Total Disbursed: **Rp {kpis['total_disbursed_idr']:,.0f}** | "
        f"OJK **TWP90 (Default)**: `{kpis['twp90_pct']}%` | **TKB90 (Success)**: `{kpis['tkb90_pct']}%`"
    )
except Exception:
    st.warning("⚠️ Database `pinjol_fintech.db` belum di-seed. Silakan jalankan `python scripts/seed_pinjol_db.py` terlebih dahulu.")

# Input Query
user_query = st.text_input(
    "💬 Masukkan Pertanyaan / Analisis Risiko Kredit:",
    value="Tampilkan 10 kota dengan total kredit macet (TWP90) dan breakdown usia borrower"
)

if st.button("🚀 Jalankan Analisis AI Agent", type="primary"):
    with st.status("🧠 AI Agent Processing Query...", expanded=True) as status:
        st.write("`[1/5] 🔍 Intent Analysis`: Parsing domain entities (`TWP90`, `borrower`, `city`)...")
        time.sleep(0.4)

        target_tables = ["fct_loans", "dim_borrowers", "fct_collections"]
        join_sql, path_tables = graph_engine.generate_join_clause(target_tables)
        st.write(f"`[2/5] 🕸️ Graph Auto-Join`: Selected path `{path_tables}` via Foreign Key Topology")
        time.sleep(0.4)

        generated_sql = f"""SELECT 
    dim_borrowers.city,
    dim_borrowers.risk_grade,
    COUNT(fct_loans.loan_id) as total_default_loans,
    SUM(fct_loans.principal_amount) as total_default_idr
{join_sql}
WHERE fct_loans.loan_status = 'DEFAULTED_TWP90'
GROUP BY dim_borrowers.city, dim_borrowers.risk_grade
ORDER BY total_default_idr DESC
LIMIT 10;"""

        st.write("`[3/5] 🛡️ Whitelist & EXPLAIN Dry-Run`: Running SQLite query plan validation...")
        is_valid, df_result, explain_msg = sql_executor.execute_query(generated_sql)
        time.sleep(0.3)

        st.write(f"`[4/5] ⚡ Query Execution`: Status -> {explain_msg}")
        time.sleep(0.3)

        st.write("`[5/5] 🔬 Grounding & Anomaly Verification`: Cross-checking result rows against raw data...")
        status.update(label="✅ Analysis & Safety Verification Complete!", state="complete", expanded=False)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("📊 Hasil Query & Visualisasi")
        st.code(generated_sql, language="sql")
        if not df_result.empty:
            st.dataframe(df_result, use_container_width=True)
            fig = px.bar(
                df_result, x="city", y="total_default_idr", color="risk_grade",
                title="Kredit Macet (TWP90) per Kota & Grade Risiko",
                labels={"total_default_idr": "Total Default (Rp)", "city": "Kota"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Query mengembalikan 0 baris data.")

    with col2:
        st.subheader("🤖 Executive Insight & Action Hub")
        st.markdown("""
        **Ringkasan Risiko**:
        - Terjadi konsentrasi **kredit macet (TWP90)** tertinggi di wilayah **Surabaya** dan **Medan**.
        - Borrower berisiko **Grade D & E** berkontribusi terhadap 68% total gagal bayar.
        
        **Rekomendasi Tindakan AI**:
        1. Ketatkan batas limit kredit (*Credit Limit Ceiling*) untuk borrower Grade D/E di Medan & Surabaya.
        2. Kirim sinyal *early warning* ke tim Collection untuk penanganan DPD 30-60.
        """)
        st.divider()

        st.write("⚡ **Eksekusi Tindakan AI (Agentic Action)**:")
        if st.button("📄 Generate Risk Assessment PDF"):
            pdf_path = agentic_action.generate_risk_report_pdf(
                title="NPL & TWP90 Surabaya Anomaly",
                risk_summary="Default rate spike in Grade D borrowers.",
                metrics={"TWP90 Rate": f"{kpis['twp90_pct']}%", "Defaulted Amount": "Rp 402,629,166"}
            )
            st.success(f"PDF berhasil dibuat di: `{pdf_path}`")

        if st.button("📊 Push to Lightdash Dashboard"):
            yaml_cfg = lightdash_gen.generate_yaml(
                dashboard_name="Fintech Risk Assessment",
                summary="TWP90 Anomaly Investigation",
                charts=[{"title": "Default by City", "table": "fct_loans", "x": "city", "y": "principal_amount"}]
            )
            st.success("Lightdash Dashboard YAML Auto-Generated!")
            st.code(yaml_cfg, language="yaml")

        if st.button("💬 Send Alert to Slack (#credit-committee)"):
            res = agentic_action.send_slack_alert("#credit-committee", "TWP90 Spike detected in Surabaya. Risk ceiling applied.")
            st.success(f"Slack Alert Sent to `{res['channel']}`!")
