"""CSV Lab: bring your own loan tables, let the AI map the schema, build a scorecard.   Run:  streamlit run lab_app.py"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from crf import ai
from crf.pipeline import run_pipeline
from studio import ui

ROOT = Path(__file__).parent
st.set_page_config(page_title="Credit Risk Lab (CSV)", page_icon="🧪", layout="wide")
ui.inject_css()
ui.sidebar_ai()
st.sidebar.radio("Data source", ["Demo loans", "Upload my CSVs"], key="lab_mode")
st.sidebar.markdown("**Overrides**")
o = {"type": st.sidebar.selectbox("Scorecard type", ["behavioral", "application"], key="o_type"),
     "dpd": st.sidebar.number_input("Bad = DPD ≥", 1, 360, 90, key="o_dpd"),
     "obs": st.sidebar.number_input("Observation months", 1, 12, 3, key="o_obs"),
     "perf": st.sidebar.number_input("Performance months", 1, 24, 9, key="o_perf"),
     "test": st.sidebar.slider("Out-of-time share", 0.1, 0.5, 0.3, 0.05, key="o_test")}
use_over = st.sidebar.checkbox("Apply overrides", True, key="use_over")
llm = ui.get_llm()
ui.header("Credit Risk Lab (CSV)", "Upload loan, repayment, customer and bureau CSVs with any column names. The AI maps them and drafts the config.", "🧪")
if "cfg_text" not in st.session_state:
    st.session_state["cfg_text"] = (ROOT / "config.yaml").read_text()

setup, results = st.tabs(["Setup", "Results"])
with setup:
    if st.session_state["lab_mode"] == "Demo loans":
        if not (ROOT / "data" / "loans.csv").exists():
            from generate_sample_data import generate
            generate(6000, out_dir=str(ROOT / "data"))
        st.info("Demo: synthetic installment loans with a deliberately leaky field (`collection_status`) that the leakage checker should catch.")
        if st.button("Reset YAML to demo config", key="reset_cfg"):
            st.session_state["cfg_text"] = (ROOT / "config.yaml").read_text()
    else:
        files = st.file_uploader("CSV files", type="csv", accept_multiple_files=True, key="uploads")
        if st.button("🤖 Detect schema & draft config", key="detect", disabled=not files):
            tmp = Path(st.session_state.setdefault("tmp_dir", tempfile.mkdtemp(prefix="crf_")))
            raw, paths = {}, {}
            for f in files:
                df = pd.read_csv(f)
                df.to_csv(tmp / f.name, index=False)
                raw[f.name], paths[f.name] = df, str(tmp / f.name)
            try:
                out = ai.suggest_config(raw, paths, llm, st.session_state.get("send_samples", True))
                st.session_state["cfg_text"] = yaml.safe_dump(out["config"], sort_keys=False)
                st.session_state["mapper_out"] = out
            except ValueError as e:
                st.error(str(e))
        mo = st.session_state.get("mapper_out")
        if mo:
            st.success(f"Config drafted via **{mo['source']}** mapping. Review before running.")
            for n in mo["notes"]:
                st.write(f"- {n}")
    with st.expander("Config (YAML)", expanded=st.session_state["lab_mode"] != "Demo loans"):
        st.text_area("config", key="cfg_text", height=380, label_visibility="collapsed")
    if st.button("▶ Run pipeline", type="primary", key="run"):
        try:
            cfg = yaml.safe_load(st.session_state["cfg_text"])
            if use_over:
                cfg["scorecard_type"] = o["type"]
                cfg.setdefault("target", {}).update({"bad_dpd": int(o["dpd"]), "observation_months": int(o["obs"]), "performance_months": int(o["perf"])})
                cfg.setdefault("split", {})["test_fraction"] = float(o["test"])
            with st.spinner("Building target, features, scorecard..."):
                st.session_state["res"] = run_pipeline(cfg, base_dir=ROOT)
            for k in ("review", "answer"):
                st.session_state.pop(k, None)
            st.success("Done. Open the Results tab.")
        except Exception as e:
            st.session_state.pop("res", None)
            st.error(f"{type(e).__name__}: {e}")

res = st.session_state.get("res")
with results:
    if res is None:
        st.info("Run the pipeline in Setup first.")
    else:
        ts = res["target_summary"]
        c = st.columns(4)
        c[0].metric("Modelling loans", f"{ts['n_model']:,}")
        c[1].metric("Bad rate", ui.pct(ts["bad_rate"]))
        c[2].metric("Train / test", f"{ts['n_train']:,} / {ts['n_test']:,}")
        c[3].metric("Bad definition", f"DPD ≥ {ts['bad_dpd']} in {ts['perf']}m" + (f" after {ts['obs']}m" if ts["obs"] else ""))
        for w in res["warnings"]:
            st.warning(w)
        a, b = st.columns(2)
        wf = res["waterfall"]
        with a:
            ui.show(go.Figure(go.Bar(x=wf["remaining"], y=wf["step"], orientation="h")).update_layout(title="Population waterfall", yaxis=dict(autorange="reversed")), 280)
        with b:
            v = res["vintage"]
            fig = go.Figure()
            fig.add_bar(x=v["month"], y=v["loans"], name="Loans", opacity=0.5)
            fig.add_scatter(x=v["month"], y=v["bad_rate"], name="Bad rate", yaxis="y2")
            fig.update_layout(title="Vintage", yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
            ui.show(fig, 280)
        ui.model_charts(res, key="lab")
        st.markdown("##### AI analyst")
        if st.button("Generate model review", key="gen_review"):
            st.session_state["review"] = ai.model_review(res, llm)
        if "review" in st.session_state:
            txt, src = st.session_state["review"]
            st.caption(f"source: {src}")
            st.markdown(txt)
        q = st.text_input("Ask about this run", key="ask_q")
        if st.button("Ask", key="ask_btn", disabled=not q):
            st.session_state["answer"] = ai.ask_results(q, res, llm)
        if "answer" in st.session_state:
            st.markdown(st.session_state["answer"])
        cfg_out = {k: v for k, v in res["cfg"].items() if not k.startswith("_")}
        st.download_button("Scorecard points (CSV)", res["points_table"].to_csv(index=False), "scorecard_points.csv", "text/csv")
        st.download_button("Config used (YAML)", yaml.safe_dump(cfg_out, sort_keys=False), "config_used.yaml", "text/yaml")
