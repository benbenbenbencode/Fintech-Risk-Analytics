"""Universal Credit Studio.

    streamlit run auto_app.py

Four questions, in order:
  1. What did the system find in this data?      -> Profile tab
  2. Which model did it choose, and why?         -> Models tab
  3. Why did this applicant get this decision?   -> Explain tab
  4. Is the decision fair across groups?         -> Fairness tab

Everything comes from one `run_auto()` call; no feature list is written by hand.
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crf.auto_pipeline import AutoResult, run_auto
from crf.ingest import (
    COLUMN_SPEC,
    derive_loans,
    required_columns,
    suggest_as_of,
    template_csv,
    template_frame,
    validate_tables,
)

EXAMPLES = Path(__file__).resolve().parent / "examples"

ROOT = Path(__file__).resolve().parent
ACCENT = "#6c5ce7"
BAD = "#e74c3c"
GOOD = "#00b894"

st.set_page_config(page_title="Universal Credit Studio", page_icon="🏦", layout="wide")

CSS = """
<style>
  .block-container {padding-top: 1.6rem; max-width: 1350px;}
  .uc-card {background:#f8f8fc;border:1px solid #e6e6ee;border-radius:12px;padding:14px 16px;}
  .uc-card .lab {font-size:.72rem;letter-spacing:.07em;text-transform:uppercase;color:#7a7a88;}
  .uc-card .val {font-size:1.5rem;font-weight:650;color:#181820;}
  .uc-band {display:inline-block;padding:2px 12px;border-radius:999px;font-weight:700;color:#fff;}
  div[data-testid="stMetricValue"] {font-size:1.4rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def card(column, label: str, value: str) -> None:
    with column:
        st.markdown(
            f"<div class='uc-card'><div class='lab'>{label}</div>"
            f"<div class='val'>{value}</div></div>",
            unsafe_allow_html=True,
        )


def band_colour(band: str) -> str:
    return {"A": "#00b894", "B": "#2ecc71", "C": "#f1c40f", "D": "#e67e22", "E": "#e74c3c"}.get(band, "#7f8c8d")


def fig_roc(roc: dict) -> go.Figure:
    figure = go.Figure()
    for name, points in roc.items():
        figure.add_trace(go.Scatter(x=points["fpr"], y=points["tpr"], mode="lines", name=name))
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random",
                                line=dict(dash="dash", color="#bbb")))
    figure.update_layout(title="ROC (out-of-time test)", xaxis_title="false positive rate",
                         yaxis_title="true positive rate", height=380,
                         margin=dict(l=10, r=10, t=45, b=10))
    return figure


def fig_auc_bars(leaderboard: pd.DataFrame, chosen: str) -> go.Figure:
    frame = leaderboard.sort_values("test_auc")
    colours = [ACCENT if name == chosen else "#c9c9d6" for name in frame["model"]]
    figure = go.Figure(go.Bar(
        x=frame["test_auc"], y=frame["label"], orientation="h", marker_color=colours,
        text=[f"{v:.3f}" for v in frame["test_auc"]], textposition="outside",
    ))
    figure.update_layout(title="Out-of-time AUC by model (highlighted = selected)",
                         xaxis_range=[max(0.5, frame["test_auc"].min() - 0.05), 1.0],
                         height=320, margin=dict(l=10, r=30, t=45, b=10))
    return figure


def fig_calibration(deciles: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=deciles["avg_pd"], y=deciles["bad_rate"], mode="markers+lines",
                                name="observed", marker=dict(size=10, color=ACCENT)))
    limit = float(max(deciles["avg_pd"].max(), deciles["bad_rate"].max())) * 1.05
    figure.add_trace(go.Scatter(x=[0, limit], y=[0, limit], mode="lines", name="perfect",
                                line=dict(dash="dash", color="#bbb")))
    figure.update_layout(title="Calibration by risk decile", xaxis_title="predicted PD",
                         yaxis_title="observed bad rate", height=340,
                         margin=dict(l=10, r=10, t=45, b=10))
    return figure


def fig_deciles(deciles: pd.DataFrame) -> go.Figure:
    figure = go.Figure(go.Bar(x=deciles["decile"], y=deciles["bad_rate"], marker_color=ACCENT,
                              text=[f"{v:.1%}" for v in deciles["bad_rate"]], textposition="outside"))
    figure.update_layout(title="Bad rate by decile (1 = riskiest)", xaxis_title="decile",
                         yaxis_title="bad rate", yaxis_tickformat=".0%", height=340,
                         margin=dict(l=10, r=10, t=45, b=10))
    return figure


def fig_importance(importance: pd.DataFrame, top: int = 12) -> go.Figure:
    frame = importance.head(top).sort_values("importance")
    figure = go.Figure(go.Bar(x=frame["importance"], y=frame["feature"], orientation="h",
                              marker_color=ACCENT))
    figure.update_layout(title="What the model relies on (AUC drop when shuffled)",
                         height=380, margin=dict(l=10, r=10, t=45, b=10))
    return figure


def fig_contributions(explanation) -> go.Figure:
    frame = explanation.contributions.head(12).sort_values("delta_pp")
    colours = [BAD if v > 0 else GOOD for v in frame["delta_pp"]]
    figure = go.Figure(go.Bar(
        x=frame["delta_pp"], y=frame["label"], orientation="h", marker_color=colours,
        text=[f"{v:+.1f} pp" for v in frame["delta_pp"]], textposition="outside",
        customdata=frame["value"],
    ))
    figure.update_layout(title="Impact on this applicant's default probability",
                         xaxis_title="percentage points vs a safe value", height=420,
                         margin=dict(l=10, r=40, t=45, b=10))
    return figure


def fig_score_gauge(explanation) -> go.Figure:
    if explanation.score is None:
        value, title = explanation.pd * 100, "Probability of default (%)"
        maximum = 100
    else:
        value, title = explanation.score, "Score"
        maximum = 850
    figure = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": f"{title} - band {explanation.band}"},
        gauge={
            "axis": {"range": [0, maximum]},
            "bar": {"color": band_colour(explanation.band)},
            "steps": [
                {"range": [0, maximum * 0.2], "color": "#eafaf1"},
                {"range": [maximum * 0.2, maximum * 0.4], "color": "#f7fbe9"},
                {"range": [maximum * 0.4, maximum * 0.6], "color": "#fef9e7"},
                {"range": [maximum * 0.6, maximum * 0.8], "color": "#fdf0e6"},
                {"range": [maximum * 0.8, maximum], "color": "#fdeaea"},
            ],
        },
    ))
    figure.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return figure


def fig_fairness(table: pd.DataFrame, group: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Bar(x=table["group"], y=table["approval_rate"], name="approval rate",
                            marker_color=ACCENT, text=[f"{v:.0%}" for v in table["approval_rate"]],
                            textposition="outside"))
    figure.add_trace(go.Scatter(x=table["group"], y=table["mean_pd"], name="mean PD", yaxis="y2",
                                mode="markers+lines", line=dict(color=BAD)))
    figure.update_layout(
        title=f"Approval rate and mean PD by {group}",
        yaxis=dict(tickformat=".0%"), yaxis2=dict(overlaying="y", side="right", tickformat=".1%"),
        height=340, margin=dict(l=10, r=10, t=45, b=10),
    )
    return figure


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

REQUIRED = {
    "loans": ["loan_id", "customer_id", "disbursement_date", "principal", "tenor_months"],
    "repayments": ["loan_id", "installment_no", "due_date", "paid_date", "due_amount"],
    "customers": ["customer_id"],
    "bureau": ["customer_id"],
}


def _normalise(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _map_columns(df: pd.DataFrame, required: list[str]) -> tuple[dict[str, str], list[str]]:
    lookup = {_normalise(c): c for c in df.columns}
    mapping, missing = {}, []
    for canonical in required:
        match = lookup.get(_normalise(canonical))
        if match:
            mapping[canonical] = match
        else:
            missing.append(canonical)
    return mapping, missing


def _format_table() -> pd.DataFrame:
    """The required format, as a table a business user can read."""
    rows = []
    for table, spec in COLUMN_SPEC.items():
        example = template_frame(table)
        for group, flag in (("required", "yes"), ("optional", "optional")):
            for column, (dtype, description) in spec.get(group, {}).items():
                sample = ""
                if column in example.columns and len(example):
                    sample = str(example[column].iloc[0])
                rows.append(
                    {
                        "table": table,
                        "column": column,
                        "type": dtype,
                        "required": flag,
                        "description": description,
                        "example": sample,
                    }
                )
    return pd.DataFrame(rows)


def _example_zip() -> bytes | None:
    paths = [EXAMPLES / f"{name}.csv" for name in ("loans", "repayments", "customers", "bureau")]
    paths.append(EXAMPLES / "config.yaml")
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in existing:
            archive.write(path, arcname=path.name)
    return buffer.getvalue()


def _render_report(report) -> None:
    st.markdown("**Format check**")
    icons = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
    for check in report.checks:
        line = f"{icons.get(check.status, '•')} **{check.name}** — {check.message}"
        if check.fix:
            line += f"  \n&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:#8a8a99'>{check.fix}</span>"
        st.markdown(line, unsafe_allow_html=True)

    if report.ok:
        st.success(report.summary())
    else:
        st.error(report.summary())


def _map_frame(df: pd.DataFrame, table: str) -> pd.DataFrame:
    mapping, missing = _map_columns(df, required_columns(table))
    if missing:
        st.warning(f"`{table}`: could not find {missing} — choose the matching column manually.")
        for column in missing:
            mapping[column] = st.selectbox(
                f"{table}.{column} ←", df.columns, key=f"map_{table}_{column}"
            )
    return df.rename(columns={raw: canonical for canonical, raw in mapping.items()})


def _upload_tables() -> tuple[dict, dict] | None:
    with st.expander("File format, example data and templates", expanded=False):
        st.caption(
            "Dates must be `YYYY-MM-DD`. Leave `paid_date` empty when an instalment was not paid. "
            "Extra columns are kept and can be used as features."
        )

        st.markdown("**Required format**")
        st.dataframe(_format_table(), hide_index=True, height=300)

        st.markdown("**Ready-to-run example dataset**")
        st.caption(
            "480 loans and 6,600 instalments with income, employment, region and a bureau snapshot. "
            "Download it, then upload the files below to see the whole pipeline work."
        )
        archive = _example_zip()
        if archive:
            st.download_button(
                "Download example dataset (ZIP, includes config.yaml)",
                data=archive,
                file_name="credit_risk_example.zip",
                mime="application/zip",
                key="dl_example_zip",
            )
            columns = st.columns(4)
            for column, table in zip(columns, ("loans", "repayments", "customers", "bureau")):
                path = EXAMPLES / f"{table}.csv"
                if path.exists():
                    with column:
                        st.download_button(
                            f"{table}.csv",
                            data=path.read_bytes(),
                            file_name=f"{table}.csv",
                            mime="text/csv",
                            key=f"dl_example_{table}",
                        )
        else:
            st.info("Example files are not present in this checkout; use the blank templates below.")

        st.markdown("**Blank templates**")
        columns = st.columns(3)
        for column, table in zip(columns, ("loans", "repayments", "customers")):
            with column:
                st.download_button(
                    f"{table}.csv template",
                    data=template_csv(table),
                    file_name=f"{table}_template.csv",
                    mime="text/csv",
                    key=f"tpl_{table}",
                )

    st.markdown(
        "**Repayment history is the core input.** The target (how late someone got) and the "
        "behavioural features both come from it. The loan table only supplies the things a repayment "
        "file cannot know: when money was disbursed, how much, and for how long."
    )

    mode = st.radio(
        "What do you have?",
        ["Repayment history only", "Loans + repayment history"],
        horizontal=True,
        key="upload_mode",
        help="Repayment history only: the loan table is derived (dates and amounts approximated). "
             "Loans + repayment history: more accurate, and lets you use income, purpose and region.",
    )

    folder = Path(st.session_state.setdefault("upload_dir", tempfile.mkdtemp(prefix="ucs_")))
    spec: dict = {}

    if mode == "Repayment history only":
        upload = st.file_uploader("Repayment history CSV", type="csv", key="upload_rep_only")
        if upload is None:
            st.info("Upload one CSV: one row per instalment, with `loan_id`, `due_date`, `paid_date` "
                    "and `due_amount`.")
            return None

        raw = pd.read_csv(upload)
        st.caption(f"{upload.name}: {len(raw):,} rows, {len(raw.columns)} columns")
        repayments = _map_frame(raw, "repayments")

        report = validate_tables({"repayments": repayments})
        _render_report(report)
        if not report.ok:
            return None

        loans, info = derive_loans(repayments)
        for note in info["notes"]:
            st.warning(note, icon="ℹ️")

        loans.to_csv(folder / "loans_derived.csv", index=False)
        repayments.to_csv(folder / "repayments.csv", index=False)
        st.session_state["suggested_as_of"] = suggest_as_of(repayments)

        spec = {
            "loans": {"path": str(folder / "loans_derived.csv"), "columns": {}},
            "repayments": {"path": str(folder / "repayments.csv"), "columns": {}},
        }
        return spec, info

    # ---- loans + repayments (+ optional extras) --------------------------
    uploads = st.file_uploader(
        "CSV files", type="csv", accept_multiple_files=True, key="upload_multi"
    )
    if not uploads:
        st.info("Upload at least two files: your loan table and your repayment history.")
        return None

    frames: dict[str, pd.DataFrame] = {}
    for item in uploads:
        frame = pd.read_csv(item)
        frames[item.name] = frame
        frame.to_csv(folder / item.name, index=False)

    st.caption("Files: " + ", ".join(f"{n} ({len(f):,} rows)" for n, f in frames.items()))

    defaults = {"loans": list(frames)[0], "repayments": list(frames)[min(1, len(frames) - 1)]}
    renamed: dict[str, pd.DataFrame] = {}

    for table in ("loans", "repayments", "customers", "bureau"):
        options = ["(none)"] + list(frames)
        default = defaults.get(table, "(none)")
        index = options.index(default) if default in options else 0
        name = st.selectbox(f"File for `{table}`", options, index=index, key=f"file_{table}")
        if name == "(none)":
            continue

        frame = _map_frame(frames[name], table)
        renamed[table] = frame
        frame.to_csv(folder / f"mapped_{table}.csv", index=False)
        spec[table] = {"path": str(folder / f"mapped_{table}.csv"), "columns": {}}

    if "loans" not in spec or "repayments" not in spec:
        st.warning("Both a loan table and a repayment history are required in this mode.")
        return None

    report = validate_tables(renamed)
    _render_report(report)
    if not report.ok:
        return None

    st.session_state["suggested_as_of"] = suggest_as_of(renamed.get("repayments"))
    return spec, {"rows": {k: len(v) for k, v in frames.items()}}


def load_result(cfg_source, tables=None, **kwargs) -> AutoResult:
    return run_auto(cfg_source, tables=tables, save_artifact=True, **kwargs)


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------


def page_start_here() -> None:
    st.markdown(
        "**This tool turns a table of past loans into a credit scoring model that can explain itself.** "
        "You do not have to tell it which columns to use: it works that out, tests several kinds of "
        "model, and reports what it chose and why."
    )

    columns = st.columns(3)
    card(columns[0], "Step 1", "Load loans and repayments")
    card(columns[1], "Step 2", "Set what counts as a bad loan")
    card(columns[2], "Step 3", "Press Run, then read the tabs")

    st.markdown("#### What each tab answers")
    st.markdown(
        "| Tab | The question it answers |\n"
        "|---|---|\n"
        "| **1 · Your data** | What did it find, what did it refuse to use, and why? |\n"
        "| **2 · Model quality** | Is the model any good, and why was this one chosen? |\n"
        "| **3 · Explaining a decision** | Why did this applicant get this answer? |\n"
        "| **4 · Fairness** | Are approval rates similar across groups of people? |\n"
    )

    with st.expander("How to read the main numbers (2-minute version)", expanded=True):
        st.markdown(
            "- **Probability of default (PD)** - the chance this loan goes bad. 3% means 3 in 100.\n"
            "- **Score and band** - the same risk on a 0-850 scale and an A-E band, where A is safest.\n"
            "- **AUC** - how well the model ranks riskier people above safer ones. 0.5 is a coin flip, "
            "0.65-0.75 is normal for credit, above 0.90 usually means leakage rather than genius.\n"
            "- **Calibration slope** - whether the percentages are honest. 1.00 is perfect, 0.9-1.1 is "
            "good; a model can rank well and still quote misleading percentages.\n"
            "- **DI ratio** - compares approval rates between groups. Below 0.80 raises a flag.\n"
        )

    with st.expander("What it deliberately refuses to use"):
        st.markdown(
            "- **Protected attributes** (gender, age, marital status, religion, ethnicity, nationality, "
            "disability). Excluded from scoring, then audited on the Fairness tab.\n"
            "- **Leaked outcomes** such as `collection_status`: filled in only after someone stops "
            "paying, so using it would flatter the model in testing and break it in production.\n"
            "- **Identifiers** (account numbers), constant columns, near-duplicates and columns that are "
            "mostly empty.\n"
        )

    st.info(
        "Full plain-English guide: `docs/BUSINESS_GUIDE.md` - worked example, glossary, and the "
        "questions people actually ask."
    )


def page_profile(result: AutoResult) -> None:
    summary = result.to_summary()
    columns = st.columns(4)
    card(columns[0], "Loans modelled", f"{summary['n_model']:,}")
    card(columns[1], "Bad rate", f"{summary['bad_rate']:.1%}")
    card(columns[2], "Features discovered", str(summary["n_features"]))
    card(columns[3], "Chosen model", summary["chosen"])

    st.markdown("#### What the system found")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Discovered features**")
        table = result.featureset.summary()
        st.dataframe(table, height=340)
        st.caption(f"{len(table)} features kept, each with its provenance (where it came from).")

    with right:
        st.markdown("**Rejected columns**")
        dropped = result.featureset.dropped
        if dropped.empty:
            st.info("Nothing was rejected.")
        else:
            st.dataframe(dropped.head(25), height=340)
            st.caption("Identifiers, protected attributes, leaked outcomes and free text are excluded by design.")

    st.markdown("**Relationships detected between tables**")
    if result.relationships:
        st.dataframe(pd.DataFrame(result.relationships))
    else:
        st.caption("No shared keys across tables; features came from the base table only.")

    st.markdown("**Population waterfall**")
    waterfall = result.waterfall
    figure = go.Figure(go.Bar(x=waterfall["remaining"], y=waterfall["step"], orientation="h",
                              marker_color=ACCENT))
    figure.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(figure)

    st.markdown("**Univariate strength (IV) and leakage screen**")
    st.dataframe(result.tournament.iv_table.head(20), height=300)


def page_models(result: AutoResult) -> None:
    leaderboard = result.tournament.leaderboard
    st.success(f"**Selected: {result.tournament.chosen}** — {result.tournament.reason}")

    columns = st.columns(4)
    chosen_row = leaderboard[leaderboard["model"] == result.tournament.chosen].iloc[0]
    card(columns[0], "AUC (out of time)", f"{chosen_row['test_auc']:.3f}")
    card(columns[1], "KS", f"{chosen_row['test_ks']:.3f}")
    card(columns[2], "Calibration slope", f"{chosen_row['test_calib_slope']:.2f}")
    card(columns[3], "Top-decile lift", f"{chosen_row['test_top_decile_lift']:.2f}x")
    st.caption(
        "**AUC** - how often the model ranks a real defaulter above a non-defaulter "
        "(0.5 = coin flip, 0.75+ = strong).  **KS** - another ranking measure, higher is better.  "
        "**Calibration slope** - whether the percentages are honest (1.00 = perfect).  "
        "**Top-decile lift** - how much riskier the riskiest 10% is compared with an average customer."
    )

    st.markdown("#### Tournament")
    display = leaderboard[["label", "test_auc", "test_ks", "test_gini", "test_brier",
                           "test_calib_slope", "test_top_decile_lift", "interpretable", "seconds"]]
    st.dataframe(display.round(4), height=220)
    st.caption("Interpretable = exposes point-level reason codes. Calibration slope near 1.0 means the "
               "predicted probabilities can be taken at face value.")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(fig_auc_bars(leaderboard, result.tournament.chosen))
    with right:
        st.plotly_chart(fig_roc(result.tournament.roc))

    left, right = st.columns(2)
    with left:
        st.plotly_chart(fig_calibration(result.deciles))
    with right:
        st.plotly_chart(fig_deciles(result.deciles))

    if not result.tournament.importance.empty:
        st.plotly_chart(fig_importance(result.tournament.importance))

    if not result.tournament.dropped_features.empty:
        with st.expander("Features excluded by the leakage screen"):
            st.dataframe(result.tournament.dropped_features)


def page_explain(result: AutoResult) -> None:
    language = st.radio("Language", ["en", "id"], horizontal=True, key="explain_lang")

    test_ids = list(result.tournament.test_ids)
    options = [str(i) for i in test_ids[:200]]
    default = st.session_state.get("explain_id", options[0] if options else None)
    if default not in options and options:
        default = options[0]

    left, right = st.columns([1, 2])
    with left:
        choice = st.selectbox("Applicant (loan_id)", options, index=options.index(default) if default else 0)
        st.caption("Held-out applicants only, so the model never saw them in training.")
    st.session_state["explain_id"] = choice

    explanation = result.explain(choice, language=language)
    if explanation is None:
        st.error("Could not explain this applicant.")
        return

    cutoff = result.fairness.get("cutoff")
    decision = "approve"
    if cutoff is not None:
        decision = "approve" if explanation.pd <= cutoff else "decline / manual review"

    columns = st.columns(4)
    card(columns[0], "Probability of default", f"{explanation.pd:.2%}")
    card(columns[1], "Score", f"{explanation.score:.0f}" if explanation.score else "n/a")
    card(columns[2], "Risk band", explanation.band)
    card(columns[3], "Recommended action", decision)

    left, right = st.columns([2, 3])
    with left:
        st.plotly_chart(fig_score_gauge(explanation))
    with right:
        st.markdown(
            f"<div class='uc-card'><div class='lab'>Why this decision</div>"
            f"<div style='white-space:pre-wrap;line-height:1.6;margin-top:6px'>{explanation.text}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Contribution of each factor")
    st.plotly_chart(fig_contributions(explanation))
    st.caption(
        "Each bar answers: how many percentage points of default risk does this factor add "
        "compared with its safest observed value? Red increases risk, green reduces it."
    )

    if explanation.reasons:
        st.markdown("**Adverse-action reasons (top factors)**")
        for item in explanation.reasons:
            st.markdown(f"- **{item['label']}** — {item['value']} (+{item['delta_pp']:.1f} pp of risk)")

    st.markdown("#### Applicant vs population")
    row = result.X.loc[choice]
    population = result.X
    comparison = pd.DataFrame(
        {
            "feature": [c for c in explanation.contributions["feature"]],
        }
    )
    comparison["applicant"] = [row.get(f) for f in comparison["feature"]]
    comparison["population median"] = [population[f].median() if result.featureset.types.get(f) == "numeric"
                                       else population[f].mode().iloc[0] if not population[f].mode().empty else None
                                       for f in comparison["feature"]]
    comparison["impact (pp)"] = explanation.contributions["delta_pp"].round(2).to_numpy()
    # Mixed numeric/categorical values in one column break Arrow serialisation.
    comparison["applicant"] = comparison["applicant"].astype(str)
    comparison["population median"] = comparison["population median"].astype(str)
    st.dataframe(comparison, height=280)

    with st.expander("Raw applicant record"):
        raw = result.applicant_frame([choice])
        record = raw.T.rename(columns={choice: "value"})
        # Arrow cannot serialise a mixed object column containing Timestamps.
        record["value"] = record["value"].astype(str)
        st.dataframe(record, height=320)


def page_fairness(result: AutoResult) -> None:
    fairness = result.fairness
    if not fairness.get("protected"):
        st.info("No protected attributes were found in the data (sex, age, marital status, religion, "
                "nationality, ethnicity, disability). Nothing to audit.")
        return

    st.caption(
        f"Decision rule: approve if PD ≤ {fairness['cutoff']:.3%} "
        f"({fairness['approval_rate']:.0%} of held-out applicants). "
        f"Detected attributes: {', '.join(fairness['protected'])}."
    )

    for flag in fairness.get("flags", []):
        st.warning(flag, icon="⚠️")
    if not fairness.get("flags"):
        st.success("No group falls below the 0.80 disparate-impact threshold.")

    for group, table in fairness["results"].items():
        ratio = float(table["di_ratio"].iloc[0])
        st.markdown(f"#### {group.replace('_', ' ').title()} — DI ratio {ratio:.3f}")
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(fig_fairness(table, group))
        with right:
            st.dataframe(
                table[["group", "applicants", "approval_rate", "mean_pd"]].round(4),
                height=300,
            )
            st.caption("DI ratio = lowest ÷ highest approval rate. Below 0.80 is the usual "
                       "four-fifths screening threshold. This is a screening signal, not a legal opinion.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("🏦 Universal Credit Studio")
    st.caption("Point it at tables, not at a feature list. It profiles the data, builds features, "
               "runs a model tournament, and explains every decision.")

    with st.sidebar:
        st.header("Setup")
        source = st.radio("Data", ["Demo loan book", "Upload CSVs"], key="source")

        st.subheader("Target")
        scorecard_type = st.selectbox(
            "Scorecard type", ["behavioral", "application"],
            help="application = decision-time data only; behavioral = also uses the first months of "
                 "repayment behaviour",
        )
        bad_dpd = st.number_input(
            "Bad = DPD ≥", 1, 360, 90,
            help="Days past due that counts as a default. 90 is the common standard; lower is stricter.",
        )
        obs = st.number_input(
            "Observation months", 0, 24, 3,
            help="How many months of behaviour are observed before the performance window starts "
                 "(behavioural scorecards only).",
        )
        perf = st.number_input(
            "Performance months", 1, 36, 9,
            help="How long after disbursement you wait to decide whether a loan went bad. Only "
                 "loans whose window has fully passed are used.",
        )
        test_fraction = st.slider(
            "Out-of-time test share", 0.1, 0.5, 0.3, 0.05,
            help="Share of the most recent loans held back to test the model honestly.",
        )
        policy = st.selectbox(
            "Model selection policy", ["balanced", "regulatory", "performance"],
            help="balanced = best prediction with trustworthy probabilities; regulatory = keep the "
                 "interpretable scorecard unless a challenger is clearly better; performance = "
                 "highest accuracy regardless of interpretability.",
        )
        approval_rate = st.slider(
            "Target approval rate (for the fairness cut-off)", 0.5, 0.95, 0.8, 0.05,
            help="Share of applicants you intend to approve. It sets the PD cut-off used on the "
                 "Fairness tab.",
        )

        run = st.button("▶ Run auto pipeline", type="primary")

    tables, cfg_source = None, ROOT / "config.yaml"

    if source == "Upload CSVs":
        uploaded = _upload_tables()
        if uploaded:
            spec, _ = uploaded
            if "loans" not in spec:
                st.error("A loan table is required.")
                return
            cfg_source = {
                "as_of_date": st.date_input("As-of date", value=pd.Timestamp("2025-12-31")).strftime("%Y-%m-%d"),
                "scorecard_type": scorecard_type,
                "data": spec,
            }
        else:
            st.info("Upload at least a loans CSV to begin. Column names are matched automatically; "
                    "unmatched required columns can be mapped below.")
            return
    else:
        if not (ROOT / "data" / "loans.csv").exists():
            st.info("Demo data not built yet — it will be generated on the first run.")
            if run:
                from generate_sample_data import generate

                with st.spinner("Generating the demo loan book…"):
                    generate(6000, out_dir=str(ROOT / "data"))
        cfg_source = ROOT / "config.yaml"

    if run:
        cfg = cfg_source if isinstance(cfg_source, dict) else None
        if cfg is None:
            import yaml

            cfg = yaml.safe_load(Path(cfg_source).read_text())
        cfg.setdefault("scorecard_type", scorecard_type)
        cfg["scorecard_type"] = scorecard_type
        cfg.setdefault("target", {}).update(
            {"bad_dpd": int(bad_dpd), "observation_months": int(obs), "performance_months": int(perf)}
        )
        cfg.setdefault("split", {})["test_fraction"] = float(test_fraction)
        cfg.setdefault("selection", {})["policy"] = policy

        with st.spinner("Profiling schema, building features, running the tournament…"):
            try:
                st.session_state["result"] = run_auto(
                    cfg, tables=tables, approval_rate=approval_rate, save_artifact=True
                )
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"{type(exc).__name__}: {exc}")
                return

    result: AutoResult | None = st.session_state.get("result")
    if result is None:
        st.stop()

    for warning in result.warnings:
        st.warning(warning)

    if result.artifact_path:
        st.caption(f"model artifact saved to `{result.artifact_path}`")

    tabs = st.tabs([
        "Start here",
        "1 · Your data",
        "2 · Model quality",
        "3 · Explaining a decision",
        "4 · Fairness",
    ])
    with tabs[0]:
        page_start_here()
    with tabs[1]:
        page_profile(result)
    with tabs[2]:
        page_models(result)
    with tabs[3]:
        page_explain(result)
    with tabs[4]:
        page_fairness(result)


if __name__ == "__main__":
    main()
