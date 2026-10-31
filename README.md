# Fintech Risk Analytics

Two independent projects about credit risk, kept in one repository:

| Project | Folder | Focus |
|---|---|---|
| **InsightCopilot** | repository root | Agentic SQL + analytics over an Indonesian *pinjol* (online lending) warehouse |
| **Credit Scoring Engine** | [`credit-scoring/`](credit-scoring/) | Universal credit scorecard: auto schema discovery, model tournament, per-applicant explanations, fairness audit |

Each project is self-contained: its own dependencies, entry point and data.

---

## 1. InsightCopilot — agentic analytics on a lending warehouse

A Streamlit application that answers questions about a pinjol portfolio by planning SQL
over a dimensional warehouse, rather than free-handing queries.

```
pinjol_csvs/            star-schema extract (loans, applications, assessments, funding,
                        repayments, collections, borrowers)
pinjol_fintech.db       SQLite build of the same data
core/
  graph_join_engine.py  resolves join paths between warehouse tables
  sql_executor.py       runs the planned SQL safely
  pinjol_analytics.py   portfolio metrics (delinquency, roll rates, vintages)
  agentic_actions.py    turns findings into suggested actions
  lightdash.py          exports dashboards
streamlit_app.py        the UI
scripts/
  generate_pinjol_data.py   regenerates the synthetic warehouse
```

**Run it**

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Regenerate the data**

```bash
python scripts/generate_pinjol_data.py
```

---

## 2. Credit Scoring Engine — `credit-scoring/`

A reusable credit-scoring framework with two modes:

- **Universal auto mode** (`auto_app.py`) — point it at any tables. It profiles the schema, discovers
  features, derives the target, runs a model tournament, explains every decision, and audits fairness.
  No feature list to write by hand.
- **Studio mode** (`app.py`) — the bundled synthetic lending book with portfolio, onboarding,
  collections and model-studio pages.

**Run it**

```bash
cd credit-scoring
pip install -r requirements.txt
streamlit run auto_app.py     # universal mode
streamlit run app.py          # studio mode
```

**Highlights**

- Schema profiling and feature discovery, with a reason for every rejected column
- Point-in-time joins so repayment behaviour after the observation window cannot leak in
- Model tournament (WoE scorecard, logistic, GBM, calibrated GBM, random forest) with an explicit
  selection policy: `balanced`, `regulatory` or `performance`
- Per-applicant explanations — exact decompositions for scorecard/linear models, labelled
  counterfactual approximations otherwise
- Fairness audit (protected attributes excluded from scoring, disparate-impact ratio reported)
- CSV ingestion with format validation, blank templates and a ready-to-run example dataset

**Docs** — [`credit-scoring/README.md`](credit-scoring/README.md) for the technical overview,
[`credit-scoring/docs/BUSINESS_GUIDE.md`](credit-scoring/docs/BUSINESS_GUIDE.md) for a plain-English
guide written for non-technical readers.

**Tests**

```bash
cd credit-scoring
python -m pytest tests -q
```

---

## Conventions

- **Synthetic data only.** No real borrower, applicant or customer records are stored here.
- **Separate environments.** The two projects have different dependency sets; install per folder.
- **Read-only by default.** Every query path in `credit-scoring/` runs in a read-only transaction with
  a statement timeout and a row cap.
- Generated artefacts (`data/`, `artifacts/`, model files) are ignored by git — rebuild them locally.

## License

Internal / personal project. Add a license before distributing.
