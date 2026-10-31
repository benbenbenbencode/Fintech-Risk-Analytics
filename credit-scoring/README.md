# Credit Risk Studio

Two ways to use this repo:

1. **Universal auto mode** (`auto_app.py`) — point it at *any* set of tables. It profiles the schema,
   discovers features, picks a model by a declared policy, explains every decision, and audits fairness.
   No feature list to write by hand.
2. **Studio mode** (`app.py`) — the original synthetic lending book with portfolio, onboarding,
   collections, cannibalization and ask-a-question pages.

> **For business users:** start with **[docs/BUSINESS_GUIDE.md](docs/BUSINESS_GUIDE.md)**. It is written
> without statistics — every number explained, each screen walked through, a worked example, a
> glossary, and an honest list of what the tool refuses to do.

```bash
pip install -r requirements.txt

streamlit run auto_app.py     # universal: bring any tables
streamlit run app.py          # studio: the demo world
```

---

## Universal auto mode

```python
from crf.auto_pipeline import run_auto

result = run_auto("config.yaml")          # or run_auto(cfg_dict, tables={...})
result.tournament.leaderboard              # every candidate, every metric
result.tournament.reason                   # why the winner won
result.explain(loan_id).text               # plain-language applicant brief
result.fairness["flags"]                   # adverse-impact warnings
```

### Bring your own data

Two files are enough: a **loan table** (one row per loan) and a **repayment history** (one row per
instalment). The target and the behavioural features both come from the repayment history; the loan
table supplies disbursement date, principal and tenor.

If you only have repayment history, the app derives the loan table (dates and amounts estimated) and
tells you which fields were approximated.

The exact columns are in the format table inside the app and in
[docs/BUSINESS_GUIDE.md](docs/BUSINESS_GUIDE.md). `examples/` ships a ready-to-run dataset — 480
loans, 6,600 instalments, with income, employment, region and a bureau snapshot — plus a matching
`config.yaml`:

```bash
python scripts/check_examples.py     # validates the format, then trains on it
```

Uploads are validated before training: missing columns, duplicate loan IDs, unparseable dates and
orphan repayments produce a blocking checklist rather than a failed model run.

### What is automatic

| Step | What it does | Where |
|---|---|---|
| **Schema profiling** | Classifies every column as identifier / date / numeric / categorical / free text, with a reason for each rejection | `crf/auto_profile.py` |
| **Feature discovery** | Keeps usable columns, derives `days_since_*`, rolls child tables up by detected keys, builds amount ratios | `crf/auto_profile.py` |
| **Point-in-time joins** | Child-table rows are only counted up to each loan's observation-window end, so repayment behaviour from the performance window cannot leak in | `crf/auto_profile.py` |
| **Guardrails** | Drops the target column, leaked-outcome names (`collection`, `writeoff`, …), protected attributes, constants, near-duplicates and >90%-missing columns | `crf/auto_profile.py` |
| **Target** | DPD-based bad definition with maturity filters and a population waterfall | `crf/target.py` |
| **Tournament** | Fits scorecard, logistic, GBM, calibrated GBM and random forest on the same out-of-time split | `crf/auto_model.py` |
| **Selection** | Picks by explicit policy: `balanced`, `regulatory` or `performance` | `crf/auto_model.py` |
| **Explanations** | Per-applicant contributions and adverse-action reasons | `crf/explain.py` |
| **Fairness** | Detects protected attributes and measures approval-rate disparity | `crf/fairness.py` |

### Model tournament, measured on the bundled demo

4,826 eligible loans, 6.9% bad rate, out-of-time test set:

| Model | AUC | Calibration slope | Selected |
|---|---|---|---|
| Logistic (raw, scaled) | 0.765 | 1.17 | ✅ balanced policy |
| WoE scorecard | 0.709 | 1.02 | — |
| Random forest | 0.694 | 1.13 | — |
| GBM (calibrated) | 0.685 | 0.91 | — |
| GBM | 0.679 | 0.51 | — |

Switch policies in the sidebar. `regulatory` keeps the scorecard unless a challenger beats it by
≥ 0.01 AUC **and** is acceptably calibrated — the trade-off is explicit, not hidden in a default.
Every decision prints the numbers behind it.

### Explanations: exact where possible

A tempting shortcut is to explain every model with a single-feature counterfactual. It breaks down
when features are correlated: in this repo `installment_amount` and `repayments_paid_amount_mean`
are near-identical, and substituting one moved an applicant's PD from 3% to 67%. That is not an
explanation, it is an artifact.

So explanations use the most exact method the model allows:

1. **Scorecard** → points are additive by construction; contribution = points contributed.
2. **Linear model** → logit is exactly `b + Σ wᵢzᵢ`; contribution = `wᵢzᵢ`.
3. **Everything else** → counterfactual, explicitly labelled *approximate*.

Collinear features (|corr| > 0.95) are dropped before modelling, keeping the higher-IV member.

### Fairness

Protected attributes (sex, age, marital status, religion, ethnicity, nationality, disability) are
**detected, excluded from scoring, and audited**. The audit reports approval rate and mean PD per
group, plus the disparate-impact ratio (lowest ÷ highest approval rate). Below 0.80 raises a flag —
the four-fifths screening rule, not a legal opinion.

### Guardrails worth knowing

- The **target never becomes a feature**. Column names like `bad`, `default`, `target`, `label` are
  blocked by exact match; `*_flag` variants by pattern. Exact match matters — substring matching
  would also drop a legitimate `bad_rate`.
- **Leakage screen** runs twice: name patterns at discovery, and a single-feature-IV check at
  selection (`IV > 2.0` is "too good to be true").
- **Every query is read-only and bounded** — training runs on a materialised frame, not on live SQL.

### UI

`streamlit run auto_app.py` — five tabs, written for a business reader:

1. **Start here** — what the tool does, what each tab answers, how to read the main numbers.
2. **Your data** — what was found, what was rejected and why; table relationships; population waterfall.
3. **Model quality** — leaderboard, AUC bars, ROC, calibration curve, decile bad rate, feature importance.
4. **Explaining a decision** — score gauge, risk band, per-feature contribution bars, adverse-action
   reasons, applicant-vs-population table, and a plain-language brief in English or Indonesian.
5. **Fairness** — approval rate and mean PD by group, DI ratio, flags.

The same content, in full prose, is in **[docs/BUSINESS_GUIDE.md](docs/BUSINESS_GUIDE.md)**.

---

## Studio mode (existing)

The synthetic worldwide lending book (10 countries, 50 cities, 100k customers, 3 products, 48 months):
portfolio, customers, onboarding with a profit-optimal cut-off simulator, credit limits, collections
with randomised-treatment uplift, cannibalization, NL→validated-SQL questions, and a model studio.

```bash
python build.py --customers 100000    # headless build
streamlit run app.py
streamlit run lab_app.py              # CSV lab: upload tables, AI maps the schema
```

Design principle unchanged: **models decide, AI explains.** The LLM writes language and parses
questions; it never produces the decision or the SQL that runs.

## Layout

```
crf/            scorecard core: WoE/IV, leakage, PSI, evaluation, scorecard, targeting
  auto_profile.py   schema profiling + feature discovery      (new)
  auto_model.py     model tournament + selection policy        (new)
  explain.py        per-applicant contributions and reason codes (new)
  fairness.py       protected attributes + disparate impact    (new)
  auto_pipeline.py  one-call orchestration                     (new)
studio/         demo world generator, models, policy, AI layer, pages
auto_app.py     universal UI (new)
app.py          studio UI
docs/           BUSINESS_GUIDE.md - plain-English guide for non-technical readers (new)
lab_app.py      CSV lab UI
tests/          core tests + auto-layer tests
scripts/        smoke tests: smoke_auto.py, test_auto_app.py, bench-style checks
```

## Tests

```bash
python -m pytest tests -q                    # 10 passed
python scripts/smoke_auto.py                 # full pipeline, prints what it found and chose
python scripts/test_auto_app.py              # headless UI: renders, runs, no exceptions
```

## Known limitations

- **Selection bias**: application models are trained on approved/matured loans only; there is no
  reject inference.
- **The demo data is synthetic.** Structure and correlations are realistic; the numbers are not a bank.
- **Fine-grained explanations for tree models are approximate** by design — the UI labels them.
- **Fairness metrics are screening signals**, not a compliance opinion; thresholds like 0.80 are
  conventions, and local regulation governs.
- **Not a production system**: no auth, model registry, audit trail or scheduled retraining.
