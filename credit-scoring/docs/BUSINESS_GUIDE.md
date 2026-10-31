# Credit Risk Studio — Business User Guide

This guide is written for people who make or review lending decisions, not for data scientists.
No statistics background is assumed. Every number the tool shows is explained in plain language.

If you only read one section, read **Section 1** and **Section 4**.

---

## 1. What this tool does, in one minute

You give it a table of past loans: who borrowed, how much, and whether they repaid.

The tool then:

1. **Works out which columns are useful** — and which ones must be ignored (for example the
   customer's gender, or a "collection status" field that was only filled in *after* someone
   stopped paying).
2. **Builds a risk model** from those columns.
3. **Tests several kinds of model** and keeps the one that predicts best, and it tells you *why*
   it picked that one.
4. **Explains every single decision** in sentences a customer could understand.
5. **Checks the results are not unfairly skewed** between groups of people.

You do not need to tell it which columns to use. That is the main difference from the older version
of this project, where every feature had to be listed by hand.

> **The one thing to remember:** the model produces a *probability of default*, and the tool turns
> that into a score, a risk band, and a short list of reasons. The reasons are the part you will use
> in conversations with customers, credit committees, and auditors.

---

## 2. What you need before you start

### The two tables you must have

| Table | One row per… | Why it matters |
|---|---|---|
| **Loans** | loan | Who borrowed, when, how much, for how long |
| **Repayments** | instalment | Whether each instalment was paid, and when |

Optional tables — customers, credit bureau snapshots, or any other table with a matching ID — are
picked up automatically and summarised per borrower.

### The exact file format

Two files are enough. Use these column names; the app can map different names, but starting here
saves time.

**loans.csv** — one row per loan

| column | type | required | example | meaning |
|---|---|---|---|---|
| `loan_id` | text | yes | `L00001` | unique reference for the loan |
| `customer_id` | text | yes | `C001` | borrower reference; repeats across loans |
| `disbursement_date` | date | yes | `2024-03-15` | when the money was paid out |
| `principal` | number | yes | `5000` | amount lent |
| `tenor_months` | integer | yes | `12` | number of monthly instalments |
| `interest_rate_apr` | number | optional | `18.5` | annual rate, in percent |
| `purpose` | text | optional | `working_capital` | loan purpose |

**repayments.csv** — one row per instalment

| column | type | required | example | meaning |
|---|---|---|---|---|
| `loan_id` | text | yes | `L00001` | must exist in loans.csv |
| `installment_no` | integer | yes | `1` | 1, 2, 3 … within each loan |
| `due_date` | date | yes | `2024-04-15` | scheduled payment date |
| `paid_date` | date | yes, may be empty | `2024-04-18` | actual payment; empty = unpaid |
| `due_amount` | number | yes | `520` | amount that was due |
| `paid_amount` | number | optional | `520` | amount actually paid |

Five rules that cause most upload failures:

1. Dates are written `YYYY-MM-DD`.
2. `loan_id` is unique in loans.csv — one row per loan.
3. Every `loan_id` in repayments.csv exists in loans.csv. Check for leading zeros and text-vs-number mismatches.
4. Leave `paid_date` empty when an instalment was not paid. Do not use `0000-00-00` or a sentinel date.
5. Extra columns are kept and can be used as features — income, employment type, region, and so on.

The app checks all of this before training and shows a pass/warn/fail list, so you find out in
seconds rather than after a failed run.

### Example data you can download

`examples/` ships a ready-to-run dataset: **480 loans, 6,600 instalments**, with income, employment
type, region and a credit-bureau snapshot.

- **In the app:** Upload CSVs → “File format, example data and templates” → *Download example dataset*.
  Individual files and blank templates are there too.
- **In the repo:** `examples/loans.csv`, `examples/repayments.csv`, `examples/customers.csv`,
  `examples/bureau.csv`, and `examples/config.yaml`.

Run it end to end:

```bash
python scripts/check_examples.py
```

That validates the format first, then trains the full pipeline on it, so you can see a complete run
before touching your own data.

### If you only have repayment history

You can upload a single repayment history file. The tool derives the loan table for you:

- disbursement date = one month before the first due date (**estimated**)
- principal = sum of scheduled instalments (**estimated**)
- tenor = highest instalment number

It tells you which fields were approximated. Columns such as income or purpose will simply not exist,
which costs predictive power. Prefer the two-file mode whenever you have it.

### Three settings you must decide

1. **What counts as "bad"?** The default is *90 or more days past due*. A stricter definition
   (60 days) catches more problems sooner but flags more people. This is a business decision, not a
   technical one — agree it with risk and compliance before running.
2. **How long is the observation window?** The first months of a loan, used to observe behaviour
   (only relevant for behavioural scorecards). Keep it shorter than the performance window.
3. **How long is the performance window?** How long after disbursement you wait to decide whether
   the loan went bad. A loan only counts if this window has fully passed by the "as of" date.

### How much data is enough?

- At least **200 loans** with a completed performance window, ideally several thousand.
- At least **a few hundred bad loans**. If only 5 loans ever went bad, no model can learn anything
  reliable — and the tool will warn you rather than pretend.

---

## 3. Running it — three steps

```bash
cd credit_risk_framework
streamlit run auto_app.py
```

Then, in the app:

1. **Left sidebar → Data.** Keep "Demo loan book" to explore with sample data, or choose
   "Upload CSVs" to use your own.
2. **Left sidebar → Target.** Set the bad definition, observation window and performance window.
3. **Click "Run auto pipeline"** and give it a few seconds.

The four tabs at the top then fill in, in the order you should read them.

---

## 4. Reading the four screens

### Tab 1 — Profile: "What did it find in my data?"

This is the inventory. Three things to check:

- **Features discovered.** These are the columns the model is allowed to use. Each one shows where it
  came from (for example `monthly_income` from the loan table, or `repayments_paid_amount_mean`
  summarised from the repayment table).
- **Rejected columns.** Every rejection has a reason. The three reasons you will see most:
  - *identifier* — an account number cannot predict anything.
  - *protected attribute* — gender, age, marital status and similar are deliberately excluded from
    scoring, then audited separately on the Fairness tab.
  - *leaked outcome* — a column like `collection_status` or `writeoff_flag` is filled in only after
    the customer stops paying. Using it would be cheating: the model would look brilliant in
    testing and be useless in production.
- **Population waterfall.** How many loans were dropped on the way to a usable dataset, and why.
  It is normal to lose a lot. If you only keep 10% of your loans, your performance window is
  probably too long for the amount of history you have.

**What to do here:** confirm that the columns kept are ones your business would actually accept
being used in a credit decision. If something looks wrong, it usually means a naming problem.

### Tab 2 — Models: "Is the model any good?"

The tool tries five kinds of model and shows a leaderboard. Do not try to read every row. Look at
three numbers for the model marked as selected.

| Number | Plain meaning | What "good" looks like |
|---|---|---|
| **AUC** | If you pick one customer who defaulted and one who did not, how often does the model rank the defaulter as riskier? | 0.50 = coin flip, 0.65–0.75 = usable, 0.75–0.85 = strong, above 0.90 on real data means something is probably leaking |
| **Calibration slope** | When the model says "5% chance of default", is it really about 5%? | 1.00 is perfect. 0.9–1.1 is good. Far from 1.0 means the percentages are not trustworthy even if the ranking is fine |
| **Top-decile lift** | How much riskier is the riskiest 10% compared with an average customer? | 3× or more means the riskiest group is clearly separable |

**Why the chosen model might not be the "scorecard".** The scorecard is the traditional bank model:
it produces a points table and reason codes that are easy to defend. A gradient-boosting model often
predicts slightly better but is harder to explain. The selection policy in the sidebar decides the
trade-off:

- `balanced` — best prediction quality, as long as probabilities are trustworthy.
- `regulatory` — keeps the scorecard unless another model is better by a clear margin **and** is well
  calibrated. Use this if you must justify every point to a regulator.
- `performance` — highest accuracy, interpretability not considered.

The app always prints the reason, with the actual numbers, in the green box at the top of the tab.

**Charts on this tab:**

- **ROC curves** — higher and further left is better. Useful only for comparing models.
- **Calibration by risk decile** — dots should sit on the diagonal. Above the line means the model
  underestimates risk in that group.
- **Bad rate by decile** — should fall steadily from decile 1 to 10. If it zig-zags, the model is
  confused.
- **Feature importance** — which columns the model actually relies on. If something surprising is at
  the top, ask why before going live.

### Tab 3 — Explain an applicant: "Why did this person get this decision?"

Pick a loan ID from the dropdown (only customers the model never saw during training are offered).

You get:

- **Probability of default** — the model's estimate, as a percentage.
- **Score and band** — the same risk expressed on a 0–850 scale and an A–E band, where A is safest.
- **Recommended action** — approve, or decline / manual review, based on the approval cut-off.
- **The bar chart** — this is the important part. Each bar answers: *how many percentage points of
  risk does this one factor add, compared with a safe value for that factor?*
  - **Red, pointing right** — this factor makes the applicant riskier. These become your adverse-action
    reasons.
  - **Green, pointing left** — this factor makes the applicant safer.
- **The written brief** — the same information in sentences, available in English or Indonesian.
- **Applicant vs population** — their value for each factor next to the typical value, so you can see
  at a glance whether they are an outlier.

**Reading it honestly:** the bars show *what the model reacted to*, not the cause of the customer's
financial situation. "High loan-to-income" is a fact about the application, not a moral judgement,
and it is exactly the kind of reason you are allowed to give a customer.

### Tab 4 — Fairness: "Are the decisions fair across groups?"

The tool automatically looks for gender, age, marital status, religion, ethnicity, nationality and
disability columns. They are excluded from scoring, then used here to compare outcomes.

For each group you see:

- **Applicants** — how many people fall in the group.
- **Approval rate** — what share of them would be approved at the current cut-off.
- **Mean PD** — the average predicted risk for the group.
- **DI ratio** — the lowest approval rate divided by the highest. **Below 0.80 raises a flag.**

The 0.80 figure is the widely used "four-fifths rule": if one group is approved at less than 80% of
the rate of the most-approved group, that is a signal worth investigating. It is a screening
heuristic, not a legal verdict, and local regulation governs what is required.

**What to do if you see a flag:** do not switch the model off in a panic. First check whether the
difference reflects a genuine risk difference in the data (for example, one group really does have
much lower income). Then discuss with compliance. The flag is a prompt for a conversation, not proof
of discrimination.

---

## 5. A worked example (numbers from the bundled demo)

An applicant comes in. The tool reports:

- **Probability of default 3.7% — band B.**
- **Reasons increasing risk:** principal amount `+1.7 pp`, average repayment amount `+1.5 pp`,
  bureau outstanding / principal `+0.7 pp`, recent bureau inquiries `+0.7 pp`.
- **Reasons reducing risk:** instalment / principal `-3.2 pp`, bureau outstanding / income `-3.0 pp`.

How to read that in a credit conversation:

> "Your application is in the second-safest band. The main factors counting against you are the size
> of the loan relative to what you told us, and how much you have borrowed elsewhere recently.
> Those are offset by a small instalment relative to the loan and low total borrowing relative to
> your income."

That is a defensible, factual statement. It does not reveal the model's internals, it does not
mention protected attributes, and every number comes from the computation rather than from the
language model.

---

## 6. Words explained

| Term | What it actually means |
|---|---|
| **PD** | Probability of Default — the chance this loan goes bad, as a percentage |
| **Bad rate** | The share of historical loans that actually did go bad |
| **DPD** | Days Past Due — how many days late a payment is |
| **AUC** | Ranking quality, from 0.5 (useless) to 1.0 (perfect) |
| **KS** | Another ranking measure; higher is better, 0.3+ is usually respectable |
| **Gini** | AUC restated on a 0–1 scale; often used in risk reporting |
| **Calibration** | Whether predicted percentages match reality |
| **Lift** | How much riskier a selected group is compared with average |
| **Score band** | A–E grouping by predicted risk; A is safest |
| **Feature** | A column the model is allowed to use |
| **Leakage** | Accidentally using information from the future, which makes a model look better than it is |
| **Out-of-time test** | Scoring on the most recent loans, which the model never saw — the honest way to test |
| **WoE / IV** | Standard credit-scoring techniques for turning columns into points and measuring their strength |
| **DI ratio** | Disparate Impact ratio — lowest group approval rate divided by highest |
| **Reason code** | A short, factual explanation given to a customer for a decision |

---

## 7. Questions people actually ask

**Why was age excluded? I thought age was predictive.**
Because age is a protected characteristic in most jurisdictions. It may well be predictive; that is
exactly why using it is regulated. It is excluded from scoring and monitored on the Fairness tab.

**Why did it pick the logistic model instead of the scorecard?**
Because on your data it predicted better and stayed well calibrated. If you need point-level
justification, switch the policy to `regulatory` and re-run — the tool will keep the scorecard unless
the challenger is clearly better.

**An AUC of 0.70 — is that bad?**
For credit risk on real data, 0.65–0.75 is normal and usable. Values above 0.90 usually mean
leakage, not genius. Compare models against each other rather than against an absolute target.

**It dropped my favourite column. Why?**
Check the "Rejected columns" table. The usual reasons are: it was 100% missing after joining, it was
constant, it duplicated another column almost exactly, or its name looked like a leaked outcome.

**What does "approximate" mean on an explanation?**
For some model types there is no exact way to split a prediction into per-factor pieces. The tool
then uses a what-if comparison and labels it *approximate*. For the scorecard and logistic models the
split is exact and the label says so.

**Can I use this for real credit decisions today?**
No. This is a working framework, not a validated production system. Before real use you need at
least: independent validation, documented model risk review, monitoring for drift, an audit trail of
decisions, and legal review of your reason codes and fair-lending obligations.

**The demo numbers look impressive. Are they real?**
They come from synthetic data generated by this project. The structure is realistic; the numbers are
not a bank's.

---

## 8. What this tool is not

- **Not a compliance sign-off.** It produces evidence to support a review, not the review itself.
- **Not trained on rejected applicants.** It learns from loans that were approved and matured, so it
  inherits whatever selection bias your approval policy already had.
- **Not a monitoring system.** There is no automatic drift detection or scheduled retraining.
- **Not magic on tiny data.** Below roughly 200 usable loans and a few hundred bad cases, treat the
  output as a demonstration, not evidence.

---

## 9. Where to go next

- **README.md** — the technical overview, how to install, how the pieces fit together.
- **Tab 1 "Start here"** inside the app — a condensed version of this guide.
- **For analysts:** `crf/auto_profile.py` (discovery), `crf/auto_model.py` (tournament),
  `crf/explain.py` (explanations), `crf/fairness.py` (audit).
- **To change behaviour without touching code:** edit `config.yaml` — bad definition, windows,
  binning, selection thresholds, score scaling.
