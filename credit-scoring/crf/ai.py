"""AI layer. Every feature works OFFLINE (heuristics / templates) and gets better with an LLM.

Privacy: only schema info (column names + optional sample values) and aggregate model
statistics are ever sent to the LLM. Row-level borrower data never leaves your machine.

Providers (auto-detected from env vars, or set explicitly in the UI):
    ANTHROPIC_API_KEY            -> Anthropic Messages API
    OPENAI_API_KEY               -> OpenAI-compatible chat completions
    CRF_LLM_BASE_URL (+ model)   -> any OpenAI-compatible server (Ollama, LM Studio, vLLM...)
    CRF_LLM_MODEL                -> override the model name
"""
from __future__ import annotations

import copy
import json
import os
import re
import urllib.request
from difflib import SequenceMatcher

import pandas as pd

from .config import DEFAULTS

# =====================================================================
# LLM client (stdlib only)
# =====================================================================


class LLM:
    def __init__(self, provider: str = "auto", model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, timeout: int = 90):
        self.timeout = timeout
        env = os.environ
        if provider == "auto":
            if api_key or env.get("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif env.get("OPENAI_API_KEY"):
                provider = "openai"
            elif base_url or env.get("CRF_LLM_BASE_URL"):
                provider = "openai"
            else:
                provider = "offline"
        self.provider = provider
        if provider == "anthropic":
            self.api_key = api_key or env.get("ANTHROPIC_API_KEY")
            self.model = model or env.get("CRF_LLM_MODEL", "claude-sonnet-5-5")
            self.base_url = base_url or "https://api.anthropic.com"
        elif provider == "openai":
            self.api_key = api_key or env.get("OPENAI_API_KEY", "")
            self.model = model or env.get("CRF_LLM_MODEL", "gpt-4o-mini")
            self.base_url = (base_url or env.get("CRF_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        else:
            self.api_key, self.model, self.base_url = None, None, None

    @property
    def available(self) -> bool:
        return self.provider in ("anthropic", "openai")

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        if not self.available:
            raise RuntimeError("LLM not configured")
        if self.provider == "anthropic":
            out = self._post(f"{self.base_url}/v1/messages",
                             {"content-type": "application/json", "x-api-key": self.api_key,
                              "anthropic-version": "2023-06-01"},
                             {"model": self.model, "max_tokens": max_tokens, "system": system,
                              "messages": [{"role": "user", "content": user}]})
            return "".join(b.get("text", "") for b in out.get("content", []))
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        out = self._post(f"{self.base_url}/chat/completions", headers,
                         {"model": self.model, "max_tokens": max_tokens,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        return out["choices"][0]["message"]["content"]

    def complete_json(self, system: str, user: str, max_tokens: int = 2500) -> dict:
        txt = self.complete(system + "\nReturn ONLY valid JSON, no markdown fences.", user, max_tokens)
        txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
        m = re.search(r"\{.*\}", txt, flags=re.S)
        return json.loads(m.group(0) if m else txt)


# =====================================================================
# 1) AI schema mapper: raw CSVs -> canonical config
# =====================================================================

CANONICAL = {
    "loans": {
        "loan_id": ["loan_id", "loanid", "contract_id", "contract_no", "account_id", "application_id", "id_pinjaman"],
        "customer_id": ["customer_id", "customerid", "borrower_id", "client_id", "user_id", "cust_id", "member_id"],
        "disbursement_date": ["disbursement_date", "disbursed_at", "disburse_date", "origination_date", "start_date",
                              "loan_date", "funded_date", "approved_date"],
        "principal": ["principal", "loan_amount", "amount", "disbursed_amount", "approved_amount", "financed_amount"],
        "tenor_months": ["tenor_months", "tenor", "term_months", "term", "loan_term", "duration_months"],
    },
    "repayments": {
        "loan_id": ["loan_id", "loanid", "contract_id", "account_id", "application_id"],
        "installment_no": ["installment_no", "installment_number", "inst_no", "payment_no", "seq", "period", "cicilan_ke"],
        "due_date": ["due_date", "scheduled_date", "payment_due_date", "jatuh_tempo"],
        "paid_date": ["paid_date", "payment_date", "paid_at", "actual_payment_date", "repayment_date"],
        "due_amount": ["due_amount", "scheduled_amount", "installment_amount", "amount_due", "expected_amount"],
        "paid_amount": ["paid_amount", "amount_paid", "payment_amount", "actual_amount"],
    },
    "customers": {
        "customer_id": ["customer_id", "customerid", "borrower_id", "client_id", "user_id", "cust_id", "member_id"],
        "monthly_income": ["monthly_income", "income", "salary", "monthly_salary", "gaji", "net_income"],
    },
    "bureau": {
        "customer_id": ["customer_id", "customerid", "borrower_id", "client_id", "user_id", "cust_id"],
        "snapshot_date": ["snapshot_date", "report_date", "pulled_at", "bureau_date", "inquiry_date", "slik_date"],
    },
}
ROLE_REQUIRED = {"loans": 5, "repayments": 5, "customers": 1, "bureau": 1}
LEAKY_HINTS = ["collection", "legal", "recovery", "charge_off", "writeoff_date", "default_flag", "dpd_current",
               "status_after", "settled", "closed_reason", "npl"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _match_column(cols: list[str], synonyms: list[str]) -> str | None:
    ncols = {_norm(c): c for c in cols}
    for s in synonyms:                                   # exact
        if _norm(s) in ncols:
            return ncols[_norm(s)]
    best, score = None, 0.0
    for s in synonyms:                                   # fuzzy
        for nc, c in ncols.items():
            r = SequenceMatcher(None, _norm(s), nc).ratio()
            if r > score:
                best, score = c, r
    return best if score >= 0.82 else None


def _heuristic_roles(raw: dict[str, pd.DataFrame]) -> dict:
    """file name -> (role, {canonical: raw col}). Greedy: best-scoring file wins each role."""
    scores = []
    for fname, df in raw.items():
        for role, spec in CANONICAL.items():
            m = {canon: _match_column(list(df.columns), syn) for canon, syn in spec.items()}
            m = {k: v for k, v in m.items() if v}
            scores.append((len(m) / len(spec), len(m), fname, role, m))
    scores.sort(key=lambda t: (-t[0], -t[1]))
    used_files, used_roles, out = set(), set(), {}
    for frac, n, fname, role, m in scores:
        if fname in used_files or role in used_roles or n == 0:
            continue
        if role in ("loans", "repayments") and n < 3:
            continue
        out[role] = {"file": fname, "columns": m}
        used_files.add(fname)
        used_roles.add(role)
    return out


def _llm_roles(raw: dict[str, pd.DataFrame], llm: LLM, send_samples: bool) -> tuple[dict, list[str], str]:
    schema = {}
    for fname, df in raw.items():
        cols = {}
        for c in df.columns:
            info = {"dtype": str(df[c].dtype)}
            if send_samples:
                info["samples"] = [str(v) for v in df[c].dropna().astype(str).unique()[:3]]
            cols[c] = info
        schema[fname] = {"rows": len(df), "columns": cols}
    system = ("You are a credit-risk data engineer. Map raw CSV files onto a canonical lending schema.")
    user = f"""Canonical tables and columns (canonical names must be used as keys):
{json.dumps({r: list(c) for r, c in CANONICAL.items()})}

Raw files:
{json.dumps(schema)[:12000]}

Return JSON:
{{"tables": {{"<role: loans|repayments|customers|bureau>": {{"file": "<file name>", "columns": {{"<canonical>": "<raw column>"}}}}}},
 "leakage_suspects": ["<raw column that is only known AFTER the loan outcome, e.g. collection/legal/recovery status>"],
 "notes": "<1-3 short sentences on assumptions or things the user should double-check>"}}
Only map columns you are confident about. loans and repayments are required."""
    out = llm.complete_json(system, user)
    roles = {r: v for r, v in out.get("tables", {}).items() if r in CANONICAL and v.get("file") in raw}
    return roles, list(out.get("leakage_suspects", [])), str(out.get("notes", ""))


def suggest_config(raw: dict[str, pd.DataFrame], paths: dict[str, str], llm: LLM | None = None,
                   send_samples: bool = True) -> dict:
    """raw: {file name: DataFrame}; paths: {file name: path to use in the config}.
    Returns {"config": dict, "notes": [..], "source": "llm"|"heuristic"}."""
    notes, source, leak_suspects = [], "heuristic", []
    roles = _heuristic_roles(raw)
    if llm is not None and llm.available:
        try:
            lroles, leak_suspects, lnote = _llm_roles(raw, llm, send_samples)
            for role, v in lroles.items():           # keep only mappings that actually exist
                cols = {k: c for k, c in v.get("columns", {}).items()
                        if k in CANONICAL[role] and c in raw[v["file"]].columns}
                if cols:
                    roles[role] = {"file": v["file"], "columns": cols}
            source = "llm"
            if lnote:
                notes.append(lnote)
        except Exception as e:                          # network, parsing, quota ...
            notes.append(f"LLM unavailable ({type(e).__name__}); used heuristic matching instead.")

    for need in ("loans", "repayments"):
        if need not in roles:
            raise ValueError(f"Could not find a '{need}' table. Check the uploaded files or map columns manually.")
        missing = [c for c in CANONICAL[need] if c not in roles[need]["columns"]
                   and c not in ("paid_amount",)]
        if missing:
            notes.append(f"{need}: could not map {missing} - fill them in manually in the YAML.")

    cfg = copy.deepcopy(DEFAULTS)
    cfg.update({"project_name": "auto_mapped_project", "scorecard_type": "behavioral"})
    cfg["data"] = {}
    for role, v in roles.items():
        cfg["data"][role] = {"path": paths[v["file"]], "columns": dict(v["columns"])}

    # ---- as_of_date guess: latest actual payment (schedules run into the future, so due_date is a bad proxy) ----
    try:
        rep = raw[roles["repayments"]["file"]]
        rc = roles["repayments"]["columns"]
        col = rc.get("paid_date") or rc.get("due_date")
        latest = pd.to_datetime(rep[col], errors="coerce").max()
        cfg["as_of_date"] = str(pd.Timestamp(latest).date())
        notes.append(f"as_of_date set to {cfg['as_of_date']} (latest payment date in repayments) - "
                     "adjust to your real extract date.")
    except Exception:
        cfg["as_of_date"] = str(pd.Timestamp.today().date())
        notes.append("Could not infer as_of_date - please set it.")

    # ---- features ----
    keys = {"loan_id", "customer_id", "installment_no", "due_date", "paid_date", "due_amount", "paid_amount",
            "disbursement_date", "snapshot_date"}
    suspects = {s.lower() for s in leak_suspects}
    num, cat = [], []
    for role in ("customers", "loans", "bureau"):
        if role not in roles:
            continue
        df = raw[roles[role]["file"]]
        mapped = roles[role]["columns"]
        raw_to_canon = {v: k for k, v in mapped.items()}
        for c in df.columns:
            canon = raw_to_canon.get(c, c)
            if canon in keys or _norm(c).endswith("id") and df[c].nunique() > 0.5 * len(df):
                continue
            if c.lower() in suspects:
                notes.append(f"'{c}' looks like a post-outcome (leaky) field - excluded.")
                continue
            if any(h in c.lower() for h in LEAKY_HINTS):
                notes.append(f"'{c}' matches a leakage pattern - excluded.")
                continue
            name = f"bureau_{canon}" if role == "bureau" else canon
            if pd.api.types.is_numeric_dtype(df[c]):
                num.append(name)
            elif df[c].nunique() <= 30:
                cat.append(name)
    if "installment_amount" not in num:
        num.append("installment_amount")
    cfg["features"]["numeric"], cfg["features"]["categorical"] = sorted(set(num)), sorted(set(cat))

    ratios = {}
    base_cols = set(num)
    if "monthly_income" in base_cols:
        ratios["installment_to_income"] = ["installment_amount", "monthly_income"]
        if "principal" in base_cols:
            ratios["loan_to_income"] = ["principal", "monthly_income"]
        if "bureau_total_outstanding" in base_cols:
            ratios["bureau_debt_to_income"] = ["bureau_total_outstanding", "monthly_income"]
    cfg["features"]["ratios"] = ratios
    return {"config": cfg, "notes": notes, "source": source}


# =====================================================================
# 2) Model review memo
# =====================================================================


def results_summary(res: dict) -> dict:
    """Aggregate-only facts about the run (safe to send to an LLM)."""
    m = res["metrics"].set_index("set")
    iv = res["iv_table"]
    leak = res["leakage"]
    te = res["deciles"]["test"]
    return {
        "scorecard_type": res["target_summary"]["type"],
        "target": {k: res["target_summary"][k] for k in ("bad_dpd", "obs", "perf", "bad_rate")},
        "population": {k: res["target_summary"][k] for k in ("n_model", "n_train", "n_test")},
        "waterfall": res["waterfall"][["step", "remaining"]].to_dict("records"),
        "metrics": {s: {k: round(float(v), 3) for k, v in m.loc[s].items() if k != "n"} for s in m.index},
        "top_iv": iv.head(8)[["feature", "iv", "strength"]].round(3).to_dict("records"),
        "selected_features": res["selected"],
        "leakage_flagged": leak[leak["flagged"]][["feature", "reason"]].to_dict("records"),
        "score_psi": round(res["score_psi"], 3),
        "unstable_features": res["feature_psi"][res["feature_psi"]["psi"] >= 0.1].round(3).to_dict("records"),
        "test_top_decile_bad_rate": round(float(te["bad_rate"].iloc[0]), 3),
        "test_bottom_decile_bad_rate": round(float(te["bad_rate"].iloc[-1]), 3),
        "calibration_gap_test": round(float((te["avg_pd"] - te["bad_rate"]).abs().mean()), 3),
        "warnings": res["warnings"],
    }


def _rating(v: float, cuts: tuple[float, float, float], labels=("weak", "fair", "good", "strong")) -> str:
    return labels[sum(v >= c for c in cuts)]


def offline_review(s: dict) -> str:
    tr, te = s["metrics"]["train"], s["metrics"]["test"]
    lines = [f"## Model review ({s['scorecard_type']} scorecard)", ""]
    lines.append(f"**Discrimination:** test AUC {te['auc']:.3f} / Gini {te['gini']:.3f} / KS {te['ks']:.3f} - "
                 f"overall **{_rating(te['auc'], (0.6, 0.7, 0.8))}**.")
    gap = tr["auc"] - te["auc"]
    lines.append(f"**Generalisation:** train-test AUC gap {gap:+.3f} - "
                 + ("looks fine." if gap < 0.04 else "possible overfitting, consider fewer features or stronger regularisation."))
    lines.append(f"**Stability:** score PSI {s['score_psi']:.3f} - "
                 + ("stable." if s["score_psi"] < 0.1 else "moderate shift, monitor." if s["score_psi"] < 0.25
                    else "significant shift, investigate before use."))
    lines.append(f"**Ranking:** riskiest test decile bad rate {s['test_top_decile_bad_rate']:.1%} vs safest "
                 f"{s['test_bottom_decile_bad_rate']:.1%}; mean calibration gap {s['calibration_gap_test']:.1%}.")
    lines += ["", "### Watch-outs"]
    flagged = s["leakage_flagged"]
    if flagged:
        lines.append("- Leakage flagged and handled: " + ", ".join(f"`{f['feature']}`" for f in flagged)
                     + ". Confirm with the data owner when each field is populated.")
    if s["unstable_features"]:
        lines.append("- Drifting features: " + ", ".join(f"`{f['feature']}` (PSI {f['psi']})" for f in s["unstable_features"]))
    if s["scorecard_type"] == "behavioral":
        lines.append("- Behavioural features dominate; the model only applies to accounts that have completed the "
                     f"{s['target']['obs']}-month observation window.")
    if te["ks"] > 0.6 or te["auc"] > 0.9:
        lines.append("- Performance is unusually high - re-check for leakage before trusting it.")
    if s["target"]["bad_rate"] < 0.03:
        lines.append("- Very few bads; metrics will be noisy. Consider a longer window or a milder bad definition.")
    for w in s["warnings"]:
        lines.append(f"- Config warning: {w}")
    lines += ["", "### Suggested next steps",
              "- Calibrate cut-offs against business economics (approval rate vs expected loss).",
              "- Run the same pipeline on a later out-of-time sample and compare KS / PSI.",
              "- Review reason codes with credit policy before using them in decline letters."]
    return "\n".join(lines)


def model_review(res: dict, llm: LLM | None = None) -> tuple[str, str]:
    s = results_summary(res)
    if llm is not None and llm.available:
        try:
            system = ("You are a senior credit risk model validator. Write a concise, honest model review memo in "
                      "markdown for a risk committee. Cover: fit for purpose, discrimination, generalisation, stability, "
                      "calibration, leakage risk, key drivers, limitations and concrete next steps. Do not invent numbers; "
                      "use only the JSON. Flag anything suspicious. Max ~350 words.")
            return llm.complete(system, json.dumps(s), max_tokens=1200), "llm"
        except Exception as e:
            return offline_review(s) + f"\n\n_(LLM call failed: {type(e).__name__}; showing template review.)_", "offline"
    return offline_review(s), "offline"


# =====================================================================
# 3) Reason codes -> plain-language letter
# =====================================================================

REASON_TEXT = {
    "installment_to_income": "The monthly installment is high relative to income",
    "loan_to_income": "The loan amount is high relative to income",
    "bureau_debt_to_income": "Existing debt is high relative to income",
    "bureau_max_collectibility": "Credit bureau shows past payment problems",
    "bureau_inquiries_3m": "Many recent credit applications",
    "bureau_active_loans": "Number of active loans",
    "bureau_has_writeoff": "A previous write-off appears on the credit record",
    "employer_tenure_months": "Short time with current employer",
    "is_repeat_borrower": "No prior repayment history with us",
    "tenor_months": "Loan term length",
    "obs_max_dpd": "Late payments on recent installments",
    "obs_mean_dpd": "Average delay on recent installments",
    "obs_late_count": "Number of late installments",
    "obs_paid_ratio": "Share of due amount paid so far is low",
    "obs_first_inst_dpd": "First installment was paid late",
}


def offline_reason_lines(reasons: list[dict]) -> list[str]:
    out = []
    for r in reasons:
        text = REASON_TEXT.get(r["feature"], r["feature"].replace("_", " ").capitalize())
        out.append(f"{text} (your value falls in '{r['bin']}', about {r['points_lost']:.0f} points below the best band)")
    return out


def reason_letter(loan_id: str, score: float, pd_: float, reasons: list[dict], llm: LLM | None = None) -> tuple[str, str]:
    if llm is not None and llm.available:
        try:
            system = ("You write clear, respectful adverse-action explanations for loan applicants. Plain English, "
                      "no jargon, no blame, no mention of internal model names or point values. Never promise "
                      "approval. Max 140 words. Use only the factors provided.")
            user = json.dumps({"risk_score": round(score), "probability_of_default": round(pd_, 3),
                               "main_factors": [{"factor": r["feature"], "band": r["bin"]} for r in reasons]})
            return llm.complete(system, user, max_tokens=500), "llm"
        except Exception:
            pass
    lines = offline_reason_lines(reasons)
    txt = ("Thank you for your application. The main factors that affected our assessment were:\n\n"
           + "\n".join(f"{i}. {l.split(' (')[0]}" for i, l in enumerate(lines, 1))
           + "\n\nYou are welcome to re-apply once these areas improve.")
    return txt, "offline"


# =====================================================================
# 4) Ask the results
# =====================================================================


def ask_results(question: str, res: dict, llm: LLM | None = None) -> str:
    if llm is None or not llm.available:
        return "Connect an LLM in the sidebar to ask free-form questions. The **Model review** tab works offline."
    system = ("You answer questions about a credit-risk scorecard run. Use ONLY the JSON summary provided. "
              "If the answer is not in it, say so. Be concise and specific.")
    ctx = json.dumps(results_summary(res))
    try:
        return llm.complete(system, f"Run summary:\n{ctx}\n\nQuestion: {question}", max_tokens=800)
    except Exception as e:
        return f"LLM call failed: {type(e).__name__}: {e}"
