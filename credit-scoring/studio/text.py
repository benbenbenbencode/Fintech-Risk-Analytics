"""Plain-language labels so business users never see raw column names."""
from __future__ import annotations

PRODUCT_NAMES = {"CARD": "Credit Card", "LOAN": "Term Loan", "BNPL": "BNPL"}
PRODUCT_ICONS = {"CARD": "💳", "LOAN": "🏦", "BNPL": "🛍️"}
BAND_NAMES = {"A": "A · Very low risk", "B": "B · Low", "C": "C · Medium", "D": "D · High", "E": "E · Very high",
              "NPL": "NPL · Already 90+ DPD", "NEW": "New"}
BAND_COLORS = {"A": "#2e9e6b", "B": "#7bc47f", "C": "#f2c14e", "D": "#ee8434", "E": "#d1495b", "NPL": "#6b1f2a"}
BUCKET_ORDER = ["Current", "1-29", "30-59", "60-89", "90+", "Written off"]
BUCKET_COLORS = {"Current": "#2e9e6b", "1-29": "#a8d08d", "30-59": "#f2c14e", "60-89": "#ee8434", "90+": "#d1495b",
                 "Written off": "#6b1f2a"}

LABELS = {
    "mob": "Months since opening", "utilization": "Credit utilization", "util_chg_3m": "Utilization change (3m)",
    "bal_growth_3m": "Balance growth (3m)", "dpd_now": "Current days past due", "max_dpd_3m": "Worst delay, last 3 months",
    "max_dpd_6m": "Worst delay, last 6 months", "late_months_6m": "Late months, last 6 months",
    "pay_ratio_3m": "Share of amount due that was paid (3m)", "spend_trend": "Spending trend",
    "orders_3m": "BNPL orders, last 3 months", "limit_chg_6m": "Limit change (6m)",
    "balance_to_income": "Balance relative to monthly income", "limit_to_income": "Limit relative to monthly income",
    "income": "Monthly income", "age": "Age", "bureau_score": "Credit bureau score",
    "cust_n_active": "Number of active credit products", "other_delinq": "Late on another product",
    "macro_stress": "Local economic stress", "country": "Country", "employment_type": "Employment type",
    "acquisition_channel": "How the customer was acquired", "monthly_income_usd": "Monthly income (USD)",
    "education": "Education", "residence_type": "Housing situation", "household_size": "Household size",
    "years_at_job": "Years at current job", "device_os": "Phone type", "bureau_history_months": "Credit history length",
    "bureau_active_accounts": "Active accounts at bureau", "bureau_inquiries_6m": "Recent credit inquiries (6m)",
    "bureau_max_dpd_12m": "Worst bureau delay (12m)", "email_age_days": "Age of email address",
    "device_shared_apps_30d": "Applications from the same device (30d)", "ip_country_match": "Location matches country",
    "requested_amount_usd": "Requested amount", "req_to_income": "Requested amount vs income",
    "tenor": "Loan term (months)", "thin_file": "No credit bureau record", "macro_stress_app": "Local economic stress",
    "installment_to_income": "Installment vs income", "city_tier": "City size tier",
    "bucket": "Delinquency stage", "balance_usd": "Outstanding balance", "prior_contact_rate": "Past contact success rate",
    "product": "Product",
}


def label(feature: str) -> str:
    return LABELS.get(feature, feature.replace("_", " ").capitalize())


def reason_phrase(feature: str, bin_label: str) -> str:
    return f"{label(feature)}: {bin_label}"


def band_for_pd(p: float) -> str:
    for b, cut in (("A", .015), ("B", .035), ("C", .08), ("D", .18)):
        if p < cut:
            return b
    return "E"
