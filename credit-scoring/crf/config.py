"""Config loading, defaults and validation."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

SCORECARD_TYPES = ("application", "behavioral")

DEFAULTS = {
    "project_name": "project",
    "scorecard_type": "application",
    "as_of_date": None,
    "data": {},
    "target": {"bad_dpd": 90, "observation_months": 3, "performance_months": 12, "exclude_if_obs_dpd_ge": 90},
    "features": {"numeric": [], "categorical": [], "ratios": {}, "rolling_windows": [1, 2]},
    "binning": {"max_bins": 8, "min_bin_pct": 0.05, "monotonic": True},
    "selection": {"min_iv": 0.02, "max_corr": 0.7, "max_features": 12, "C": 1.0},
    "leakage": {"max_iv": 2.0,
                "banned_patterns": ["collection", "legal", "recovery", "writeoff_after", "charge_off"],
                "auto_drop": True},
    "split": {"mode": "out_of_time", "test_fraction": 0.3, "seed": 42},
    "scorecard": {"base_score": 600, "base_odds": 50, "pdo": 20},
}

REQUIRED_TABLES = ("loans", "repayments")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(src, base_dir: str | Path | None = None) -> dict:
    """src: path to a YAML file, YAML text, or an already-parsed dict."""
    raw, folder = None, base_dir
    if isinstance(src, dict):
        raw = src
    else:
        text, p = str(src), None
        try:
            p = Path(text)
            is_file = p.exists() and p.is_file()
        except (OSError, ValueError):
            is_file = False
        if is_file:
            raw, folder = yaml.safe_load(p.read_text()), folder or p.parent
        else:
            raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    cfg = _merge(DEFAULTS, raw)
    cfg["_base_dir"] = str(folder or Path.cwd())
    return validate(cfg)


def validate(cfg: dict) -> dict:
    if cfg["scorecard_type"] not in SCORECARD_TYPES:
        raise ValueError(f"scorecard_type must be one of {SCORECARD_TYPES}")
    if not cfg.get("as_of_date"):
        raise ValueError("as_of_date is required (e.g. '2025-12-31').")
    for t in REQUIRED_TABLES:
        if t not in cfg["data"]:
            raise ValueError(f"data.{t} is required.")
    tg = cfg["target"]
    for k in ("bad_dpd", "observation_months", "performance_months", "exclude_if_obs_dpd_ge"):
        tg[k] = int(tg[k])
    if cfg["scorecard_type"] == "application":
        tg["observation_months"] = 0
    if tg["performance_months"] <= 0:
        raise ValueError("target.performance_months must be > 0.")
    sp = cfg["split"]
    if sp["mode"] not in ("out_of_time", "random"):
        raise ValueError("split.mode must be out_of_time or random.")
    if not 0.05 <= float(sp["test_fraction"]) <= 0.6:
        raise ValueError("split.test_fraction should be between 0.05 and 0.6.")
    return cfg
