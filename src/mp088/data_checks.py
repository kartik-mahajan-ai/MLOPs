"""Find the seven kinds of data problem in train.csv.

Run from the repo root:   python -m mp088.data_checks   (with src/ on PYTHONPATH)
Every function takes the raw DataFrame and returns a dict shaped like the
matching block in reports/findings.yaml.
"""
import re

import numpy as np
import pandas as pd

from mp088.config import load_csv, load_params


def find_leaking_column(train, holdout, cfg):
    """A column that is in train but missing from holdout, and ~equals the target."""
    target = cfg["columns"]["target"]
    for col in train.columns:
        if col == target or col in holdout.columns:
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            corr = train[col].corr(train[target])
            if abs(corr) >= cfg["checks"]["leak_corr"]:
                return {"present": True, "column": col, "corr": round(corr, 4)}
    return {"present": False, "column": None}


def find_unit_mixup(train, cfg):
    """One group's values are many times bigger than the other groups'.

    1. Ratio of the group's median to the other groups' median > 5 -> suspicious.
    2. The factor is the exact unit conversion nearest to that rough ratio
       (compared on a log scale) ...
    3. ... confirmed by checking the group's values divide exactly by it
       (e.g. minutes typed as seconds are all multiples of 60).
    """
    group = cfg["columns"]["group"]
    for col in cfg["columns"]["numeric"]:
        medians = train.groupby(group)[col].median()
        for g, m in medians.items():
            others = medians.drop(g).median()
            if others <= 0 or m / others < 5:
                continue
            ratio = m / others
            factor = min(cfg["checks"]["unit_factors"],
                         key=lambda f: abs(np.log(ratio / f)))
            vals = train.loc[train[group] == g, col]
            share_exact = float((vals % factor == 0).mean())
            return {"present": True, "column": col, "group_value": g,
                    "factor": factor, "ratio_seen": round(ratio, 2),
                    "share_divisible": round(share_exact, 3)}
    return {"present": False, "column": None, "group_value": None, "factor": None}


def find_sentinel_value(train, cfg):
    """An impossible special number (e.g. -999) repeated many times."""
    for col in cfg["columns"]["numeric"]:
        counts = train[col].value_counts()
        for v in cfg["checks"]["sentinel_candidates"]:
            if counts.get(v, 0) >= 3:
                return {"present": True, "column": col, "value": v,
                        "count": int(counts[v])}
    return {"present": False, "column": None, "value": None, "count": None}


def find_duplicate_rows(train, cfg):
    """Number of EXTRA copies (rows we would delete)."""
    n = int(train.duplicated().sum())
    return {"present": n > 0, "count": n if n else None}


def find_stuck_sensor(train, cfg):
    """After sorting by date, the longest run of one repeated value in a column."""
    date = cfg["columns"]["date"]
    df = train.drop_duplicates().sort_values([date, cfg["columns"]["id"]],
                                             kind="stable").reset_index(drop=True)
    best = None
    for col in cfg["columns"]["numeric"]:
        run_id = (df[col] != df[col].shift()).cumsum()  # new id each time value changes
        runs = df.groupby(run_id).agg(value=(col, "first"), length=(col, "size"),
                                      start=(date, "min"), end=(date, "max"))
        top = runs.loc[runs["length"].idxmax()]
        if top["length"] >= cfg["checks"]["stuck_min_run"] and (
                best is None or top["length"] > best["run_length"]):
            best = {"present": True, "column": col, "start_date": top["start"],
                    "end_date": top["end"], "value": top["value"].item(),
                    "run_length": int(top["length"])}
    return best or {"present": False, "column": None, "start_date": None, "end_date": None}


def find_target_typos(train, cfg):
    """Targets typed with an extra zero sit far above the rest (> 5x the median)."""
    y = train[cfg["columns"]["target"]]
    top = y.sort_values().tail(5).tolist()
    n = int((y > 5 * y.median()).sum())
    return {"present": n > 0, "count": n if n else None, "largest_values": top}


def _normalise(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).strip().lower())


def find_spelling_variants(train, later, cfg):
    """More distinct spellings in train than real categories (later.csv is clean)."""
    for col in cfg["columns"]["categorical"]:
        raw = train[col].nunique()
        true_levels = later[col].nunique()
        if raw > true_levels:
            return {"present": True, "column": col, "true_levels": int(true_levels),
                    "raw_levels": int(raw),
                    "normalised_levels": train[col].map(_normalise).nunique()}
    return {"present": False, "column": None, "true_levels": None}


def run_all_checks():
    cfg = load_params()
    train, later, holdout = load_csv("train", cfg), load_csv("later", cfg), load_csv("holdout", cfg)
    return {
        "leaking_column": find_leaking_column(train, holdout, cfg),
        "unit_mixup": find_unit_mixup(train, cfg),
        "sentinel_value": find_sentinel_value(train, cfg),
        "duplicate_rows": find_duplicate_rows(train, cfg),
        "stuck_sensor": find_stuck_sensor(train, cfg),
        "target_typos": find_target_typos(train, cfg),
        "spelling_variants": find_spelling_variants(train, later, cfg),
    }


if __name__ == "__main__":
    for name, result in run_all_checks().items():
        print(f"{name:18s} {result}")
