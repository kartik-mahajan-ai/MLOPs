"""Clean train.csv using the problems found in data_checks.py.

later.csv and holdout_features.csv are already clean, so they only go
through `prepare` (drop the leaking column). The fixes in `clean_train`
apply to train.csv only: e.g. later.csv's Shed 1 power cuts are already
in minutes and must NOT be divided by 60.
"""
import numpy as np
import pandas as pd

FEATURES_NUM = ["loom_age_yr", "thread_count", "power_cuts_min",
                "weaver_exp_yr", "humidity_pct", "shift_hours"]
FEATURES_CAT = ["fabric", "shed"]
FEATURES = FEATURES_NUM + FEATURES_CAT


def prepare(df, cfg):
    """Drop the leaking column (it is only known after the shift)."""
    return df.drop(columns=[cfg["cleaning"]["leak_column"]], errors="ignore")


def clean_train(df, cfg):
    c = cfg["cleaning"]
    df = df.copy()

    # 1. duplicate rows: keep the first copy
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. leaking column
    df = prepare(df, cfg)

    # 3. unit mixup: Shed 1 typed power cuts in seconds -> back to minutes
    u = c["unit_mixup"]
    rows = df[cfg["columns"]["group"]] == u["group_value"]
    df.loc[rows, u["column"]] = df.loc[rows, u["column"]] / u["factor"]

    # 4. sentinel -999 means "no reading" -> missing
    s = c["sentinel"]
    df[s["column"]] = df[s["column"]].replace(s["value"], np.nan)

    # 5. stuck sensor: the repeated value inside the stuck window is not a real
    #    reading -> missing (the model's imputer fills it with the median)
    st = c["stuck"]
    in_window = df["date"].between(st["start_date"], st["end_date"])
    df.loc[in_window & (df[st["column"]] == st["value"]), st["column"]] = np.nan

    # target typos and spelling variants: not present in our train.csv
    return df


def sort_by_date(df):
    return df.sort_values(["date", "record_id"], kind="stable").reset_index(drop=True)
