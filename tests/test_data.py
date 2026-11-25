"""Tests for cleaning and data checks, on small data built here (no CSVs needed)."""
import numpy as np
import pandas as pd

from mp088.cleaning import clean_train
from mp088.config import load_params
from mp088.data_checks import find_duplicate_rows, find_sentinel_value, find_unit_mixup

CFG = load_params()


def tiny_train():
    return pd.DataFrame({
        "record_id": ["A", "A", "F", "B", "C", "D", "E"],
        "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-03-30", "2026-03-30",
                 "2026-05-01", "2026-05-02"],
        "loom_age_yr": [5, 5, 7, 10, 15, 20, 25],
        "thread_count": [100, 100, 110, 95, 120, 95, 130],
        # Shed 1 typed seconds: 3000 s = 50 min, 1800 s = 30 min
        "power_cuts_min": [3000.0, 3000.0, 1800.0, 40.0, 60.0, 30.0, 50.0],
        "weaver_exp_yr": [1, 1, 2, 2, 3, 4, 5],
        "humidity_pct": [50, 50, 55, -999, 60, 70, 80],
        "shift_hours": [8.0, 8.0, 8.5, 9.0, 10.0, 11.0, 12.0],
        "fabric": ["cotton"] * 7,
        "shed": ["Shed 1", "Shed 1", "Shed 1", "Shed 2", "Shed 3", "Shed 2", "Shed 3"],
        "metres_checked_at_folding": [40.0] * 7,
        "metres_per_shift": [40.0] * 7,
    })


def test_clean_train_fixes_every_problem():
    out = clean_train(tiny_train(), CFG)
    assert len(out) == 6                                   # duplicate removed
    assert "metres_checked_at_folding" not in out.columns  # leak dropped
    assert out.loc[out.record_id == "A", "power_cuts_min"].item() == 50.0  # 3000 s / 60
    assert out["humidity_pct"].isna().sum() == 1           # -999 -> missing
    assert np.isnan(out.loc[out.record_id == "B", "thread_count"].item())  # 95 in stuck window
    assert out.loc[out.record_id == "D", "thread_count"].item() == 95      # 95 outside window kept


def test_checks_find_planted_problems():
    df = tiny_train()
    assert find_duplicate_rows(df, CFG)["count"] == 1
    df.loc[[4, 5, 6], "humidity_pct"] = -999   # a sentinel must repeat (>= 3 times)
    s = find_sentinel_value(df, CFG)
    assert (s["column"], s["value"], s["count"]) == ("humidity_pct", -999, 4)
    u = find_unit_mixup(df, CFG)
    assert (u["column"], u["group_value"], u["factor"]) == ("power_cuts_min", "Shed 1", 60)
