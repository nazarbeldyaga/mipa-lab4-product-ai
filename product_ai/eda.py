"""1. Автоматичний EDA: pandas рахує профіль, LLM його інтерпретує."""
import numpy as np
import pandas as pd

from . import prompts
from .data import END, START
from .factcheck import check_answer, facts_json


def _num(x, nd=2):
    return None if pd.isna(x) else round(float(x), nd)


def profile_table(df, name="table", period=(START, END), max_top=5):
    """Профіль таблиці. Тут немає LLM, тільки pandas."""
    n = len(df)
    prof = {
        "table": name,
        "rows": int(n),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns_profile": [],
    }
    for col in df.columns:
        s = df[col]
        c = {
            "column": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 2) if n else 0.0,
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_bool_dtype(s):
            c["true_pct"] = round(100 * s.mean(), 2) if n else 0.0
        elif pd.api.types.is_datetime64_any_dtype(s):
            c["min"] = str(s.min())
            c["max"] = str(s.max())
            outside = s.notna() & ((s < period[0]) | (s >= period[1]))
            c["outside_period"] = int(outside.sum())
            c["outside_period_pct"] = round(100 * outside.mean(), 2) if n else 0.0
            if s.notna().any():
                by_day = s.dt.date.value_counts()
                c["busiest_day"] = str(by_day.idxmax())
                c["busiest_day_rows"] = int(by_day.max())
        elif pd.api.types.is_numeric_dtype(s):
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            c.update({
                "min": _num(s.min()), "p25": _num(q1), "median": _num(s.median()),
                "mean": _num(s.mean()), "p75": _num(q3), "max": _num(s.max()),
                "negative": int((s < 0).sum()),
                "zeros": int((s == 0).sum()),
                "iqr_outliers": int(((s < low) | (s > high)).sum()),
                "iqr_outliers_pct": round(100 * ((s < low) | (s > high)).mean(), 2) if n else 0.0,
            })
            if c["unique"] == n and n > 0:
                c["looks_like_id"] = True
        else:
            vc = s.astype(str).value_counts(dropna=True).head(max_top)
            c["top_values"] = {str(k): int(v) for k, v in vc.items()}
            c["top_share_pct"] = round(100 * vc.iloc[0] / n, 2) if n and len(vc) else 0.0
        prof["columns_profile"].append(c)
    return prof


def auto_eda(df, name, llm, period=(START, END)):
    """Профіль + інтерпретація від LLM + перевірка чисел у відповіді."""
    if df is None or len(df) == 0:
        return {"kind": "eda", "facts": {"table": name, "rows": 0}, "llm": None,
                "warnings": ["Таблиця порожня, аналізувати нічого"]}
    prof = profile_table(df, name, period)
    task = (f"Оціни якість даних у таблиці {name} і поясни розподіли ключових колонок. "
            f"Період спостереження: {period[0].date()} - {(period[1] - pd.Timedelta(days=1)).date()}.")
    ans = llm.structured(prompts.EDA_SYSTEM, prompts.user_message(task, facts_json(prof)),
                         prompts.EDA_SCHEMA, "eda_report")
    return {"kind": "eda", "facts": prof, "llm": ans, "warnings": check_answer(ans, prof)}
