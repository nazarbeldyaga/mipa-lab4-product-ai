"""4. Пояснення зміни метрики між двома періодами.

pandas розкладає зміну метрики за сегментами (платформа, країна, стать, вік) і
знаходить, хто дав найбільший внесок. Для сум/кількостей внесок сегмента - це
просто різниця. Для метрик-відношень (середня сесія, частка платників) зміна
ділиться на mix (змінилась частка сегмента в аудиторії) і rate (змінилось
значення всередині сегмента):
    total = sum_i (w1_i - w0_i) * r0_i  +  sum_i w1_i * (r1_i - r0_i)
LLM отримує готовий розклад і формулює гіпотези причин.
"""
import pandas as pd

from . import prompts
from .data import in_period
from .factcheck import check_answer, facts_json

METRICS = {
    # назва: (тип, опис)
    "new_users": ("sum", "нові користувачі (реєстрації)"),
    "active_users": ("sum", "активні користувачі (мали сесію)"),
    "sessions": ("sum", "кількість сесій"),
    "revenue": ("sum", "виручка, $"),
    "payers": ("sum", "платники"),
    "avg_session_min": ("ratio", "середня тривалість сесії, хв"),
    "payer_conversion": ("ratio", "частка платників серед активних, %"),
}
DIMENSIONS = {"platform": "платформа", "country_name": "країна", "gender": "стать", "age_group": "вік"}


def _parts(data, metric, start, end, dim):
    """Повертає (num, den) по сегментах; для sum den=None."""
    u, p, s = data["users"], data["payments"], data.get("sessions")
    if metric == "new_users":
        x = u[u["is_new"] & in_period(u["reg_dt"], start, end)]
        return x.groupby(dim)["user_id"].nunique(), None
    if metric == "revenue":
        x = p[in_period(p["pay_dt"], start, end)]
        return x.groupby(dim)["amount_usd"].sum(), None
    if metric == "payers":
        x = p[in_period(p["pay_dt"], start, end)]
        return x.groupby(dim)["user_id"].nunique(), None
    x = s[in_period(s["session_start_dt"], start, end)]
    if metric == "active_users":
        return x.groupby(dim)["user_id"].nunique(), None
    if metric == "sessions":
        return x.groupby(dim).size(), None
    if metric == "avg_session_min":
        return x.groupby(dim)["duration_min"].sum(), x.groupby(dim).size()
    if metric == "payer_conversion":
        y = p[in_period(p["pay_dt"], start, end)]
        return y.groupby(dim)["user_id"].nunique() * 100, x.groupby(dim)["user_id"].nunique()
    raise ValueError(f"Невідома метрика {metric}. Доступні: {', '.join(METRICS)}")


def decompose(data, metric, base, target, dim, top=6):
    kind = METRICS[metric][0]
    n0, d0 = _parts(data, metric, *base, dim)
    n1, d1 = _parts(data, metric, *target, dim)
    idx = n0.index.union(n1.index)
    if d0 is not None:
        idx = idx.union(d0.index).union(d1.index)
    n0, n1 = n0.reindex(idx, fill_value=0), n1.reindex(idx, fill_value=0)
    t = pd.DataFrame(index=idx)
    if kind == "sum":
        t["base"], t["target"] = n0, n1
        t["contribution"] = n1 - n0
        total0, total1 = n0.sum(), n1.sum()
        t["mix"] = 0.0
        t["rate"] = t["contribution"]
    else:
        d0, d1 = d0.reindex(idx, fill_value=0), d1.reindex(idx, fill_value=0)
        w0, w1 = d0 / d0.sum(), d1 / d1.sum()
        r0 = (n0 / d0).where(d0 > 0, 0.0)
        r1 = (n1 / d1).where(d1 > 0, 0.0)
        r0 = r0.where(d0 > 0, r1)  # новий сегмент: rate-ефекту немає, тільки mix
        t["base"], t["target"] = r0, r1
        t["base_share_pct"], t["target_share_pct"] = 100 * w0, 100 * w1
        t["mix"] = (w1 - w0) * r0
        t["rate"] = w1 * (r1 - r0)
        t["contribution"] = t["mix"] + t["rate"]
        total0, total1 = n0.sum() / d0.sum(), n1.sum() / d1.sum()
    change = total1 - total0
    t["share_of_change_pct"] = 100 * t["contribution"] / change if change else 0.0
    t = t.reindex(t["contribution"].abs().sort_values(ascending=False).index)
    head = t.head(top)
    rest = t.iloc[top:]
    rows = []
    for seg, r in head.iterrows():
        row = {"segment": str(seg)}
        for c in t.columns:
            row[c] = round(float(r[c]), 3)
        rows.append(row)
    if len(rest):
        rows.append({"segment": f"інші ({len(rest)})", "contribution": round(float(rest["contribution"].sum()), 3),
                     "share_of_change_pct": round(float(rest["share_of_change_pct"].sum()), 2)})
    top_share = float(head["share_of_change_pct"].abs().iloc[0]) if len(head) else 0.0
    return {"dimension": DIMENSIONS[dim], "segments": rows, "top_segment_share_pct": round(top_share, 2)}, \
        round(float(total0), 3), round(float(total1), 3)


def explain_change(data, metric, base, target, llm, dims=tuple(DIMENSIONS)):
    """base і target - пари дат ('2024-04-22', '2024-04-28'), обидві включно."""
    if metric not in METRICS:
        raise ValueError(f"Невідома метрика {metric}. Доступні: {', '.join(METRICS)}")
    breakdowns = []
    for dim in dims:
        b, total0, total1 = decompose(data, metric, base, target, dim)
        breakdowns.append(b)
    change = total1 - total0
    facts = {
        "metric": metric, "metric_description": METRICS[metric][1],
        "metric_type": METRICS[metric][0],
        "base_period": f"{base[0]} - {base[1]}", "target_period": f"{target[0]} - {target[1]}",
        "base_value": total0, "target_value": total1,
        "change_abs": round(change, 3),
        "change_pct": round(100 * change / total0, 2) if total0 else None,
        "breakdowns": breakdowns,
    }
    if abs(change) < 1e-9:
        return {"kind": "change", "facts": facts, "llm": None, "warnings": ["Метрика не змінилась"]}
    task = (f"Поясни, чому метрика «{METRICS[metric][1]}» змінилась з {total0} до {total1} "
            f"між періодами. Назви головні драйвери і гіпотези причин.")
    ans = llm.structured(prompts.CHANGE_SYSTEM, prompts.user_message(task, facts_json(facts)),
                         prompts.CHANGE_SCHEMA, "change_explanation")
    return {"kind": "change", "facts": facts, "llm": ans, "warnings": check_answer(ans, facts)}
