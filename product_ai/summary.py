"""5. Executive summary: KPI за період проти попереднього такого ж періоду."""
import pandas as pd

from . import prompts
from .data import period_kpis
from .factcheck import check_answer, facts_json

KPI_NAMES = {
    "new_users": "нові користувачі", "active_users": "активні користувачі",
    "sessions": "сесії", "avg_session_min": "середня сесія, хв",
    "revenue": "виручка, $", "payers": "платники", "payments": "платежі",
    "arppu": "ARPPU, $", "payer_conversion_pct": "частка платників, %",
}


def previous_period(start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    days = (e - s).days + 1
    ps, pe = s - pd.Timedelta(days=days), s - pd.Timedelta(days=1)
    return ps.strftime("%Y-%m-%d"), pe.strftime("%Y-%m-%d")


def kpi_table(data, start, end):
    prev = previous_period(start, end)
    cur, old = period_kpis(data, start, end), period_kpis(data, *prev)
    rows = {}
    for k, name in KPI_NAMES.items():
        if k not in cur:
            continue
        a, b = old[k], cur[k]
        rows[k] = {"name": name, "previous": a, "current": b,
                   f"{k}_change_pct": round(100 * (b - a) / a, 2) if a else None}
    return {"period": f"{start} - {end}", "previous_period": f"{prev[0]} - {prev[1]}", "kpis": rows}


def exec_summary(data, start, end, llm, extra_facts=None):
    facts = kpi_table(data, start, end)
    if extra_facts:
        facts["extra"] = extra_facts
    task = (f"Напиши executive summary для продакт-менеджера за період {facts['period']} "
            f"у порівнянні з {facts['previous_period']}.")
    ans = llm.structured(prompts.SUMMARY_SYSTEM, prompts.user_message(task, facts_json(facts)),
                         prompts.SUMMARY_SCHEMA, "executive_summary")
    return {"kind": "summary", "facts": facts, "llm": ans, "warnings": check_answer(ans, facts)}
