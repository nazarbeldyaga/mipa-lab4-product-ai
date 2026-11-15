"""2. Генерація гіпотез.

Спочатку pandas рахує набір «сигналів» - порівнянь сегментів, які можуть бути
цікавими (конверсія по платформах, віку, країнах, тривалість сесій, день тижня...).
Кожен сигнал отримує id. LLM вибирає з них найцікавіші і формулює гіпотези, а
код перевіряє, що гіпотези посилаються на існуючі факти і не містять вигаданих чисел.
"""
import pandas as pd

from . import prompts
from .data import in_period
from .factcheck import check_text, facts_json

WEEKDAYS = ["понеділок", "вівторок", "середа", "четвер", "пʼятниця", "субота", "неділя"]


def _segment_table(sess_users, payers, revenue, dim, min_users=None):
    """Активні користувачі, частка платників і дохід на активного по сегментах dim."""
    g = sess_users.groupby(dim)["user_id"].nunique().rename("active_users").to_frame()
    g["payers"] = payers.groupby(dim)["user_id"].nunique()
    g["revenue"] = revenue.groupby(dim)["amount_usd"].sum()
    g = g.fillna(0)
    if min_users is None:
        min_users = max(30, int(0.005 * sess_users["user_id"].nunique()))
    g = g[g["active_users"] >= min_users]
    g["payer_conversion_pct"] = (100 * g["payers"] / g["active_users"]).round(2)
    g["revenue_per_active_usd"] = (g["revenue"] / g["active_users"]).round(3)
    return g.sort_values("payer_conversion_pct", ascending=False)


def _rows(df, cols):
    out = []
    for idx, r in df.iterrows():
        d = {"segment": str(idx)}
        for c in cols:
            v = r[c]
            d[c] = int(v) if float(v).is_integer() and c in ("active_users", "payers", "users", "sessions") else round(float(v), 2)
        out.append(d)
    return out


def compute_signals(data, start="2024-04-05", end="2024-05-11"):
    s = data["sessions"][in_period(data["sessions"]["session_start_dt"], start, end)]
    p = data["payments"][in_period(data["payments"]["pay_dt"], start, end)]
    u = data["users"]
    facts = []

    def add(topic, payload):
        facts.append({"id": f"F{len(facts) + 1:02d}", "topic": topic, **payload})

    total_active = s["user_id"].nunique()
    total_payers = p["user_id"].nunique()
    overall_conv = round(100 * total_payers / total_active, 2)

    for dim, name in [("platform", "платформа"), ("age_group", "вік"), ("gender", "стать")]:
        t = _segment_table(s, p, p, dim)
        best, worst = t.iloc[0], t.iloc[-1]
        add(f"Частка платників серед активних, розріз: {name}", {
            "overall_payer_conversion_pct": overall_conv,
            "segments": _rows(t, ["active_users", "payers", "payer_conversion_pct", "revenue_per_active_usd"]),
            "best_vs_worst_ratio": round(best["payer_conversion_pct"] / worst["payer_conversion_pct"], 2)
            if worst["payer_conversion_pct"] else None,
        })

    top_c = s.groupby("country_name")["user_id"].nunique().nlargest(10).index
    t = _segment_table(s[s["country_name"].isin(top_c)], p[p["country_name"].isin(top_c)],
                       p[p["country_name"].isin(top_c)], "country_name")
    add("Топ-10 країн за активними користувачами: частка платників і дохід на активного", {
        "overall_payer_conversion_pct": overall_conv,
        "overall_revenue_per_active_usd": round(p["amount_usd"].sum() / total_active, 3),
        "segments": _rows(t, ["active_users", "payer_conversion_pct", "revenue_per_active_usd"]),
    })

    for dim, name in [("platform", "платформа"), ("age_group", "вік")]:
        g = s.groupby(dim).agg(sessions=("user_id", "size"), users=("user_id", "nunique"),
                               avg_session_min=("duration_min", "mean"))
        g = g[g["users"] >= max(30, int(0.005 * total_active))]
        g["sessions_per_user"] = g["sessions"] / g["users"]
        add(f"Тривалість і частота сесій, розріз: {name}", {
            "overall_avg_session_min": round(s["duration_min"].mean(), 2),
            "overall_sessions_per_user": round(len(s) / total_active, 2),
            "segments": _rows(g.sort_values("avg_session_min"), ["users", "avg_session_min", "sessions_per_user"]),
        })

    new = u[u["is_new"] & in_period(u["reg_dt"], start, end)]
    per_day = new.groupby(new["reg_dt"].dt.date)["user_id"].nunique()
    first_week = per_day.iloc[:7].mean()
    last_week = per_day.iloc[-8:-1].mean()  # останній день неповний, тому без нього
    add("Нові реєстрації: перший і останній повний тиждень", {
        "first_week": f"{per_day.index[0]} - {per_day.index[6]}",
        "first_week_avg_per_day": round(first_week, 1),
        "last_full_week": f"{per_day.index[-8]} - {per_day.index[-2]}",
        "last_week_avg_per_day": round(last_week, 1),
        "growth_ratio": round(last_week / first_week, 2),
        "peak_day": str(per_day.idxmax()), "peak_day_new_users": int(per_day.max()),
    })

    full_days = p[p["pay_dt"].dt.date < pd.Timestamp(end).date()]
    rev_day = full_days.groupby(full_days["pay_dt"].dt.date)["amount_usd"].sum()
    rev_day.index = pd.to_datetime(rev_day.index)
    # виручка росте разом з аудиторією, тому день тижня порівнюю відносно сусідніх днів
    rel = rev_day / rev_day.rolling(7, center=True, min_periods=4).mean()
    wd = rel.groupby(rel.index.dayofweek).mean()
    add("Виручка за днями тижня відносно ковзного 7-денного середнього (1.0 = звичайний день)", {
        "by_weekday": {WEEKDAYS[i]: round(float(v), 3) for i, v in wd.items()},
        "strongest_day": WEEKDAYS[int(wd.idxmax())], "weakest_day": WEEKDAYS[int(wd.idxmin())],
    })

    vip = p.groupby("platform")["feature_type_name"].apply(lambda x: (x == "vip").mean() * 100)
    cnt = p.groupby("platform").size()
    add("Частка VIP-покупок серед платежів за платформою (решта premium)", {
        "overall_vip_share_pct": round(100 * (p["feature_type_name"] == "vip").mean(), 2),
        "overall_avg_payment_usd": round(p["amount_usd"].mean(), 2),
        "segments": [{"segment": k, "payments": int(cnt[k]), "vip_share_pct": round(v, 2)}
                     for k, v in vip.sort_values().items() if cnt[k] >= 100],
    })

    dau = s.groupby(s["session_start_dt"].dt.date)["user_id"].nunique()
    dau = dau.iloc[:-1]  # останній день неповний
    jump = dau.pct_change() * 100
    peak = dau.idxmax()
    ab_start = data["splits"]["dt"].min().date() if "splits" in data else None
    add("Денні активні користувачі (DAU): різкі стрибки і тренд", {
        "dau_first_day": int(dau.iloc[0]), "dau_peak": int(dau.max()), "dau_peak_day": str(peak),
        "dau_last_full_day": int(dau.iloc[-1]), "last_full_day": str(dau.index[-1]),
        "biggest_daily_jump_day": str(jump.idxmax()), "biggest_daily_jump_pct": round(float(jump.max()), 1),
        "dau_day_before_jump": int(dau.shift(1)[jump.idxmax()]), "dau_on_jump_day": int(dau[jump.idxmax()]),
        "decline_from_peak_pct": round(100 * (dau.iloc[-1] / dau.max() - 1), 1),
        "ab_test_start_day": str(ab_start),
    })

    first_pay = p.groupby("user_id")["pay_dt"].min()
    nn = new.set_index("user_id")["reg_dt"]
    lag = (first_pay.reindex(nn.index) - nn).dropna()
    lag = lag[lag >= pd.Timedelta(0)]
    add("Скільки часу проходить від реєстрації нового користувача до першої оплати", {
        "new_users": int(len(nn)), "new_users_who_paid": int(len(lag)),
        "new_user_payer_pct": round(100 * len(lag) / len(nn), 2),
        "paid_within_1_hour_pct_of_payers": round(100 * (lag <= pd.Timedelta(hours=1)).mean(), 2),
        "paid_within_1_day_pct_of_payers": round(100 * (lag <= pd.Timedelta(days=1)).mean(), 2),
        "median_hours_to_first_payment": round(lag.median() / pd.Timedelta(hours=1), 2),
    })
    return facts


def suggest_hypotheses(data, llm, n=7):
    facts = compute_signals(data)
    task = (f"Запропонуй {n} гіпотез (не менше 5 і не більше 10), які варто перевірити, "
            f"відсортуй від найважливішої.")
    ans = llm.structured(prompts.HYP_SYSTEM, prompts.user_message(task, facts_json(facts)),
                         prompts.HYP_SCHEMA, "hypotheses")
    by_id = {f["id"]: f for f in facts}
    warnings = []
    for i, h in enumerate(ans.get("hypotheses", []), 1):
        ids = [x for x in h.get("fact_ids", []) if x in by_id]
        missing = [x for x in h.get("fact_ids", []) if x not in by_id]
        if missing:
            warnings.append(f"Гіпотеза {i}: посилання на неіснуючі факти {missing}")
        if not ids:
            warnings.append(f"Гіпотеза {i}: не спирається на жоден факт")
        cited = [by_id[x] for x in ids] or facts
        bad = check_text(h.get("statement", "") + " " + h.get("why_it_matters", ""), cited)
        if bad:
            warnings.append(f"Гіпотеза {i}: числа {', '.join(f'{b:g}' for b in bad)} не знайдені у фактах {ids}")
    return {"kind": "hypotheses", "facts": facts, "llm": ans, "warnings": warnings}
