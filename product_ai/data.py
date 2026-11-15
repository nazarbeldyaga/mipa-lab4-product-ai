"""Завантаження датасету мобільного застосунку і підготовка таблиць."""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# межі періоду спостереження: сесії й платежі є з 5 квітня по 11 травня 2024
START = pd.Timestamp("2024-04-05")
END = pd.Timestamp("2024-05-12")  # не включно

PLATFORMS = {"mobile": "Mobile web", "ios": "iOS", "android": "Android", "desktop": "Desktop"}
AGE_BINS = [0, 17, 24, 34, 44, 54, 200]
AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-54", "55+"]


def _read(name, data_dir, **kw):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    return pd.read_csv(path, **kw)


def load_data(data_dir=DATA_DIR, with_sessions=True):
    """Повертає dict з таблицями. users - довідник користувачів з атрибутами,
    які потім використовуються як розрізи (країна, платформа, стать, вік)."""
    # код Намібії "NA" pandas за замовчуванням читає як пропуск, тому тут NaN - тільки порожні рядки
    na = {"keep_default_na": False, "na_values": [""]}
    countries = _read("country_info_dict.csv", data_dir, **na).drop_duplicates("country_code")
    features = _read("feature_type_dict.csv", data_dir)

    users = _read("registrations.csv", data_dir, parse_dates=["reg_dt"], **na)
    users = users.merge(countries, on="country_code", how="left")
    users["country_name"] = users["country_name"].fillna("Unknown")
    users["platform"] = users["app"].map(PLATFORMS).fillna("Unknown")
    users["age_group"] = pd.cut(users["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    users["is_new"] = (users["reg_dt"] >= START) & (users["reg_dt"] < END)

    attrs = users[["user_id", "country_name", "platform", "gender", "age_group"]]

    payments = _read("payments.csv", data_dir, parse_dates=["pay_dt"])
    payments = payments.merge(features, on="feature_type_id", how="left")
    payments = payments.merge(attrs, on="user_id", how="left")

    splits = _read("user_split_groups.csv", data_dir, parse_dates=["dt"])

    data = {"users": users, "payments": payments, "splits": splits}
    if with_sessions:
        sessions = _read("sessions.csv", data_dir, parse_dates=["session_start_dt"])
        sessions["duration_min"] = sessions["duration"] / 60
        sessions = sessions.merge(attrs, on="user_id", how="left")
        data["sessions"] = sessions
    return data


def in_period(s, start, end):
    """Маска дат від start до end включно (дати як рядки 'YYYY-MM-DD')."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) + pd.Timedelta(days=1)
    return (s >= start) & (s < end)


def period_kpis(data, start, end):
    """Основні метрики за період. Усе рахує pandas."""
    u, p = data["users"], data["payments"]
    s = data.get("sessions")
    new = u[in_period(u["reg_dt"], start, end)]
    pay = p[in_period(p["pay_dt"], start, end)]
    k = {
        "new_users": int(new["user_id"].nunique()),
        "revenue": round(float(pay["amount_usd"].sum()), 2),
        "payers": int(pay["user_id"].nunique()),
        "payments": int(len(pay)),
    }
    k["arppu"] = round(k["revenue"] / k["payers"], 2) if k["payers"] else 0.0
    if s is not None:
        ses = s[in_period(s["session_start_dt"], start, end)]
        k["active_users"] = int(ses["user_id"].nunique())
        k["sessions"] = int(len(ses))
        k["avg_session_min"] = round(float(ses["duration_min"].mean()), 2) if len(ses) else 0.0
        k["payer_conversion_pct"] = round(100 * k["payers"] / k["active_users"], 2) if k["active_users"] else 0.0
    return k


def daily(data, metric):
    """Ряд метрики по днях (для графіків у демо)."""
    if metric == "new_users":
        u = data["users"][data["users"]["is_new"]]
        return u.groupby(u["reg_dt"].dt.date)["user_id"].nunique()
    if metric == "revenue":
        p = data["payments"]
        return p.groupby(p["pay_dt"].dt.date)["amount_usd"].sum()
    if metric == "active_users":
        s = data["sessions"]
        return s.groupby(s["session_start_dt"].dt.date)["user_id"].nunique()
    raise ValueError(metric)


def safe_ratio(a, b):
    return float(a) / float(b) if b else np.nan
