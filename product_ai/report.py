"""Перетворення результатів у Markdown для ноутбука."""
import pandas as pd

SEV = {"high": "🔴", "medium": "🟠", "low": "🟢"}
PRIO = {"high": "висока", "medium": "середня", "low": "низька"}


def _fmt(v):
    return f"{int(v):,}".replace(",", " ") if float(v).is_integer() and abs(v) >= 1000 else (str(int(v)) if float(v).is_integer() else str(v))


def _warnings(res):
    w = res.get("warnings") or []
    if not w:
        return "\n\n✅ Перевірка чисел: усі числа у відповіді є у вхідних фактах."
    return "\n\n⚠️ **Перевірка чисел / попередження:**\n" + "\n".join(f"- {x}" for x in w)


def to_markdown(res):
    k, a = res["kind"], res.get("llm") or {}
    if k == "eda":
        f = res["facts"]
        md = [f"### EDA: {f['table']} ({f.get('rows', 0)} рядків)"]
        if not a:
            return md[0] + _warnings(res)
        md.append(f"**Оцінка якості: {a['quality_score']}/10.** {a['summary']}")
        md.append("\n**Проблеми:**")
        md += [f"- {SEV.get(i['severity'], '')} `{i['column']}`: {i['problem']} ({i['evidence']})" for i in a["issues"]]
        md.append("\n**Розподіли:**")
        md += [f"- `{d['column']}`: {d['insight']}" for d in a["distributions"]]
        md.append("\n**Що робити далі:**")
        md += [f"- {s}" for s in a["next_steps"]]
    elif k == "hypotheses":
        md = ["### Гіпотези для перевірки"]
        for i, h in enumerate(a.get("hypotheses", []), 1):
            md.append(f"**{i}. {h['title']}** (пріоритет: {PRIO.get(h['priority'], h['priority'])}; "
                      f"факти: {', '.join(h['fact_ids'])})  \n{h['statement']}  \n"
                      f"*Чому важливо:* {h['why_it_matters']}  \n*Як перевірити:* {h['how_to_check']}")
    elif k == "ask":
        md = [f"### Питання: {res['question']}"]
        if a.get("explanation"):
            md.append(a["explanation"])
        if res.get("code"):
            md.append("```python\n" + res["code"] + "\n```")
        if len(res.get("attempts", [])) > 1:
            md.append(f"*Перша спроба впала ({res['attempts'][0]['error']}), модель виправила код.*")
        r = res.get("result")
        if isinstance(r, (pd.DataFrame, pd.Series)):
            md.append(r.to_frame().to_markdown() if isinstance(r, pd.Series) else r.to_markdown())
        elif r is not None:
            md.append(f"**Відповідь:** {r}")
        return "\n\n".join(md) + ("\n\n⚠️ " + "; ".join(res["warnings"]) if res["warnings"] else "")
    elif k == "change":
        f = res["facts"]
        md = [f"### Зміна метрики «{f['metric_description']}»: {_fmt(f['base_value'])} → {_fmt(f['target_value'])} "
              f"({f['change_pct']}%)", f"{f['base_period']} проти {f['target_period']}"]
        if a:
            md.append(f"**{a['headline']}**")
            md.append("\n**Головні драйвери:**")
            md += [f"- {d['dimension']} / {d['segment']}: {d['effect']}" for d in a["drivers"]]
            md.append("\n**Гіпотези причин:**")
            md += [f"- {h['cause']}. *Доказ:* {h['evidence']}. *Перевірити:* {h['check']}" for h in a["hypotheses"]]
            if a.get("caveats"):
                md.append("\n**Застереження:**")
                md += [f"- {c}" for c in a["caveats"]]
    elif k == "summary":
        f = res["facts"]
        md = [f"### Executive summary: {f['period']}", f"**{a['headline']}**", "\n**Що сталося:**"]
        md += [f"- {x}" for x in a["what_happened"]]
        md.append(f"\n**Чому це важливо:** {a['why_it_matters']}")
        md.append("\n**Наступні кроки:**")
        md += [f"- {x}" for x in a["next_steps"]]
    else:
        md = [str(res)]
    return "\n".join(md) + _warnings(res)


def show(res):
    md = to_markdown(res)
    try:
        from IPython.display import Markdown, display
        display(Markdown(md))
    except ImportError:
        print(md)
