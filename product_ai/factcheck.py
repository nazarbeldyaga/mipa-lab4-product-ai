"""Перевірка чисел у тексті від LLM.

Модель може впевнено написати число, якого немає в даних. Тому кожне число
з її відповіді я шукаю серед чисел, які сам передав їй у фактах (з допуском на
округлення). Те, що не знайшлося, повертається як можлива галюцинація.
"""
import json
import re

# дати і діапазони дат прибираю до пошуку чисел, щоб "15.04" не вважалось числом 15.04
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\.\d{1,2}(?:\.\d{2,4})?\b(?=\s*(?:-|–|по|до|\)|$|,|\s))")
_NUM_RE = re.compile(
    r"(?<![\w.,])[-+−]?\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?"   # 139 793 / 1 234,5
    r"|(?<![\w.,])[-+−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"          # 139,793.5
    r"|(?<![\w.,])[-+−]?\d+(?:[.,]\d+)?"                        # 30,1 / 4.0 / 12
)
_SCALE_RE = re.compile(r"^\s*(тис|тис\.|k|К|K|млн|m)\b", re.I)


def _to_float(tok):
    t = tok.replace(" ", "").replace(" ", "").replace("−", "-")
    if "," in t and "." in t:
        t = t.replace(",", "")
    elif t.count(",") == 1 and len(t.split(",")[1]) != 3:
        t = t.replace(",", ".")
    elif "," in t:
        t = t.replace(",", "")
    return float(t)


def extract_numbers(text):
    """Усі числа з тексту як float (з урахуванням 'тис.' і 'млн')."""
    text = _DATE_RE.sub(" ", str(text))
    out = []
    for m in _NUM_RE.finditer(text):
        try:
            v = _to_float(m.group())
        except ValueError:
            continue
        rest = text[m.end():m.end() + 6]
        sm = _SCALE_RE.match(rest)
        if sm:
            v *= 1_000_000 if sm.group(1).lower() in ("млн", "m") else 1000
        out.append(v)
    return out


def _collect(obj, acc):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, str):
        acc.extend(extract_numbers(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, acc)


def known_numbers(facts):
    acc = []
    _collect(facts, acc)
    nums = set()
    for k in acc:
        nums.update({k, abs(k), k * 100, abs(k) * 100, k / 100})
    return nums


def _matches(v, known):
    v = abs(v)
    for k in known:
        k = abs(k)
        if k == 0:
            if v == 0:
                return True
            continue
        # допуск на округлення: до одного знаку після коми або ~1.5% для великих/приблизних чисел
        if abs(v - k) <= 0.051 or abs(v - k) / k <= 0.015:
            return True
    return False


def check_text(text, facts, ignore_small=10):
    """Повертає список чисел з text, яких немає серед facts.
    Маленькі цілі (до ignore_small) і роки не перевіряю: це зазвичай
    кількість пунктів, номер тижня тощо."""
    known = known_numbers(facts)
    bad = []
    for v in extract_numbers(text):
        if v.is_integer() and (abs(v) <= ignore_small or 2000 <= v <= 2100):
            continue
        if not _matches(v, known):
            bad.append(v)
    return bad


def check_answer(answer, facts):
    """Перевіряє всі текстові поля відповіді LLM. Повертає список попереджень."""
    texts = []

    def walk(o, path):
        if isinstance(o, str):
            texts.append((path, o))
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(answer, "")
    warnings = []
    for path, t in texts:
        bad = check_text(t, facts)
        if bad:
            nums = ", ".join(f"{b:g}" for b in bad)
            warnings.append(f"{path}: числа {nums} не знайдені у фактах")
    return warnings


def facts_json(facts):
    return json.dumps(facts, ensure_ascii=False, indent=1, default=str)
