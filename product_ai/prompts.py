"""Усі промпти й JSON-схеми відповідей в одному місці.
Пояснення, чому вони саме такі, і як я їх змінював - у prompts.md."""

BASE_RULES = """Ти досвідчений продуктовий аналітик мобільного застосунку знайомств.
Пиши українською, коротко і конкретно, без води.
Головне правило: використовуй ТІЛЬКИ числа, які є у блоці ФАКТИ. Не рахуй нових чисел
сам (сум, різниць, відсотків), якщо їх немає у фактах. Якщо для висновку не вистачає
даних, так і напиши.
Числа пиши так само, як у фактах (можна округлити до 1 знаку після коми)."""

# ---------- 1. EDA ----------
EDA_SYSTEM = BASE_RULES + """
Тобі дають профіль таблиці, порахований pandas. Твоє завдання - оцінити якість даних
і пояснити розподіли людською мовою. Для кожної проблеми вкажи доказ: конкретне число
з профілю (кількість пропусків, викидів, рядків поза періодом тощо)."""

EDA_SCHEMA = {
    "type": "object",
    "properties": {
        "quality_score": {"type": "integer", "description": "1-10, де 10 - дані чисті"},
        "summary": {"type": "string", "description": "2-3 речення про таблицю"},
        "issues": {"type": "array", "items": {"type": "object", "properties": {
            "column": {"type": "string"},
            "problem": {"type": "string"},
            "evidence": {"type": "string", "description": "число з профілю"},
            "severity": {"type": "string", "enum": ["high", "medium", "low"]},
        }, "required": ["column", "problem", "evidence", "severity"]}},
        "distributions": {"type": "array", "items": {"type": "object", "properties": {
            "column": {"type": "string"}, "insight": {"type": "string"},
        }, "required": ["column", "insight"]}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["quality_score", "summary", "issues", "distributions", "next_steps"],
}

# ---------- 2. Гіпотези ----------
HYP_SYSTEM = BASE_RULES + """
Тобі дають список фактів про продукт, кожен має id (F01, F02...). Сформулюй гіпотези,
які аналітику варто перевірити далі. Вимоги:
- кожна гіпотеза спирається на 1-3 факти і посилається на них через fact_ids;
- у формулюванні є конкретні числа з цих фактів (наприклад, «у 3,5 раза вища», «13,2% проти 3,8%»);
- не повторюй один і той самий факт у кількох гіпотезах різними словами;
- якщо сегмент маленький (у факті є кількість користувачів), врахуй це в пріоритеті;
- how_to_check - конкретна дія: який розріз, тест або дані подивитись."""

HYP_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string", "description": "до 10 слів"},
            "statement": {"type": "string", "description": "гіпотеза з числами"},
            "why_it_matters": {"type": "string"},
            "how_to_check": {"type": "string"},
            "fact_ids": {"type": "array", "items": {"type": "string"}},
            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
        }, "required": ["title", "statement", "why_it_matters", "how_to_check", "fact_ids", "priority"]}},
    },
    "required": ["hypotheses"],
}

# ---------- 3. Питання природною мовою -> pandas ----------
NL_SYSTEM = """Ти пишеш код pandas, який відповідає на питання аналітика.
Доступні тільки змінні pd, np і таблиці, описані в СХЕМІ (вони вже завантажені як DataFrame).
Правила для коду:
- нічого не імпортуй, не читай і не записуй файли, не використовуй eval/exec/open;
- фінальну відповідь поклади у змінну result (DataFrame, Series або число);
- якщо результат таблиця, відсортуй її і обмеж до потрібної кількості рядків;
- користувачів рахуй через nunique(), а не count();
- дати фільтруй через колонки типу datetime (pd.Timestamp('2024-04-15')).
Якщо на питання не можна відповісти цими даними (немає потрібних колонок або питання
нелогічне), постав can_answer=false і поясни чому в reason, code залиш порожнім.
explanation - 1-2 речення українською, що робить код."""

NL_SCHEMA = {
    "type": "object",
    "properties": {
        "can_answer": {"type": "boolean"},
        "reason": {"type": "string"},
        "code": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["can_answer", "reason", "code", "explanation"],
}

NL_FIX = """Твій попередній код впав з помилкою. Виправ його.
ПОПЕРЕДНІЙ КОД:
{code}
ПОМИЛКА:
{error}"""

# ---------- 4. Пояснення зміни метрики ----------
CHANGE_SYSTEM = BASE_RULES + """
Тобі дають зміну метрики між двома періодами і розклад цієї зміни за сегментами
(платформа, країна, стать, вік). contribution - скільки сегмент додав у загальну зміну,
share_of_change_pct - його частка в зміні. Для метрик-відношень є дві частини внеску:
mix (змінилась частка сегмента в аудиторії) і rate (змінилось значення метрики
всередині сегмента).
Знайди 2-4 головні драйвери і сформулюй гіпотези причин. Відрізняй «сегмент став
гіршим» (rate) від «сегмента стало більше/менше» (mix). Не вигадуй зовнішніх подій як
факт, тільки як гіпотезу для перевірки."""

CHANGE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "description": "одне речення: що змінилось і на скільки"},
        "drivers": {"type": "array", "items": {"type": "object", "properties": {
            "dimension": {"type": "string"}, "segment": {"type": "string"},
            "effect": {"type": "string", "description": "внесок з числами"},
        }, "required": ["dimension", "segment", "effect"]}},
        "hypotheses": {"type": "array", "items": {"type": "object", "properties": {
            "cause": {"type": "string"}, "evidence": {"type": "string"}, "check": {"type": "string"},
        }, "required": ["cause", "evidence", "check"]}},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "drivers", "hypotheses", "caveats"],
}

# ---------- 5. Executive summary ----------
SUMMARY_SYSTEM = BASE_RULES + """
Пишеш короткий підсумок для продакт-менеджера або CEO, який не буде вчитуватись у
таблиці. Структура: заголовок з головною думкою, 3-4 пункти «що сталось» з числами,
одне-два речення «чому це важливо», 2-3 конкретні наступні кроки.
Зміни у відсотках бери з полів *_change_pct, не рахуй їх сам."""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "what_happened": {"type": "array", "items": {"type": "string"}},
        "why_it_matters": {"type": "string"},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "what_happened", "why_it_matters", "next_steps"],
}


def user_message(task, facts_text):
    return f"ЗАВДАННЯ:\n{task}\n\nФАКТИ:\n{facts_text}"
