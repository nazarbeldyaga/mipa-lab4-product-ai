"""3. Питання природною мовою -> код pandas -> результат.

Код завжди повертається разом із результатом, щоб аналітик бачив логіку.
Перед запуском код перевіряється через ast: без імпортів, файлів, eval/exec і
службових атрибутів. Виконується в окремому просторі імен, де є тільки pd, np і
таблиці. Якщо код падає, моделі один раз відправляється помилка на виправлення.
"""
import ast

import numpy as np
import pandas as pd

from . import prompts

FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
    "os", "sys", "subprocess", "shutil", "pathlib", "importlib", "builtins",
}
FORBIDDEN_ATTRS = {
    "to_csv", "to_excel", "to_pickle", "to_parquet", "to_sql", "to_json", "to_hdf",
    "read_csv", "read_excel", "read_pickle", "read_parquet", "read_sql", "read_json",
    "system", "popen", "remove", "unlink", "rmdir",
}
SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k) for k in [
    "len", "min", "max", "sum", "round", "abs", "sorted", "range", "list", "dict", "set",
    "tuple", "str", "int", "float", "bool", "enumerate", "zip", "isinstance", "print", "any", "all",
]}


class UnsafeCode(Exception):
    pass


def check_code(code):
    """Кидає UnsafeCode, якщо в коді є щось небезпечне."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCode(f"синтаксична помилка: {e}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCode("імпорти заборонені")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeCode(f"заборонене ім'я {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise UnsafeCode(f"службовий атрибут {node.attr}")
            if node.attr in FORBIDDEN_ATTRS:
                raise UnsafeCode(f"заборонений метод {node.attr}")
    if not any(isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and any(
            isinstance(t, ast.Name) and t.id == "result"
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))
            for n in ast.walk(tree)):
        raise UnsafeCode("код не записує відповідь у змінну result")


def run_code(code, tables):
    check_code(code)
    env = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np}
    env.update({k: v for k, v in tables.items()})
    exec(code, env)  # код уже перевірений check_code і виконується без доступу до builtins
    return env["result"]


def describe_tables(tables, examples=2):
    """Опис таблиць для промпту: колонки, типи, приклади значень."""
    lines = []
    for name, df in tables.items():
        lines.append(f"Таблиця {name} ({len(df)} рядків):")
        for col in df.columns:
            ex = ", ".join(map(str, df[col].dropna().astype(str).unique()[:examples]))
            lines.append(f"  - {col} ({df[col].dtype}), напр.: {ex}")
    return "\n".join(lines)


TABLE_NOTES = """Примітки:
- users: один рядок на користувача; is_new=True для реєстрацій у періоді 2024-04-05..2024-05-11;
- sessions: одна сесія на рядок, duration у секундах, duration_min у хвилинах;
- payments: один платіж на рядок, amount_usd у доларах, feature_type_name = premium/vip;
- splits: розподіл користувачів у A/B-тесті (group_id 0 або 1);
- у sessions і payments вже є атрибути користувача: country_name, platform, gender, age_group."""


def ask(question, data, llm, max_rows=20):
    tables = {k: v for k, v in data.items() if isinstance(v, pd.DataFrame)}
    schema_text = describe_tables(tables) + "\n" + TABLE_NOTES
    user = f"СХЕМА:\n{schema_text}\n\nПИТАННЯ:\n{question}"
    ans = llm.structured(prompts.NL_SYSTEM, user, prompts.NL_SCHEMA, "pandas_code")
    out = {"kind": "ask", "question": question, "llm": ans, "attempts": [], "warnings": []}
    if not ans.get("can_answer") or not ans.get("code", "").strip():
        out["result"] = None
        out["warnings"].append("Модель відмовилась відповідати: " + ans.get("reason", ""))
        return out

    code = ans["code"]
    for attempt in range(2):
        try:
            res = run_code(code, tables)
            out["attempts"].append({"code": code, "error": None})
            out["code"] = code
            if isinstance(res, (pd.DataFrame, pd.Series)):
                res = res.head(max_rows)
            out["result"] = res
            return out
        except UnsafeCode as e:
            out["attempts"].append({"code": code, "error": f"Небезпечний код: {e}"})
            if attempt == 1:
                break
            err = f"Небезпечний код: {e}"
        except Exception as e:  # будь-яка помилка pandas
            out["attempts"].append({"code": code, "error": f"{type(e).__name__}: {e}"})
            if attempt == 1:
                break
            err = f"{type(e).__name__}: {e}"
        fix = llm.structured(prompts.NL_SYSTEM,
                             user + "\n\n" + prompts.NL_FIX.format(code=code, error=err),
                             prompts.NL_SCHEMA, "pandas_code")
        if not fix.get("can_answer") or not fix.get("code"):
            break
        code = fix["code"]
        out["llm_fix"] = fix
    out["code"] = code
    out["result"] = None
    out["warnings"].append("Код не вдалося виконати: " + out["attempts"][-1]["error"])
    return out
