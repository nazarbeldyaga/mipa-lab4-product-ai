"""AI-асистент продуктового аналітика: pandas рахує, Gemini пояснює."""
from .changes import METRICS, explain_change
from .data import load_data, period_kpis
from .eda import auto_eda, profile_table
from .hypotheses import compute_signals, suggest_hypotheses
from .llm import GeminiLLM
from .nl2pandas import ask
from .report import show, to_markdown
from .summary import exec_summary


class Analyst:
    """Зручна обгортка для ноутбука: дані і модель задаються один раз."""

    def __init__(self, data, llm):
        self.data, self.llm = data, llm

    def eda(self, table):
        return auto_eda(self.data[table], table, self.llm)

    def hypotheses(self, n=7):
        return suggest_hypotheses(self.data, self.llm, n)

    def ask(self, question):
        return ask(question, self.data, self.llm)

    def explain(self, metric, base, target):
        return explain_change(self.data, metric, base, target, self.llm)

    def summary(self, start, end):
        return exec_summary(self.data, start, end, self.llm)


__all__ = ["Analyst", "load_data", "period_kpis", "auto_eda", "profile_table", "compute_signals",
           "suggest_hypotheses", "ask", "explain_change", "exec_summary", "GeminiLLM", "show",
           "to_markdown", "METRICS"]
