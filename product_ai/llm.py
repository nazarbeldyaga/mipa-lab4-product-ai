"""Клієнт до Gemini API (Google AI Studio).

Відповідь моделі завжди прошу у вигляді JSON за схемою (structured output):
модель повертає готовий об'єкт, а не текст, з якого треба витягувати числа
регулярками. Ключ береться з GEMINI_API_KEY, модель можна змінити через
GEMINI_MODEL.
"""
import json
import os
import time

DEFAULT_MODEL = "gemini-flash-latest"


class GeminiLLM:
    def __init__(self, api_key=None, model=None, max_retries=4):
        from google import genai  # імпорт тут, щоб пакет можна було імпортувати і без google-genai

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Не знайдено GEMINI_API_KEY. Ключ безкоштовно створюється на "
                "https://aistudio.google.com/apikey (див. README)."
            )
        self.client = genai.Client(api_key=key)
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.max_retries = max_retries
        # для аналітичних текстів довге «обдумування» не потрібне, а відповідь з ним
        # приходить у кілька разів довше; якщо модель не підтримує рівень, вимикаю
        self.thinking_level = os.environ.get("GEMINI_THINKING", "low")
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def structured(self, system, user, schema, tool_name="answer"):
        """Повертає dict, який відповідає schema (JSON Schema)."""
        from google.genai import errors, types

        def make_config():
            kw = dict(system_instruction=system, temperature=0,
                      response_mime_type="application/json", response_json_schema=schema)
            if self.thinking_level:
                kw["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
            return types.GenerateContentConfig(**kw)

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=user, config=make_config())
                break
            except errors.APIError as e:
                if getattr(e, "code", None) == 400 and self.thinking_level and "think" in str(e).lower():
                    print("Модель не підтримує thinking_level, працюю без нього")
                    self.thinking_level = None
                    continue
                # 429 - ліміт безкоштовного тарифу, 5xx - сервер перевантажений
                if getattr(e, "code", None) in (429, 500, 503) and attempt < self.max_retries:
                    wait = 15 * (attempt + 1)
                    print(f"Gemini відповів {getattr(e, 'code', '?')}, чекаю {wait} с...")
                    time.sleep(wait)
                    continue
                raise
        self.usage["calls"] += 1
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            self.usage["input_tokens"] += meta.prompt_token_count or 0
            self.usage["output_tokens"] += meta.candidates_token_count or 0
        if getattr(resp, "parsed", None) is not None and isinstance(resp.parsed, dict):
            return resp.parsed
        return json.loads(resp.text)
