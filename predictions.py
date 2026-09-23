"""Реальные прогнозы счёта от 5 моделей через OpenRouter.

Каждую модель спрашиваем отдельно и одинаково: матч, дата, турнир ->
строгий JSON со счётом. Для моделей без встроенного поиска включается
веб-плагин OpenRouter (WEB_SEARCH=true), чтобы учитывались свежие новости,
травмы и составы. Perplexity ищет в вебе сам.

ID моделей у OpenRouter быстро устаревают, поэтому у каждого слота есть
«предпочтительный» id и префикс-фоллбэк: если id пропал из каталога,
берётся самая свежая модель вендора с этим префиксом.
"""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests

import stats

log = logging.getLogger("predict")

OPENROUTER = "https://openrouter.ai/api/v1"

# (подпись на бланке, ключ иконки, env-переменная, id по умолчанию, префикс-фоллбэк)
SLOTS = [
    ("ChatGPT", "chatgpt", "MODEL_CHATGPT", "openai/gpt-5", "openai/gpt-"),
    ("Claude", "claude", "MODEL_CLAUDE", "anthropic/claude-sonnet-4.5", "anthropic/claude-"),
    ("Gemini", "gemini", "MODEL_GEMINI", "google/gemini-2.5-pro", "google/gemini-"),
    ("Perplexity", "perplexity", "MODEL_PERPLEXITY", "perplexity/sonar-pro", "perplexity/sonar"),
    ("Grok", "grok", "MODEL_GROK", "x-ai/grok-4", "x-ai/grok-"),
]

_SKIP = ("image", "audio", "vision-preview", "embed", "tts", "batch", ":free", "-mini", "-nano", "-lite")  # "-mini", а не "mini": иначе отсеется "gemini"

PROMPT = """You are a football analyst. Predict the exact final score (after 90 minutes plus stoppage time) of this UPCOMING match — it has not been played yet:

Today's date: {today}
{home} vs {away}
Competition: {competition}
Match date: {date}
{stats_block}
Consider the stats above (if given) together with current form, injuries/suspensions,
squad news, head-to-head and home advantage. Weigh recent news over historical stats
when they conflict (e.g. a key injury, a managerial change, a title already decided).
Answer ONLY with JSON, no markdown:
{{"home_goals": <int 0-9>, "away_goals": <int 0-9>, "reason": "<one short sentence, max 20 words>"}}"""


def _headers():
    return {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "HTTP-Referer": "https://t.me/aimatchlab",
        "X-Title": "AI Match Lab",
    }


def _catalog() -> list:
    try:
        r = requests.get(f"{OPENROUTER}/models", timeout=30)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        log.warning("Каталог OpenRouter недоступен (%s) — беру id как есть", e)
        return []


def resolve_models() -> list:
    """[(label, icon, model_id)] с проверкой по каталогу OpenRouter."""
    cat = _catalog()
    ids = {m["id"] for m in cat}
    out = []
    for label, icon, env, default, prefix in SLOTS:
        want = os.environ.get(env, "").strip() or default
        if not cat or want in ids:
            out.append((label, icon, want))
            continue
        cands = [m for m in cat if m["id"].startswith(prefix)
                 and not any(s in m["id"] for s in _SKIP)]
        cands.sort(key=lambda m: m.get("created", 0), reverse=True)
        if cands:
            log.info("%s: %s нет в каталоге, беру свежую %s", label, want, cands[0]["id"])
            out.append((label, icon, cands[0]["id"]))
        else:
            log.warning("%s: не нашёл ни %s, ни модели с префиксом %s", label, want, prefix)
            out.append((label, icon, want))
    return out


def _parse(text: str):
    text = (text or "").strip()
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            h, a = int(data["home_goals"]), int(data["away_goals"])
            if 0 <= h <= 9 and 0 <= a <= 9:
                return h, a, str(data.get("reason", ""))[:160]
        except Exception:
            pass
    # JSON мог обрезаться по max_tokens ещё до закрывающей скобки —
    # вытаскиваем поля по отдельности, не дожидаясь валидного объекта.
    hm = re.search(r'"home_goals"\s*:\s*(\d)', text)
    am = re.search(r'"away_goals"\s*:\s*(\d)', text)
    if hm and am:
        return int(hm.group(1)), int(am.group(1)), ""
    # совсем без JSON: первое «2-1» / «2:1» в ответе
    m = re.search(r"\b(\d)\s*[-:–]\s*(\d)\b", text)
    if m:
        return int(m.group(1)), int(m.group(2)), ""
    raise ValueError(f"не распарсил ответ: {text[:200]!r}")


# Reasoning-модели (gpt-5, gemini-2.5-pro, grok с thinking) тратят часть
# max_tokens на внутренние рассуждения — при низком лимите на сам JSON-ответ
# ничего не остаётся (пустой content) или он обрезается на середине.
# Даём большой запас и просим минимум размышлений — нам нужен только счёт,
# не глубокий анализ, а платим за reasoning-токены так же, как за обычные.
_REASONING = {"effort": "low", "exclude": True}


def ask(model: str, match: dict, web: bool, stats_block: str) -> tuple:
    import datetime
    prompt = PROMPT.format(**match, today=datetime.date.today().isoformat(),
                           stats_block=f"\nReal stats:\n{stats_block}\n" if stats_block else "\n")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 1500,
        "reasoning": _REASONING,
    }
    if web and not model.startswith("perplexity/"):
        body["plugins"] = [{"id": "web", "max_results": 4}]
    last = None
    for attempt in range(3):
        try:
            r = requests.post(f"{OPENROUTER}/chat/completions", headers=_headers(),
                              json=body, timeout=180)
            if r.status_code >= 400:
                raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
            content = r.json()["choices"][0]["message"].get("content") or ""
            return _parse(content)
        except Exception as e:
            last = e
            log.warning("%s: попытка %d не удалась — %s", model, attempt + 1, e)
    raise RuntimeError(f"{model}: {last}")


def predict_all(match: dict) -> list:
    """
    match: {home, away, competition, date}
    -> [{"label","icon","model","home","away","reason"}] в порядке SLOTS.
    """
    web = os.environ.get("WEB_SEARCH", "true").lower() == "true"
    models = resolve_models()

    stats_block = ""
    if os.environ.get("USE_STATS", "true").lower() == "true":
        try:
            real = stats.lookup(match["home"], match["away"], match.get("competition", ""))
            stats_block = stats.format_for_prompt(real, match["home"], match["away"])
            if stats_block:
                log.info("Статистика найдена (%s vs %s):\n%s", match["home"], match["away"], stats_block)
        except Exception as e:
            log.warning("Не удалось получить статистику: %s", e)

    def one(slot):
        label, icon, model = slot
        h, a, why = ask(model, match, web, stats_block)
        log.info("%-10s %s  %d-%d  %s", label, model, h, a, why)
        return {"label": label, "icon": icon, "model": model, "home": h, "away": a, "reason": why}

    with ThreadPoolExecutor(max_workers=len(models)) as ex:
        return list(ex.map(one, models))
