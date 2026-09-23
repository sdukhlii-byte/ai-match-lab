"""Персистентное состояние: какие матчи уже публиковались.

Зачем: `fixtures.fetch()` всегда берёт ближайший по времени матч из окна
[сейчас, +FIXTURES_DAYS_AHEAD]. Если запускать `--auto` раз в день по крону,
а матч ещё не сыгран (например, до него 3 дня), это БУДЕТ ТОТ ЖЕ САМЫЙ матч
на следующий день — без дедупликации в группу каждый день уходил бы дубль
одного и того же прогноза, пока матч наконец не пройдёт.

Файл переживает между отдельными прогонами --auto, ТОЛЬКО если лежит на
persistent volume (в Railway: Settings -> Volumes, примонтировать том на
путь, совпадающий с POSTED_STATE_FILE / его директорией, по умолчанию
/app/state/posted.json). Без тома контейнер каждый раз стартует с чистого
диска, и дедупликация не работает — тогда стоит либо примонтировать том,
либо просто знать, что редкий повтор возможен.
"""

from __future__ import annotations

import json
import logging
import os
import time

from config import env_str

log = logging.getLogger("state")

RETENTION_DAYS = 60  # не даём файлу расти вечно


def _path() -> str:
    return env_str(
        "POSTED_STATE_FILE",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "posted.json"),
    )


def _load() -> dict:
    p = _path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as e:
        log.warning("posted-state %s битый (%s) — начинаю с чистого листа", p, e)
        return {}


def already_posted(keys: set) -> set:
    """Какие из переданных ключей уже отмечены как опубликованные."""
    if not keys:
        return set()
    return set(_load()) & keys


def mark_posted(key: str) -> None:
    if not key:
        return
    p = _path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    data = _load()
    data[key] = int(time.time())
    cutoff = int(time.time()) - RETENTION_DAYS * 86400
    data = {k: v for k, v in data.items() if isinstance(v, (int, float)) and v >= cutoff}
    tmp = f"{p}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, p)
    except OSError as e:
        # состояние — best-effort: не срывать успешный прогон из-за диска
        log.warning("Не удалось сохранить posted-state %s (%s)", p, e)
