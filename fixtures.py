"""Автоматический подбор РЕАЛЬНЫХ ближайших матчей — чтобы не редактировать
matches.json руками каждый день.

Источник: football-data.org (v4) — бесплатный API с расписанием топ-лиг.
Это ДРУГОЙ сервис, чем football-data.co.uk в stats.py (там только архивные
CSV с результатами прошлых сезонов, будущих матчей там нет).

Нужен бесплатный ключ (2 минуты, без карты):
  1. https://www.football-data.org/client/register
  2. FOOTBALL_DATA_API_KEY=<ключ из письма> в переменные окружения.

Без ключа --auto просто упадёт с понятной ошибкой — сгенерировать
несуществующие матчи он не может, а без реального ключа фикстур не достать.
"""

import datetime
import logging
import os

import requests

log = logging.getLogger("fixtures")

API = "https://api.football-data.org/v4"

# Коды соревнований football-data.org, которые стоит освещать по умолчанию —
# топ-5 лиг + основные еврокубки. Переопределяется через FIXTURES_COMPETITIONS
# (например "PL,CL" — только АПЛ и Лига чемпионов).
DEFAULT_COMPETITIONS = ["PL", "PD", "SA", "BL1", "FL1", "CL", "ELC"]


def _headers():
    key = os.environ.get("FOOTBALL_DATA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY не задан. Бесплатный ключ: "
            "https://www.football-data.org/client/register — без него --auto "
            "не может узнать, какие матчи реально будут (а выдумывать их нельзя)."
        )
    return {"X-Auth-Token": key}


def fetch(days_ahead: int = 3, per_run: int = 1) -> list:
    """
    -> [{"home","away","home_flag","away_flag","competition","date"}, ...]

    Реальные SCHEDULED-матчи из окна [сегодня, +days_ahead] по всем
    настроенным лигам, отсортированные по дате — берём первые `per_run`
    (самые близкие по времени, чтобы прогноз был максимально свежим).
    """
    codes_env = os.environ.get("FIXTURES_COMPETITIONS", "").strip()
    codes = [c.strip().upper() for c in codes_env.split(",") if c.strip()] or DEFAULT_COMPETITIONS

    today = datetime.date.today()
    params = {
        "competitions": ",".join(codes),
        "dateFrom": today.isoformat(),
        "dateTo": (today + datetime.timedelta(days=days_ahead)).isoformat(),
        "status": "SCHEDULED",
    }
    r = requests.get(f"{API}/matches", headers=_headers(), params=params, timeout=30)
    if r.status_code == 429:
        raise RuntimeError("football-data.org: превышен лимит запросов (10/мин на бесплатном тарифе) — "
                            "попробуй запустить чуть позже")
    r.raise_for_status()

    matches = r.json().get("matches", [])
    candidates = []
    for m in matches:
        home, away = m.get("homeTeam") or {}, m.get("awayTeam") or {}
        if not home.get("name") or not away.get("name"):
            continue
        candidates.append({
            "home": home["name"],
            "away": away["name"],
            "home_flag": home.get("crest") or "",
            "away_flag": away.get("crest") or "",
            # имя турнира от football-data.org уже в том же виде,
            # что и ключи stats.LEAGUE_CODES ("Premier League", "Serie A", ...)
            "competition": (m.get("competition") or {}).get("name", ""),
            "date": m["utcDate"][:10],
            "_sort": m["utcDate"],
        })

    if not candidates:
        raise RuntimeError(
            f"Не нашёл ни одного запланированного матча за {days_ahead} дн. в лигах {codes} — "
            "либо пауза в календаре (межсезонье/международное окно), либо лиги не те. "
            "Увеличь FIXTURES_DAYS_AHEAD или поменяй FIXTURES_COMPETITIONS."
        )

    candidates.sort(key=lambda c: c["_sort"])
    for c in candidates:
        c.pop("_sort", None)

    picked = candidates[:max(1, per_run)]
    log.info("Подобрано %d матч(ей) из %d кандидатов: %s", len(picked), len(candidates),
             "; ".join(f'{c["home"]} vs {c["away"]} ({c["competition"]}, {c["date"]})' for c in picked))
    return picked
