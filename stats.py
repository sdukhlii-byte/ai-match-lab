"""Реальная статистика клубных матчей с football-data.co.uk — подмешивается
в промпт моделям, чтобы прогноз счёта опирался на цифры, а не только на
то, что модель нашла в вебе.

football-data.co.uk отдаёт бесплатные CSV по сезону и коду лиги, без
ключа API: https://www.football-data.co.uk/mmz4281/{season}/{league}.csv
Покрывает только клубные лиги (не сборные, не эсенспорт) — для матчей
сборных/турниров вроде ЧМ этот модуль просто ничего не найдёт, и промпт
уйдёт без статистики, как раньше.

Считаем:
  * Elo обеих команд (наш собственный, честно посчитанный по всей истории
    сезона — CSV коэффициентов не даёт готового рейтинга);
  * форма (W/D/L) и голы за последние 5 матчей каждой команды;
  * личные встречи (H2H) за всё время, что есть в скачанных сезонах.
"""

import csv
import io
import logging
import os
import time
from collections import defaultdict

import requests

log = logging.getLogger("stats")

BASE = "https://www.football-data.co.uk/mmz4281"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "football-data")
CACHE_TTL = 12 * 3600

# Код лиги football-data.co.uk по названию турнира (свободный текст из
# match["competition"], нижний регистр, без учёта диакритики).
LEAGUE_CODES = {
    "premier league": "E0", "epl": "E0", "english premier league": "E0",
    "championship": "E1",
    "la liga": "SP1", "laliga": "SP1", "primera division": "SP1",
    "segunda division": "SP2",
    "bundesliga": "D1",
    "2. bundesliga": "D2",
    "serie a": "I1",
    "serie b": "I2",
    "ligue 1": "F1",
    "ligue 2": "F2",
    "eredivisie": "N1",
    "primeira liga": "P1",
    "scottish premiership": "SC0",
    "super lig": "T1",
    "jupiler pro league": "B1",
}

# Псевдонимы клубов: как встречается у пользователя/в новостях -> как в CSV.
ALIASES = {
    "man city": "Man City", "manchester city": "Man City",
    "man utd": "Man United", "manchester united": "Man United", "man united": "Man United",
    "spurs": "Tottenham", "wolves": "Wolves",
    "atletico madrid": "Ata. Madrid", "atletico": "Ata. Madrid",
    "real madrid": "Real Madrid", "barcelona": "Barcelona", "barca": "Barcelona",
    "inter": "Inter", "internazionale": "Inter", "ac milan": "Milan",
    "psg": "Paris SG", "paris saint-germain": "Paris SG",
    "bayern munich": "Bayern Munich", "bayern": "Bayern Munich",
    "dortmund": "Dortmund", "borussia dortmund": "Dortmund",
}


def _norm(name: str) -> str:
    return ALIASES.get(name.strip().lower(), name.strip())


def league_code(competition: str) -> str | None:
    key = (competition or "").strip().lower()
    return LEAGUE_CODES.get(key)


def _season_codes(n_seasons: int = 2) -> list:
    """Текущий сезон + предыдущие, формат '2526' для 2025/26."""
    import datetime
    today = datetime.date.today()
    # сезон стартует летом; до июля считаем, что текущий сезон начался в прошлом году
    start_year = today.year if today.month >= 7 else today.year - 1
    return [f"{(start_year - i) % 100:02d}{(start_year - i + 1) % 100:02d}"
            for i in range(n_seasons)]


def _fetch_csv(league: str, season: str) -> list:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{league}_{season}.csv")
    if os.path.exists(cache_path) and time.time() - os.path.getmtime(cache_path) < CACHE_TTL:
        with open(cache_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        url = f"{BASE}/{season}/{league}.csv"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200 or len(r.content) < 200:
                log.info("Нет данных %s (сезон %s, код %d)", url, season, r.status_code)
                return []
            text = r.content.decode("utf-8", errors="replace")
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            log.warning("Не скачал %s: %s", url, e)
            return []

    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        if not row.get("HomeTeam") or not row.get("FTR"):
            continue
        try:
            row["FTHG"], row["FTAG"] = int(row["FTHG"]), int(row["FTAG"])
        except (ValueError, KeyError):
            continue
        rows.append(row)
    return rows


def _load_matches(league: str) -> list:
    matches = []
    for season in _season_codes():
        matches.extend(_fetch_csv(league, season))
    # даты не всегда ISO, но в пределах CSV они уже идут по возрастанию —
    # сезоны просто конкатенируем от старого к новому.
    matches.reverse()
    return list(reversed(matches))


def _compute_elo(matches: list, k: float = 24.0, home_adv: float = 60.0) -> dict:
    """Классический Elo, старт у всех с 1500, пересчёт по всей истории по порядку."""
    elo = defaultdict(lambda: 1500.0)
    for m in reversed(matches):  # matches идёт от новых к старым -> считаем от старых
        pass
    for m in matches[::-1]:
        h, a = m["HomeTeam"], m["AwayTeam"]
        rh, ra = elo[h], elo[a]
        exp_h = 1 / (1 + 10 ** (((ra) - (rh + home_adv)) / 400))
        score_h = 1.0 if m["FTHG"] > m["FTAG"] else 0.0 if m["FTHG"] < m["FTAG"] else 0.5
        elo[h] += k * (score_h - exp_h)
        elo[a] += k * ((1 - score_h) - (1 - exp_h))
    return dict(elo)


def _form(matches: list, team: str, n: int = 5) -> dict:
    """Последние n матчей команды (из уже отсортированных от новых к старым)."""
    played = []
    for m in matches:
        if m["HomeTeam"] == team:
            played.append(("H", m["FTHG"], m["FTAG"]))
        elif m["AwayTeam"] == team:
            played.append(("A", m["FTAG"], m["FTHG"]))
        if len(played) >= n:
            break
    if not played:
        return {}
    w = sum(1 for _, gf, ga in played if gf > ga)
    d = sum(1 for _, gf, ga in played if gf == ga)
    l = sum(1 for _, gf, ga in played if gf < ga)
    gf = sum(gf for _, gf, _ in played)
    ga = sum(ga for _, _, ga in played)
    seq = "".join("W" if gf > ga else "D" if gf == ga else "L" for _, gf, ga in played)
    return {"played": len(played), "w": w, "d": d, "l": l,
            "goals_for_avg": round(gf / len(played), 2),
            "goals_against_avg": round(ga / len(played), 2), "sequence": seq}


def _h2h(matches: list, home: str, away: str, n: int = 5) -> dict:
    games = [m for m in matches
             if {m["HomeTeam"], m["AwayTeam"]} == {home, away}][:n]
    if not games:
        return {}
    home_wins = sum(1 for m in games
                    if (m["HomeTeam"] == home and m["FTHG"] > m["FTAG"])
                    or (m["AwayTeam"] == home and m["FTAG"] > m["FTHG"]))
    draws = sum(1 for m in games if m["FTHG"] == m["FTAG"])
    avg_goals = round(sum(m["FTHG"] + m["FTAG"] for m in games) / len(games), 2)
    lines = [f'{m["Date"]}: {m["HomeTeam"]} {m["FTHG"]}-{m["FTAG"]} {m["AwayTeam"]}' for m in games]
    return {"played": len(games), "home_wins": home_wins, "away_wins": len(games) - home_wins - draws,
            "draws": draws, "avg_goals": avg_goals, "matches": lines}


def lookup(home: str, away: str, competition: str) -> dict:
    """
    -> {} если лига не поддерживается или данных нет, иначе:
    {"home_elo","away_elo","home_form","away_form","h2h"}
    """
    code = league_code(competition)
    if not code:
        log.info("Лига %r не в списке football-data.co.uk — без статистики", competition)
        return {}

    matches = _load_matches(code)
    if not matches:
        return {}

    h, a = _norm(home), _norm(away)
    known = {m["HomeTeam"] for m in matches} | {m["AwayTeam"] for m in matches}
    if h not in known or a not in known:
        log.info("Команда не найдена в CSV (%s / %s уже переименована?) — известные: %s",
                 h, a, sorted(known)[:6])
        return {}

    elo = _compute_elo(matches)
    return {
        "home_elo": round(elo.get(h, 1500)),
        "away_elo": round(elo.get(a, 1500)),
        "home_form": _form(matches, h),
        "away_form": _form(matches, a),
        "h2h": _h2h(matches, h, a),
    }


def format_for_prompt(stats: dict, home: str, away: str) -> str:
    """Текстовый блок для вставки в промпт модели. Пусто, если данных нет."""
    if not stats:
        return ""
    hf, af, h2h = stats["home_form"], stats["away_form"], stats["h2h"]
    lines = [f"Elo rating: {home} {stats['home_elo']} vs {away} {stats['away_elo']} "
             f"(difference {stats['home_elo'] - stats['away_elo']:+d})"]
    if hf:
        lines.append(f"{home} last {hf['played']}: {hf['sequence']} "
                     f"({hf['w']}W {hf['d']}D {hf['l']}L), "
                     f"avg {hf['goals_for_avg']} scored / {hf['goals_against_avg']} conceded")
    if af:
        lines.append(f"{away} last {af['played']}: {af['sequence']} "
                     f"({af['w']}W {af['d']}D {af['l']}L), "
                     f"avg {af['goals_for_avg']} scored / {af['goals_against_avg']} conceded")
    if h2h:
        lines.append(f"Head-to-head last {h2h['played']}: {home} won {h2h['home_wins']}, "
                     f"{away} won {h2h['away_wins']}, {h2h['draws']} draws, "
                     f"avg {h2h['avg_goals']} goals/game")
    return "\n".join(lines)
