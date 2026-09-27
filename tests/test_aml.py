"""Тесты без сети и без ключей: всё внешнее подменяется заглушками.

Запуск:  python -m pytest -q     (или  python tests/test_aml.py )
"""

import datetime
import json
import os
import sys
import zipfile

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config          # noqa: E402
import fixtures        # noqa: E402
import generate        # noqa: E402
import poster          # noqa: E402
import predictions     # noqa: E402
import stats           # noqa: E402
import state           # noqa: E402
import video           # noqa: E402


# ------------------------------------------------------------------ config --

def test_env_int_survives_garbage(monkeypatch):
    monkeypatch.setenv("SEGMENTS", "два")
    assert config.env_int("SEGMENTS", 1) == 1          # раньше: ValueError и падение прогона


def test_env_float_accepts_comma(monkeypatch):
    monkeypatch.setenv("HOLD_SEC", "2,5")
    assert config.env_float("HOLD_SEC", 2.0) == 2.5


def test_env_int_clamps(monkeypatch):
    monkeypatch.setenv("N", "999")
    assert config.env_int("N", 1, lo=1, hi=10) == 10


# ------------------------------------------------------------------- stats --

def _m(date, home, away, hg, ag):
    return {"Date": date, "HomeTeam": home, "AwayTeam": away,
            "FTHG": hg, "FTAG": ag, "FTR": "H" if hg > ag else "A" if ag > hg else "D"}


SEASONS = {
    "2627": [_m("15/08/2026", "A", "B", 3, 0),
             _m("22/08/2026", "A", "C", 2, 2),
             _m("01/09/2026", "A", "D", 0, 4)],
    "2526": [_m("03/05/2026", "A", "F", 5, 0),
             _m("10/05/2026", "A", "E", 1, 1)],
}


@pytest.fixture
def league(monkeypatch):
    monkeypatch.setattr(stats, "_fetch_csv",
                        lambda lg, season: stats._parse_csv(_as_csv(SEASONS.get(season, []))))
    return "E0"


def _as_csv(rows):
    head = "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    return head + "".join(
        f'{r["Date"]},{r["HomeTeam"]},{r["AwayTeam"]},{r["FTHG"]},{r["FTAG"]},{r["FTR"]}\n'
        for r in rows)


def test_matches_sorted_newest_first(league, monkeypatch):
    monkeypatch.setattr(stats, "season_codes", lambda n=2, today=None: ["2627", "2526"])
    got = [m["when"].strftime("%d/%m/%Y") for m in stats.load_matches(league)]
    assert got == ["01/09/2026", "22/08/2026", "15/08/2026", "10/05/2026", "03/05/2026"]


def test_form_uses_most_recent_matches(league, monkeypatch):
    """Регрессия: двойной reverse в _load_matches отдавал ПЕРВЫЕ матчи сезона
    вместо последних, и последовательность шла задом наперёд."""
    monkeypatch.setattr(stats, "season_codes", lambda n=2, today=None: ["2627", "2526"])
    form = stats._form(stats.load_matches(league), "A", n=3)
    assert form["sequence"] == "LDW"       # 01/09 L, 22/08 D, 15/08 W — от свежего к старому
    assert form["played"] == 3


def test_elo_goes_forward_in_time():
    """Победа в последнем матче должна поднимать рейтинг; при обратном порядке
    расчёта знак дельты «перетекал» не туда."""
    matches = sorted(
        stats._parse_csv(_as_csv([_m("01/02/2026", "X", "Y", 1, 0),
                                  _m("01/03/2026", "X", "Y", 1, 0)])),
        key=lambda m: m["when"], reverse=True)
    elo = stats.compute_elo(matches)
    assert elo["X"] > 1500 > elo["Y"]
    assert round(elo["X"] + elo["Y"]) == 3000      # нулевая сумма


def test_league_code_is_tolerant():
    assert stats.league_code("Premier League") == "E0"
    assert stats.league_code("  premier  league ") == "E0"
    assert stats.league_code("Primera Division") == "SP1"
    assert stats.league_code("UEFA Champions League") is None


def test_team_aliases():
    assert stats._norm("Arsenal FC") == "Arsenal"
    assert stats._norm("Club Atlético de Madrid") == "Ath Madrid"   # было "Ata. Madrid" — нет в CSV
    assert stats._norm("Paris Saint-Germain FC") == "Paris SG"
    assert stats._norm("Brighton & Hove Albion FC") == "Brighton"


def test_season_codes():
    assert stats.season_codes(2, datetime.date(2026, 9, 1)) == ["2627", "2526"]
    assert stats.season_codes(2, datetime.date(2026, 3, 1)) == ["2526", "2425"]


def test_cp1252_decoding():
    assert "ö" in stats._decode("Malmö".encode("cp1252"))


# ------------------------------------------------------------- predictions --

def test_parse_picks_the_real_json_not_the_first_braces():
    """Регрессия: нежадный поиск первой пары {} цеплял пример из рассуждения."""
    text = ('Формат такой: {"home_goals": <int>, "away_goals": <int>}\n'
            'Ответ: {"home_goals": 2, "away_goals": 1, "reason": "форма"}')
    assert predictions._parse(text) == (2, 1, "форма")


def test_parse_truncated_json():
    assert predictions._parse('{"home_goals": 3, "away_goals": 0, "reason": "об') == (3, 0, "")


def test_parse_bare_score():
    assert predictions._parse("I think it ends 2-1 for the hosts") == (2, 1, "")


def test_parse_rejects_nonsense():
    with pytest.raises(predictions.ModelError):
        predictions._parse("no idea, sorry")


def test_prompt_ignores_extra_match_keys():
    """match содержит home_flag/scores/kickoff_utc — format(**match) на них падал."""
    p = predictions._build_prompt(
        {"home": "A", "away": "B", "competition": "PL", "date": "2026-10-01",
         "home_flag": "es", "scores": "", "today": "мусор"}, "")
    assert "A vs B" in p and "мусор" not in p


def test_one_dead_model_does_not_kill_the_match(monkeypatch):
    monkeypatch.setenv("MIN_MODELS", "4")
    monkeypatch.setenv("USE_STATS", "false")
    monkeypatch.setattr(predictions, "resolve_models",
                        lambda: [(lbl, ic, f"v/{ic}") for lbl, ic, *_ in predictions.SLOTS])

    def fake_ask(model, match, web, block):
        if "grok" in model:
            raise predictions.ModelError("503")
        return 2, 1, "ok"

    monkeypatch.setattr(predictions, "ask", fake_ask)
    rows = predictions.predict_all({"home": "A", "away": "B"})
    assert len(rows) == 4                        # раньше падал весь ex.map()


def test_too_few_models_is_an_error(monkeypatch):
    monkeypatch.setenv("MIN_MODELS", "5")
    monkeypatch.setenv("USE_STATS", "false")
    monkeypatch.setattr(predictions, "resolve_models",
                        lambda: [(lbl, ic, f"v/{ic}") for lbl, ic, *_ in predictions.SLOTS])
    monkeypatch.setattr(predictions, "ask",
                        lambda *a, **k: (_ for _ in ()).throw(predictions.ModelError("500")))
    with pytest.raises(predictions.ModelError):
        predictions.predict_all({"home": "A", "away": "B"})


def test_resolve_models_tolerates_null_created(monkeypatch):
    monkeypatch.delenv("MODEL_CHATGPT", raising=False)
    monkeypatch.setattr(predictions, "_catalog", lambda: [
        {"id": "openai/gpt-4o", "created": None},
        {"id": "openai/gpt-4.1", "created": 100},
        {"id": "anthropic/claude-sonnet-4.5", "created": 1},
        {"id": "google/gemini-2.5-pro", "created": 1},
        {"id": "perplexity/sonar-pro", "created": 1},
        {"id": "x-ai/grok-4", "created": 1},
    ])
    out = dict((lbl, mid) for lbl, _, mid in predictions.resolve_models())
    assert out["ChatGPT"] == "openai/gpt-4.1"    # раньше: TypeError на None < int


# ---------------------------------------------------------------- fixtures --

def _api_match(mid, status, when, home="Arsenal FC", away="Chelsea FC"):
    return {"id": mid, "status": status, "utcDate": when,
            "homeTeam": {"name": home, "crest": ""}, "awayTeam": {"name": away, "crest": ""},
            "competition": {"name": "Premier League"}}


def test_timed_matches_are_not_skipped(monkeypatch):
    """Регрессия: фильтр status=SCHEDULED прятал матчи со статусом TIMED,
    а это как раз все ближайшие игры."""
    soon = (datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(fixtures, "_request", lambda code, params: [_api_match(1, "TIMED", soon)])
    got = fixtures.fetch(days_ahead=7, per_run=1)
    assert got[0]["home"] == "Arsenal FC"


def test_finished_and_imminent_matches_filtered(monkeypatch):
    now = datetime.datetime.now(datetime.timezone.utc)
    rows = [
        _api_match(1, "FINISHED", (now - datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")),
        _api_match(2, "TIMED", (now + datetime.timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")),
        _api_match(3, "TIMED", (now + datetime.timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")),
    ]
    monkeypatch.setattr(fixtures, "_request", lambda code, params: rows)
    got = fixtures.fetch(days_ahead=7, per_run=5)
    assert len(got) == 1 and got[0]["date"] == (now + datetime.timedelta(days=3)).date().isoformat()


def test_no_fixtures_raises_dedicated_error(monkeypatch):
    monkeypatch.setattr(fixtures, "_request", lambda code, params: [])
    with pytest.raises(fixtures.NoFixturesFound):
        fixtures.fetch()


def test_fetch_queries_per_competition_endpoint(monkeypatch):
    """Регрессия: /v4/matches?competitions=PL,PD,... не существует и молча
    игнорирует этот параметр — правильный путь per-competition
    /v4/competitions/{code}/matches, по одному запросу на лигу."""
    monkeypatch.setenv("FIXTURES_COMPETITIONS", "PL,PD")
    seen_codes = []

    def fake_request(code, params):
        seen_codes.append(code)
        assert "competitions" not in params           # раньше протекал сюда
        assert set(params) == {"dateFrom", "dateTo"}
        return []

    monkeypatch.setattr(fixtures, "_request", fake_request)
    with pytest.raises(fixtures.NoFixturesFound):
        fixtures.fetch()
    assert seen_codes == ["PL", "PD"]                 # один запрос на каждую лигу


def test_one_bad_competition_does_not_kill_the_others(monkeypatch):
    when = (datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setenv("FIXTURES_COMPETITIONS", "PL,ELC")

    def fake_request(code, params):
        if code == "PL":
            raise RuntimeError("football-data.org отклонил ключ или тариф для лиги PL (403): ...")
        return [_api_match(1, "TIMED", when)]

    monkeypatch.setattr(fixtures, "_request", fake_request)
    got = fixtures.fetch()
    assert len(got) == 1                              # ELC всё равно нашёл матч


def test_duplicate_matches_deduped(monkeypatch):
    when = (datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(fixtures, "_request",
                        lambda code, params: [_api_match(7, "TIMED", when), _api_match(7, "TIMED", when)])
    assert len(fixtures.fetch(per_run=5)) == 1


# ------------------------------------------------------- posted-state dedup --

def test_already_posted_match_is_skipped(monkeypatch):
    """Регрессия: без этого ежедневный крон присылал бы один и тот же
    ближайший ещё не сыгранный матч каждый день подряд."""
    when = (datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [_api_match(1, "TIMED", when, "Arsenal FC", "Chelsea FC"),
            _api_match(2, "TIMED", when, "Real Madrid", "Barcelona")]
    monkeypatch.setattr(fixtures, "_request", lambda code, params: rows)

    first = fixtures.fetch(per_run=1)
    assert first[0]["home"] == "Arsenal FC"           # ближайший (тот же id) — берём первым
    state.mark_posted(first[0]["id"])

    second = fixtures.fetch(per_run=1)
    assert second[0]["home"] == "Real Madrid"         # первый уже отмечен — пропускаем его


def test_all_candidates_posted_raises_no_fixtures(monkeypatch):
    when = (datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(fixtures, "_request", lambda code, params: [_api_match(9, "TIMED", when)])
    state.mark_posted("9")
    with pytest.raises(fixtures.NoFixturesFound):
        fixtures.fetch()


def _api_match_comp(mid, status, when, competition, home="Team A", away="Team B"):
    return {"id": mid, "status": status, "utcDate": when,
            "homeTeam": {"name": home, "crest": ""}, "awayTeam": {"name": away, "crest": ""},
            "competition": {"name": competition}}


def test_dense_league_does_not_dominate_every_pick(monkeypatch):
    """Регрессия: «берём просто ближайший по времени матч из всех лиг»
    систематически выбирал одну и ту же лигу (например, Série A Бразилии,
    где игры почти каждый день) — не повтор матча, а перекос отбора в
    сторону лиги с более плотным календарём."""
    now = datetime.datetime.now(datetime.timezone.utc)
    soon = (now + datetime.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    later = (now + datetime.timedelta(days=2, hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [
        _api_match_comp(1, "TIMED", soon, "Brasileirão Série A", "Santos FC", "São Paulo FC"),
        _api_match_comp(2, "TIMED", later, "Premier League", "Arsenal FC", "Chelsea FC"),
    ]
    monkeypatch.setattr(fixtures, "_request", lambda code, params: rows)

    # вчера уже публиковали Série A — без ротации сегодня снова выбрали бы
    # её же (id=1 раньше по времени), хотя есть свежая альтернатива.
    state.mark_posted("0", competition="Brasileirão Série A")

    got = fixtures.fetch(per_run=1)
    assert got[0]["home"] == "Arsenal FC"


def test_rotation_never_blocks_posting_when_no_alternative(monkeypatch):
    """Если во всех остальных лигах пауза — ротация не должна ронять прогон,
    просто снова берём то, что есть."""
    when = (datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [_api_match_comp(1, "TIMED", when, "Brasileirão Série A")]
    monkeypatch.setattr(fixtures, "_request", lambda code, params: rows)
    state.mark_posted("0", competition="Brasileirão Série A")
    state.mark_posted("00", competition="Brasileirão Série A")

    got = fixtures.fetch(per_run=1)
    assert got[0]["id"] == "1"


def test_recent_competitions_reads_newest_first(monkeypatch):
    state.mark_posted("a", competition="Premier League")
    state.mark_posted("b", competition="La Liga")
    state.mark_posted("c", competition="Serie A")
    assert state.recent_competitions(2) == ["Serie A", "La Liga"]


def test_skip_posted_can_be_disabled(monkeypatch):
    when = (datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(fixtures, "_request", lambda code, params: [_api_match(5, "TIMED", when)])
    monkeypatch.setenv("FIXTURES_SKIP_POSTED", "false")
    state.mark_posted("5")
    got = fixtures.fetch()  # без дедупликации всё равно вернёт уже "опубликованный" матч
    assert got[0]["home"] == "Arsenal FC"


# ---------------------------------------------------------------- generate --

def test_slugify_non_latin_names():
    """Регрессия: кириллица давала пустой слаг, out_dir совпадал с out/,
    а архив назывался '.zip'."""
    cyr = generate.slugify("Спартак-vs-Зенит-2026-10-01")
    assert cyr and cyr != "-" and cyr == generate.slugify("Спартак-vs-Зенит-2026-10-01")
    assert cyr != generate.slugify("Динамо-vs-Зенит-2026-10-01")   # разные матчи — разные папки
    assert generate.slugify("—").startswith("match-")
    assert generate.slugify("Arsenal-vs-Chelsea-2026-10-01") == "arsenal-vs-chelsea-2026-10-01"


def test_parse_scores_errors():
    with pytest.raises(ValueError):
        generate.parse_scores("2-1,3-0", 5)
    with pytest.raises(ValueError):
        generate.parse_scores("2-1,abc,1-1,0-0,2-2", 5)
    assert generate.parse_scores("2-1, 1:0, 0–0, 3-3, 1-2", 5)[2] == (0, 0)


def test_segments_split(monkeypatch):
    # и cyber, и marker по умолчанию режут по одной строке (2 клетки) на
    # сегмент — паковать несколько строк в один клип ломало реальную
    # генерацию (цифры наплывали на гербы, подписи двоились), см. коммент
    # в _default_segments().
    monkeypatch.delenv("SEGMENTS", raising=False)
    monkeypatch.delenv("VIDEO_STYLE", raising=False)
    assert generate._segments(5) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    monkeypatch.setenv("VIDEO_STYLE", "marker")
    assert generate._segments(5) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    monkeypatch.setenv("SEGMENTS", "1")
    assert generate._segments(5) == [(0, 5)]


def test_segments_respect_prefilled_start(monkeypatch):
    """Первая строка уже заполнена на стартовом кадре, значит анимировать её
    заново не нужно — сегменты обязаны начинаться со следующей."""
    monkeypatch.setenv("SEGMENTS", "2")
    assert generate._segments(5, start=1) == [(1, 3), (3, 5)]
    # вырожденный случай: заполнено всё — анимировать нечего
    assert generate._segments(5, start=5) == []


def test_check_date_rejects_past():
    with pytest.raises(ValueError):
        generate._check_date({"home": "A", "away": "B", "date": "2020-01-01"}, allow_past=False)
    generate._check_date({"home": "A", "away": "B", "date": "2020-01-01"}, allow_past=True)


def test_captions_scale_with_row_count():
    rows = [{"label": f"M{i}", "home": 2, "away": 1} for i in range(4)]
    caps = generate.captions({"home": "A", "away": "B", "competition": "PL", "date": "2026-10-01"},
                             rows)
    assert "(4/4 models)" in caps["threads"]      # раньше было жёстко "/5"
    assert len(caps["x"]) <= 280


def test_build_kit_writes_valid_zip(tmp_path):
    cover = tmp_path / "filled.jpg"
    Image.new("RGB", (60, 100), "black").save(cover)
    rows = [{"label": "ChatGPT", "home": 2, "away": 1}]
    z = generate.build_kit(str(tmp_path), "a-vs-b",
                           {"home": "A", "away": "B", "competition": "PL", "date": "2026-10-01"},
                           rows, "", str(cover))
    with zipfile.ZipFile(z) as zf:
        assert zf.testzip() is None
        kit = json.loads(zf.read("kit.json"))
        assert set(kit["platforms"]) == {"threads", "instagram", "x"}
        assert kit["platforms"]["x"]["images"] == ["01.jpg"]
    assert not os.path.exists(str(z) + ".tmp")


def test_load_flag_falls_back_to_placeholder(monkeypatch):
    monkeypatch.setattr(generate.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no net")))
    img = generate.load_flag("zzz", "Testonia")   # раньше сеть роняла весь матч
    assert img.size == (600, 400)


# ------------------------------------------------------------------ poster --

def _match(rows=3):
    flag = Image.new("RGBA", (600, 400), (200, 30, 30, 255))
    return poster.Match(
        home="Alpha", away="Beta", home_flag=flag, away_flag=flag,
        rows=[poster.Row(f"MODEL{i}", i % 5, (i + 1) % 5, "claude") for i in range(rows)])


def test_render_paper_shapes():
    img = poster.render_paper(_match(), filled_rows=0)
    assert img.size == (poster.PAPER_W, poster.PAPER_H) and img.mode == "RGB"


def test_screen_is_phone_shaped():
    """Экран должен быть телефонным (~9:19.5), а не A4: на бумаге неоновое
    свечение цифр физически необъяснимо."""
    assert 0.40 < poster.PAPER_W / poster.PAPER_H < 0.52


def test_progress_and_verdict_light_up_only_when_ready():
    """Полоса прогресса и вердикт — главный сигнал «ИИ досчитал».

    На старте область прогресса и плашка вердикта должны отличаться от
    финального кадра, иначе ролик не показывает никакой работы.
    """
    m = _match(5)
    m.consensus = "ALPHA · 4/5"
    start = poster.render_paper(m, filled_rows=0)
    end = poster.render_paper(m, filled_rows=5)

    bar = (poster.TABLE_X0, poster.PROGRESS_Y - 60, poster.TABLE_X1, poster.PROGRESS_Y + 20)
    verdict = (poster.TABLE_X0, poster.CONSENSUS_Y0,
               poster.TABLE_X1, poster.CONSENSUS_Y0 + 132)
    assert start.crop(bar).tobytes() != end.crop(bar).tobytes()
    assert start.crop(verdict).tobytes() != end.crop(verdict).tobytes()


def test_hidden_verdict_is_unreadable_before_the_end():
    """До финала вердикт стоит на месте, но контраст почти нулевой — иначе
    ролик выдаёт ответ в первом же кадре."""
    m = _match(5)
    m.consensus = "ALPHA · 4/5"
    box = (poster.TABLE_X0 + 40, poster.CONSENSUS_Y0 + 60,
           poster.TABLE_X1 - 40, poster.CONSENSUS_Y0 + 125)
    import numpy as np
    dim = np.asarray(poster.render_paper(m, filled_rows=0).crop(box).convert("L"))
    lit = np.asarray(poster.render_paper(m, filled_rows=5).crop(box).convert("L"))
    assert dim.max() < 90          # тусклее любого читаемого текста
    assert lit.max() > 200         # а в финале — горит


def test_pretty_date_and_consensus_text():
    assert generate._pretty_date("2026-10-02") == "2 OCT 2026"
    assert generate._pretty_date("") == ""
    assert generate._pretty_date("не дата") == "не дата"   # не роняем прогон

    rows = [{"label": "A", "home": 2, "away": 1},
            {"label": "B", "home": 1, "away": 0},
            {"label": "C", "home": 0, "away": 2}]
    assert generate.consensus_text(rows, "Alpha", "Beta") == "Alpha"
    assert generate.consensus_note(rows, "Alpha", "Beta") == "2 of 3 models agree"
    assert generate.consensus_text([], "Alpha", "Beta") == ""
    assert generate.consensus_note([], "Alpha", "Beta") == ""


def test_frames_are_deterministic_across_segments():
    """Уже заполненные строки не должны перерисовываться при добавлении новых,
    иначе на стыке сегментов цифры «прыгают».

    Нижние 100 px области исключаем сознательно: неоновое свечение цифры —
    источник света, и ореол СЛЕДУЮЩЕЙ строки законно подсвечивает край
    предыдущей. Сами цифры при этом обязаны совпадать пиксель в пиксель.
    """
    m = _match(4)
    partial = poster.render_paper(m, filled_rows=2)
    full = poster.render_paper(m, filled_rows=4)
    glow_bleed = 100
    top = (0, poster.TABLE_Y0,
           poster.PAPER_W, poster.TABLE_Y0 + 2 * poster.ROW_H - glow_bleed)
    assert partial.crop(top).tobytes() == full.crop(top).tobytes()


def test_compose_frame_size_and_bad_table_image(tmp_path):
    frame = poster.compose_frame(poster.render_paper(_match(), 1), "")
    assert frame.size == (poster.FRAME_W, poster.FRAME_H)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    assert poster.compose_frame(poster.render_paper(_match(), 1), str(broken)).size == frame.size


def test_empty_rows_do_not_crash():
    m = poster.Match(home="A", away="B",
                     home_flag=Image.new("RGBA", (10, 10)), away_flag=Image.new("RGBA", (10, 10)),
                     rows=[])
    assert poster.render_paper(m, 0).size == (poster.PAPER_W, poster.PAPER_H)


# ------------------------------------------------------------------ video --

def test_prompt_stays_under_fal_length_limit(monkeypatch):
    """Регрессия: fal/Kling отклоняет prompt длиннее 2500 символов (422
    'String should have at most 2500 characters') — с длинными именами команд
    и всеми 5 строками в одном сегменте текст раньше вылезал за лимит.
    Проверяем оба стиля: у каждого свой текст и свой negative."""
    rows = [{"label": "Perplexity", "home": "2", "away": "1"} for _ in range(5)]
    for style in ("cyber", "marker"):
        monkeypatch.setenv("VIDEO_STYLE", style)
        for first, last in ((0, 5), (0, 3), (3, 5)):
            assert len(video.build_prompt(rows, first, last)) < 2500
        assert len(video.negative()) < 2500


def test_cyber_style_is_default_and_has_no_hands(monkeypatch):
    """Смысл стиля cyber — в кадре нет ни рук, ни маркера: именно они давали
    лишние пальцы, размазанные следы чернил и маркер на пол-кадра."""
    monkeypatch.delenv("VIDEO_STYLE", raising=False)
    assert video.style() == "cyber"
    rows = [{"label": "ChatGPT", "home": "2", "away": "1"}]
    prompt = video.build_prompt(rows, 0, 1).lower()
    # Маркер в киберпромпте может упоминаться только как запрет ("NO ... markers").
    assert "no hands, people, pens or markers" in prompt
    # ...но рисования как действия в нём быть не должно вообще.
    for word in ("write ", "stroke", "the tip", "hand moves"):
        assert word not in prompt
    assert "hands" in video.negative().lower()


def test_marker_style_still_available(monkeypatch):
    monkeypatch.setenv("VIDEO_STYLE", "marker")
    rows = [{"label": "ChatGPT", "home": "2", "away": "1"}]
    assert "marker" in video.build_prompt(rows, 0, 1).lower()


def test_unknown_style_falls_back_to_cyber(monkeypatch):
    monkeypatch.setenv("VIDEO_STYLE", "disco")
    assert video.style() == "cyber"


def _tiny_image():
    return Image.new("RGB", (4, 4))


def test_generate_segment_dispatches_to_openrouter(monkeypatch):
    monkeypatch.setenv("VIDEO_PROVIDER", "or-veo31lite")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []
    monkeypatch.setattr(video, "_or_submit", lambda model, prompt, first, last: calls.append(model) or "job1")
    monkeypatch.setattr(video, "_or_poll", lambda job_id: {"unsigned_urls": ["https://x/video.mp4"]})
    monkeypatch.setattr(video, "_download",
                        lambda url, out_path, headers=None: calls.append(("dl", url, out_path, headers)))
    monkeypatch.setattr(video, "_run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fal не должен вызываться")))

    video.generate_segment(_tiny_image(), _tiny_image(), "prompt", "/tmp/out.mp4")
    assert calls[0] == "google/veo-3.1-lite"
    kind, url, out_path, headers = calls[1]
    assert (kind, url, out_path) == ("dl", "https://x/video.mp4", "/tmp/out.mp4")
    # Регрессия: "unsigned_urls" от OpenRouter на деле требуют тот же Bearer,
    # что и submit/poll — без заголовка скачивание падает 401 уже ПОСЛЕ того,
    # как генерация оплачена, и run() валится, так и не пометив матч
    # опубликованным (отсюда был повтор одного и того же матча изо дня в день).
    assert headers is not None
    assert headers["Authorization"] == "Bearer test-key"


def test_generate_segment_dispatches_to_fal(monkeypatch):
    monkeypatch.setenv("VIDEO_PROVIDER", "kling25")
    monkeypatch.setattr(video, "_run", lambda endpoint, payload: {"video": {"url": "https://x/v.mp4"}})
    monkeypatch.setattr(video, "_download", lambda url, out_path, headers=None: None)
    monkeypatch.setattr(video, "_or_submit",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("openrouter не должен вызываться")))

    assert video.generate_segment(_tiny_image(), _tiny_image(), "prompt", "/tmp/out.mp4") == "/tmp/out.mp4"


def test_unknown_video_provider_lists_both_backends(monkeypatch):
    monkeypatch.setenv("VIDEO_PROVIDER", "does-not-exist")
    with pytest.raises(video.VideoError) as exc:
        video.generate_segment(_tiny_image(), _tiny_image(), "prompt", "/tmp/out.mp4")
    assert "or-veo31lite" in str(exc.value) and "kling25" in str(exc.value)


def test_or_submit_payload_shape(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"id": "job123"}

    def fake_post(url, headers, json, timeout):
        captured["url"], captured["headers"], captured["json"] = url, headers, json
        return FakeResp()

    monkeypatch.setattr(video.requests, "post", fake_post)
    job_id = video._or_submit("google/veo-3.1-lite", "a prompt", _tiny_image(), _tiny_image())

    assert job_id == "job123"
    assert captured["url"] == f"{video.OR_API}/videos"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    body = captured["json"]
    assert body["model"] == "google/veo-3.1-lite"
    assert body["aspect_ratio"] == "9:16"
    types = {f["frame_type"] for f in body["frame_images"]}
    assert types == {"first_frame", "last_frame"}


def test_assemble_applies_music_offset(monkeypatch, tmp_path):
    # MUSIC_OFFSET должен попасть в atrim ДО amix, а не быть проигнорирован —
    # иначе трек всегда звучит с начала файла и пик не подвести под вердикт.
    monkeypatch.setenv("MUSIC_OFFSET", "12")
    monkeypatch.setenv("MUSIC_VOLUME", "0.3")
    calls = []

    def fake_ff(*args):
        calls.append(args)
        # эмулируем результат ffmpeg — конечный файл должен появиться
        out = args[-1]
        if out.endswith(".mp4"):
            with open(out, "wb") as f:
                f.write(b"0")

    monkeypatch.setattr(video, "_ff", fake_ff)
    monkeypatch.setattr(video, "_has_audio", lambda p: False)
    monkeypatch.setattr(video, "ensure_tools", lambda: None)

    seg = tmp_path / "seg0.mp4"
    seg.write_bytes(b"0")
    music = tmp_path / "music.mp3"
    music.write_bytes(b"0")
    out = tmp_path / "out.mp4"

    video.assemble([str(seg)], str(out), hold_sec=2.0, music=str(music))

    mix_call = calls[-1]
    filter_complex = mix_call[mix_call.index("-filter_complex") + 1]
    assert "atrim=start=12.0" in filter_complex
    assert "volume=0.3" in filter_complex


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
