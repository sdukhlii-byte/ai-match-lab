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


def test_segments_split():
    assert generate._segments(5) == [(0, 5)]
    os.environ["SEGMENTS"] = "3"
    try:
        assert generate._segments(5) == [(0, 2), (2, 4), (4, 5)]
    finally:
        del os.environ["SEGMENTS"]


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


def test_frames_are_deterministic_across_segments():
    """Первые строки на промежуточном и финальном кадре должны совпадать
    пиксель в пиксель, иначе на стыке сегментов цифры «прыгают»."""
    m = _match(4)
    partial = poster.render_paper(m, filled_rows=2)
    full = poster.render_paper(m, filled_rows=4)
    top = (0, poster.TABLE_Y0, poster.PAPER_W, poster.TABLE_Y0 + 2 * poster.ROW_H)
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

def test_prompt_stays_under_fal_length_limit():
    """Регрессия: fal/Kling отклоняет prompt длиннее 2500 символов (422
    'String should have at most 2500 characters') — с длинными именами команд
    и всеми 5 строками в одном сегменте текст раньше вылезал за лимит."""
    rows = [{"label": "Perplexity", "home": "2", "away": "1"} for _ in range(5)]
    for first, last in ((0, 5), (0, 3), (3, 5)):
        assert len(video.build_prompt(rows, first, last)) < 2500
    assert len(video.NEGATIVE) < 2500


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
