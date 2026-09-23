"""AI Match Lab: матч -> прогнозы 5 моделей -> видео с рукой -> zip-кит -> Telegram.

Примеры:
  python generate.py --home Spain --away Argentina --home-flag es --away-flag ar \
      --competition "World Cup Final" --date 2026-07-19

  # пачка матчей из файла
  python generate.py --match-file matches.json

  # без ручного списка: сам подбирает ближайшие реальные матчи (нужен
  # FOOTBALL_DATA_API_KEY, см. fixtures.py) — это и есть команда для крона
  python generate.py --auto

  # без видео (проверить бланк и промпт), или со своими цифрами без API моделей
  python generate.py ... --no-video
  python generate.py ... --scores "1-2,2-1,2-2,1-0,2-1"

Результат: out/<slug>/ с blank.jpg, filled.jpg, prompt.txt, video.mp4,
predictions.json и <slug>.zip — кит в формате автопостера
(kit.json + threads/ instagram/ x/). Если заданы TELEGRAM_BOT_TOKEN и
TELEGRAM_CHAT_ID — zip уходит в группу, откуда его забирает автопостер.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import io
import json
import logging
import os
import re
import sys
import unicodedata
import zipfile

import requests
from PIL import Image, ImageDraw, ImageFont

import fixtures
import poster
import predictions
import video
from config import env_bool, env_float, env_int, env_str

log = logging.getLogger("aml")
HERE = os.path.dirname(os.path.abspath(__file__))

TELEGRAM_VIDEO_LIMIT = 49 * 1024 * 1024  # лимит бота на sendVideo — 50 МБ
FLAG_TIMEOUT = 20


# ---------------------------------------------------------------- флаги ----

def _open_image(data: bytes, as_svg: bool = False) -> Image.Image:
    if as_svg:
        import cairosvg
        data = cairosvg.svg2png(bytestring=data, output_width=800)
    with Image.open(io.BytesIO(data)) as im:
        return im.convert("RGBA")


def _placeholder(team: str) -> Image.Image:
    img = Image.new("RGB", (600, 400), (235, 238, 245))
    d = ImageDraw.Draw(img)
    try:
        f = poster._font("RussoOne-Regular.ttf", 64)
    except poster.AssetMissing:
        f = ImageFont.load_default()
    d.text((300, 200), (team or "?").upper()[:12], font=f, fill=poster.NAVY, anchor="mm")
    return img.convert("RGBA")


def load_flag(spec: str, team: str) -> Image.Image:
    """
    spec: ISO-код страны ("es", "gb-eng"), путь к файлу или URL эмблемы.
    Порядок: локальный файл -> assets/flags/<код>.png -> flagcdn -> flag-icons SVG -> заглушка.

    Любая осечка — заглушка с названием команды, а не исключение: из-за
    недоступного CDN раньше падал весь матч целиком.
    """
    spec = (spec or "").strip()

    if spec and os.path.exists(spec):
        try:
            with open(spec, "rb") as f:
                return _open_image(f.read(), as_svg=spec.lower().endswith(".svg"))
        except Exception as e:
            log.warning("Файл флага %s не открылся (%s)", spec, e)

    if spec.lower().startswith(("http://", "https://")):
        try:
            r = requests.get(spec, timeout=FLAG_TIMEOUT)
            r.raise_for_status()
            is_svg = spec.lower().endswith(".svg") or "svg" in r.headers.get("content-type", "")
            return _open_image(r.content, as_svg=is_svg)
        except Exception as e:
            log.warning("Эмблема по ссылке %s не загрузилась (%s)", spec, e)

    code = re.sub(r"[^a-z0-9-]", "", spec.lower())
    if code:
        cached = os.path.join(HERE, "assets", "flags", f"{code}.png")
        if os.path.exists(cached):
            try:
                with open(cached, "rb") as f:
                    return _open_image(f.read())
            except Exception as e:
                log.warning("Локальный флаг %s битый (%s)", cached, e)
        for url, svg in ((f"https://flagcdn.com/w640/{code}.png", False),
                         (f"https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/{code}.svg", True)):
            try:
                r = requests.get(url, timeout=FLAG_TIMEOUT)
                if r.ok and r.content:
                    return _open_image(r.content, as_svg=svg)
            except Exception:
                continue
        log.warning("Флаг %r не найден — рисую заглушку с названием", spec)

    return _placeholder(team)


# -------------------------------------------------------------- тексты -----

def _verdict(rows, home, away):
    outcomes = collections.Counter(
        home if r["home"] > r["away"] else away if r["away"] > r["home"] else "Draw" for r in rows)
    pick, n = outcomes.most_common(1)[0]
    scores = collections.Counter(f'{r["home"]}–{r["away"]}' for r in rows)
    top_score, k = scores.most_common(1)[0]
    pick_txt = "Draw" if pick == "Draw" else f"{pick} to win"
    return pick_txt, n, top_score, k


def captions(match: dict, rows: list) -> dict:
    home, away = match["home"], match["away"]
    pick, n, top, k = _verdict(rows, home, away)
    total = len(rows)  # раньше было жёстко "/5" — при MIN_MODELS<5 подпись врала
    lines = "\n".join(f'{r["label"]} — {r["home"]}–{r["away"]}' for r in rows)
    head = f"🤖⚽ {total} AI MODELS PREDICT: {home} vs {away}"
    sub = " · ".join(x for x in (match.get("competition"), match.get("date")) if x)
    top_line = f"\n🎯 Most common score: {top} ({k}/{total})" if k > 1 else ""

    long = (f"{head}\n{sub}\n\n{lines}\n\n"
            f"📊 AI consensus: {pick} ({n}/{total} models){top_line}\n\n"
            f"Which AI gets it right? Drop your score 👇\n\n"
            f"🎁 Link in bio\n18+ | Analysis and entertainment only.")
    short_scores = " · ".join(f'{r["label"]} {r["home"]}–{r["away"]}' for r in rows)
    x = (f"🤖 {total} AIs predict {home} vs {away}\n\n{short_scores}\n\n"
         f"Consensus: {pick} ({n}/{total})\nYour score? 👇\n\nLink in bio")
    if len(x) > 280:  # X режет длинные посты — подстраховываемся коротким вариантом
        x = f"🤖 {home} vs {away}\n{short_scores}\nConsensus: {pick} ({n}/{total})\nLink in bio"[:280]
    return {"threads": long, "instagram": long, "x": x}


# ----------------------------------------------------------------- кит -----

def _letters(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def slugify(s: str) -> str:
    """ASCII-слаг.

    Для кириллицы/арабицы транслитерации нет: раньше слаг выходил пустым,
    out_dir совпадал с корневой папкой out/, а архив назывался ".zip" —
    следующий матч затирал предыдущий. Теперь потерянные при транслитерации
    имена компенсируются коротким хешем, а совсем пустой слаг заменяется целиком.
    """
    ascii_s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_s.lower()).strip("-")
    digest = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
    if not slug:
        return f"match-{digest}"
    if _letters(ascii_s) < _letters(s):
        slug = f"{slug[:88]}-{digest}".strip("-")
    return slug[:100]


def _write_json(path: str, data) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def build_kit(out_dir: str, slug: str, match: dict, rows: list,
              video_path: str, cover: str) -> str:
    caps = captions(match, rows)
    kit = {
        "match_id": slug,
        "team_a": match["home"],
        "team_b": match["away"],
        "format": "ai-match-lab-video",
        "platforms": {},
    }
    zpath = os.path.join(out_dir, f"{slug}.zip")
    tmp = f"{zpath}.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for plat in ("threads", "instagram", "x"):
            z.writestr(f"{plat}/post.txt", caps[plat])
            spec = {"text_file": "post.txt"}
            if video_path and os.path.exists(video_path):
                z.write(video_path, f"{plat}/video.mp4")
                spec["videos"] = ["video.mp4"]
            else:
                z.write(cover, f"{plat}/01.jpg")
                spec["images"] = ["01.jpg"]
            kit["platforms"][plat] = spec
        z.writestr("kit.json", json.dumps(kit, ensure_ascii=False, indent=2))
    os.replace(tmp, zpath)  # автопостер не подхватит недописанный архив
    return zpath


def send_telegram(zpath: str, match: dict, preview: str = "") -> bool:
    token, chat = env_str("TELEGRAM_BOT_TOKEN"), env_str("TELEGRAM_CHAT_ID")
    if not (token and chat):
        log.info("TELEGRAM_BOT_TOKEN/CHAT_ID не заданы — в Telegram не отправляю")
        return False
    api = f"https://api.telegram.org/bot{token}"
    title = f'Coinplay AI Lab · {match["home"]} vs {match["away"]}'

    # Превью — приятный бонус, но если оно не ушло (тайм-аут, слишком большой
    # файл), кит всё равно должен попасть в группу: раньше исключение здесь
    # обрывало отправку архива.
    if preview and os.path.exists(preview):
        size = os.path.getsize(preview)
        if size < TELEGRAM_VIDEO_LIMIT:
            try:
                with open(preview, "rb") as f:
                    r = requests.post(f"{api}/sendVideo",
                                      data={"chat_id": chat, "caption": title,
                                            "supports_streaming": "true"},
                                      files={"video": f}, timeout=300)
                r.raise_for_status()
            except Exception as e:
                log.warning("Превью в Telegram не ушло (%s) — отправляю только кит", e)
        else:
            log.info("Превью %.1f МБ больше лимита бота — отправляю только кит",
                     size / 1024 / 1024)

    with open(zpath, "rb") as f:
        r = requests.post(f"{api}/sendDocument", data={"chat_id": chat, "caption": title},
                          files={"document": (os.path.basename(zpath), f, "application/zip")},
                          timeout=300)
    r.raise_for_status()
    log.info("Кит отправлен в Telegram: %s", os.path.basename(zpath))
    return True


# ------------------------------------------------------------ пайплайн -----

def parse_scores(s: str, expected: int) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        bits = re.split(r"\s*[-:–]\s*", part)
        if len(bits) != 2 or not all(b.strip().isdigit() for b in bits):
            raise ValueError(f'--scores: не понял счёт {part!r}, нужен вид "2-1"')
        out.append((int(bits[0]), int(bits[1])))
    if len(out) != expected:
        raise ValueError(f"--scores: нужно {expected} счетов через запятую, получено {len(out)}")
    return out


def _parse_scores(s: str) -> list[tuple[int, int]]:
    """Совместимость со старым именем."""
    return parse_scores(s, len(predictions.SLOTS))


def _segments(n_rows: int) -> list[tuple[int, int]]:
    """Разбивка строк по видео-сегментам. SEGMENTS=1 — один ролик на все строки."""
    if n_rows <= 0:
        return []
    k = max(1, min(env_int("SEGMENTS", 1, lo=1), n_rows))
    base, rem = divmod(n_rows, k)
    bounds, start = [], 0
    for i in range(k):
        size = base + (1 if i < rem else 0)  # остаток отдаём первым сегментам: 5/3 -> 2,2,1
        bounds.append((start, start + size))
        start += size
    return bounds


def _check_date(match: dict, allow_past: bool) -> None:
    """
    Матч с прошедшей датой уже имеет реальный результат — модели этого не
    знают и просто нафантазируют правдоподобный "прогноз". Останавливаем до
    похода к API моделей, а не постфактум.
    """
    raw = (match.get("date") or "").strip()
    if not raw:
        return
    try:
        d = datetime.date.fromisoformat(raw)
    except ValueError:
        log.warning("%s vs %s: дату %r не разобрал (нужен формат YYYY-MM-DD) — не проверяю",
                    match["home"], match["away"], raw)
        return
    today = datetime.date.today()
    if d < today:
        msg = (f'{match["home"]} vs {match["away"]}: дата {raw} уже в прошлом '
               f'(сегодня {today.isoformat()}) — у матча есть реальный результат, '
               f'прогноз бессмыслен')
        if allow_past:
            log.warning("%s — пропускаю проверку (ALLOW_PAST_DATES=true)", msg)
        else:
            raise ValueError(msg)


def _rows_for(match: dict) -> list[dict]:
    if match.get("scores"):
        pairs = parse_scores(match["scores"], len(predictions.SLOTS))
        return [{"label": s[0], "icon": s[1], "model": "manual", "home": h, "away": a, "reason": ""}
                for s, (h, a) in zip(predictions.SLOTS, pairs)]
    return predictions.predict_all(match)


def run(match: dict, args) -> str:
    _check_date(match, allow_past=env_bool("ALLOW_PAST_DATES", False))
    slug = slugify(f'{match["home"]}-vs-{match["away"]}-{match.get("date", "")}')
    out_dir = os.path.join(args.out, slug)
    os.makedirs(out_dir, exist_ok=True)
    log.info("=== %s vs %s -> %s", match["home"], match["away"], out_dir)

    if not args.no_video:
        video.ensure_tools()  # проверяем ffmpeg ДО того, как потратим деньги на модели

    # 1. прогнозы
    rows = _rows_for(match)
    _write_json(os.path.join(out_dir, "predictions.json"), {"match": match, "rows": rows})

    # 2. кадры
    m = poster.Match(
        home=match["home"], away=match["away"],
        home_flag=load_flag(match.get("home_flag", ""), match["home"]),
        away_flag=load_flag(match.get("away_flag", ""), match["away"]),
        rows=[poster.Row(r["label"], r["home"], r["away"], r["icon"]) for r in rows],
    )
    table = env_str("TABLE_IMAGE", os.path.join(HERE, "assets", "table.jpg"))
    segs = _segments(len(rows))
    keyframes = [poster.compose_frame(poster.render_paper(m, 0), table)]
    for _, end in segs:
        keyframes.append(poster.compose_frame(poster.render_paper(m, end), table))
    blank_path = os.path.join(out_dir, "blank.jpg")
    filled_path = os.path.join(out_dir, "filled.jpg")
    keyframes[0].save(blank_path, quality=93)
    keyframes[-1].save(filled_path, quality=93)

    prompts = [video.build_prompt(rows, a, b) for a, b in segs]
    with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(prompts))

    # 3. видео
    video_path = ""
    parts: list[str] = []
    if not args.no_video:
        try:
            for i, ((a, b), prompt) in enumerate(zip(segs, prompts)):
                p = os.path.join(out_dir, f"_seg{i}.mp4")
                video.generate_segment(keyframes[i], keyframes[i + 1], prompt, p)
                parts.append(p)
            video_path = video.assemble(parts, os.path.join(out_dir, "video.mp4"),
                                        hold_sec=env_float("HOLD_SEC", 2.0, lo=0.0, hi=15.0),
                                        music=env_str("MUSIC_FILE"))
        finally:
            if not args.keep_temp:
                for p in parts:  # промежуточные сегменты раньше оставались в out/ навсегда
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    # 4. кит + Telegram
    zpath = build_kit(out_dir, slug, match, rows, video_path, filled_path)
    log.info("Кит: %s", zpath)
    if not args.no_send:
        send_telegram(zpath, match, video_path)
    return zpath


def _load_match_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise SystemExit(f"{path}: ожидался JSON-список объектов матчей")
    bad = [i for i, m in enumerate(data) if not (m.get("home") and m.get("away"))]
    if bad:
        raise SystemExit(f"{path}: в записях {bad} нет обязательных полей home/away")
    return data


def main() -> None:
    logging.basicConfig(level=env_str("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home")
    ap.add_argument("--away")
    ap.add_argument("--home-flag", default="", help="ISO-код (es), путь или URL эмблемы")
    ap.add_argument("--away-flag", default="")
    ap.add_argument("--competition", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--scores", default="", help='свои счета без API: "1-2,2-1,2-2,1-0,2-1"')
    ap.add_argument("--match-file", default="", help="JSON-список матчей с теми же полями")
    ap.add_argument("--auto", action="store_true",
                    help="не читать --match-file — самому подобрать ближайшие реальные "
                         "матчи через football-data.org (см. fixtures.py)")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--no-video", action="store_true", help="только кадры, промпт и кит с картинкой")
    ap.add_argument("--no-send", action="store_true", help="не отправлять в Telegram")
    ap.add_argument("--keep-temp", action="store_true", help="не удалять промежуточные _seg*.mp4")
    args = ap.parse_args()

    if args.auto:
        try:
            matches = fixtures.fetch(
                days_ahead=env_int("FIXTURES_DAYS_AHEAD", 10, lo=1, hi=90),
                per_run=env_int("FIXTURES_PER_RUN", 1, lo=1, hi=20),
            )
        except fixtures.NoFixturesFound as e:
            # Пауза в календаре — нормальное состояние для крон-джобы, а не сбой:
            # выходим чисто (код 0), чтобы Railway не решил, что контейнер упал,
            # и не ушёл в рестарт-луп, долбящий API по кругу.
            log.warning("%s", e)
            sys.exit(0)
        except Exception as e:
            log.error("Не удалось получить расписание: %s", e)
            sys.exit(1)
    elif args.match_file:
        matches = _load_match_file(args.match_file)
    elif args.home and args.away:
        matches = [{"home": args.home, "away": args.away, "home_flag": args.home_flag,
                    "away_flag": args.away_flag, "competition": args.competition,
                    "date": args.date, "scores": args.scores}]
    else:
        ap.error("нужны --auto, --home/--away или --match-file")

    done, failed = [], []
    for mt in matches:
        mt.setdefault("competition", "")
        mt.setdefault("date", "")
        label = f'{mt.get("home")} vs {mt.get("away")}'
        try:
            run(mt, args)
            done.append(label)
        except Exception as e:
            failed.append(label)
            log.exception("Матч %s не собран: %s", label, e)

    log.info("Итог: собрано %d, с ошибкой %d%s", len(done), len(failed),
             (" — " + ", ".join(failed)) if failed else "")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
