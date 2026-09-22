"""AI Match Lab: матч -> прогнозы 5 моделей -> видео с рукой -> zip-кит -> Telegram.

Примеры:
  python generate.py --home Spain --away Argentina --home-flag es --away-flag ar \
      --competition "World Cup Final" --date 2026-07-19

  # пачка матчей из файла
  python generate.py --match-file matches.json

  # без видео (проверить бланк и промпт), или со своими цифрами без API моделей
  python generate.py ... --no-video
  python generate.py ... --scores "1-2,2-1,2-2,1-0,2-1"

Результат: out/<slug>/ с blank.jpg, filled.jpg, prompt.txt, video.mp4,
predictions.json и <slug>.zip — кит в формате автопостера
(kit.json + threads/ instagram/ x/). Если заданы TELEGRAM_BOT_TOKEN и
TELEGRAM_CHAT_ID — zip уходит в группу, откуда его забирает автопостер.
"""

import argparse
import collections
import io
import json
import logging
import os
import re
import sys
import zipfile

import requests
from PIL import Image, ImageDraw, ImageFont

import poster
import predictions
import video

log = logging.getLogger("aml")
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- флаги ----

def load_flag(spec: str, team: str) -> Image.Image:
    """
    spec: ISO-код страны ("es", "gb-eng"), путь к файлу или URL эмблемы.
    Порядок: локальный файл -> assets/flags/<код>.png -> flagcdn -> flag-icons SVG -> заглушка.
    """
    spec = (spec or "").strip()
    if spec and os.path.exists(spec):
        return Image.open(spec)
    if spec.startswith("http"):
        r = requests.get(spec, timeout=30)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))

    code = spec.lower()
    if code:
        cached = os.path.join(HERE, "assets", "flags", f"{code}.png")
        if os.path.exists(cached):
            return Image.open(cached)
        try:
            r = requests.get(f"https://flagcdn.com/w640/{code}.png", timeout=20)
            if r.ok:
                return Image.open(io.BytesIO(r.content))
        except Exception:
            pass
        try:
            import cairosvg
            r = requests.get(f"https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/{code}.svg",
                             timeout=20)
            if r.ok:
                return Image.open(io.BytesIO(cairosvg.svg2png(bytestring=r.content, output_width=800)))
        except Exception:
            pass
        log.warning("Флаг %r не найден — рисую заглушку с названием", spec)

    img = Image.new("RGB", (600, 400), (235, 238, 245))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(os.path.join(HERE, "assets", "fonts", "RussoOne-Regular.ttf"), 64)
    d.text((300, 200), team.upper()[:12], font=f, fill=poster.NAVY, anchor="mm")
    return img


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
    lines = "\n".join(f'{r["label"]} — {r["home"]}–{r["away"]}' for r in rows)
    head = f"🤖⚽ 5 AI MODELS PREDICT: {home} vs {away}"
    sub = " · ".join(x for x in (match.get("competition"), match.get("date")) if x)
    top_line = f"\n🎯 Most common score: {top} ({k}/5)" if k > 1 else ""

    long = (f"{head}\n{sub}\n\n{lines}\n\n"
            f"📊 AI consensus: {pick} ({n}/5 models){top_line}\n\n"
            f"Which AI gets it right? Drop your score 👇\n\n"
            f"🎁 Link in bio\n18+ | Analysis and entertainment only.")
    short_scores = " · ".join(f'{r["label"]} {r["home"]}–{r["away"]}' for r in rows)
    x = (f"🤖 5 AIs predict {home} vs {away}\n\n{short_scores}\n\n"
         f"Consensus: {pick} ({n}/5)\nYour score? 👇\n\nLink in bio")
    return {"threads": long, "instagram": long, "x": x}


# ----------------------------------------------------------------- кит -----

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def build_kit(out_dir: str, slug: str, match: dict, rows: list, video_path: str, cover: str) -> str:
    caps = captions(match, rows)
    kit = {
        "match_id": slug,
        "team_a": match["home"],
        "team_b": match["away"],
        "format": "ai-match-lab-video",
        "platforms": {},
    }
    zpath = os.path.join(out_dir, f"{slug}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for plat in ("threads", "instagram", "x"):
            z.writestr(f"{plat}/post.txt", caps[plat])
            spec = {"text_file": "post.txt"}
            if video_path:
                z.write(video_path, f"{plat}/video.mp4")
                spec["videos"] = ["video.mp4"]
            else:
                z.write(cover, f"{plat}/01.jpg")
                spec["images"] = ["01.jpg"]
            kit["platforms"][plat] = spec
        z.writestr("kit.json", json.dumps(kit, ensure_ascii=False, indent=2))
    return zpath


def send_telegram(zpath: str, match: dict, preview: str = ""):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        log.info("TELEGRAM_BOT_TOKEN/CHAT_ID не заданы — в Telegram не отправляю")
        return
    api = f"https://api.telegram.org/bot{token}"
    title = f'AI Match Lab · {match["home"]} vs {match["away"]}'
    if preview and os.path.exists(preview) and os.path.getsize(preview) < 49 * 1024 * 1024:
        with open(preview, "rb") as f:
            requests.post(f"{api}/sendVideo", data={"chat_id": chat, "caption": title,
                                                    "supports_streaming": "true"},
                          files={"video": f}, timeout=300).raise_for_status()
    with open(zpath, "rb") as f:
        r = requests.post(f"{api}/sendDocument", data={"chat_id": chat, "caption": title},
                          files={"document": (os.path.basename(zpath), f, "application/zip")},
                          timeout=300)
    r.raise_for_status()
    log.info("Кит отправлен в Telegram: %s", os.path.basename(zpath))


# ------------------------------------------------------------ пайплайн -----

def _parse_scores(s: str) -> list:
    out = []
    for part in s.split(","):
        h, a = re.split(r"\s*[-:–]\s*", part.strip())
        out.append((int(h), int(a)))
    if len(out) != len(predictions.SLOTS):
        raise SystemExit(f"--scores: нужно {len(predictions.SLOTS)} счетов через запятую")
    return out


def _segments(n_rows: int) -> list:
    """Разбивка строк по видео-сегментам. SEGMENTS=1 — один ролик на все 5 строк."""
    k = max(1, min(int(os.environ.get("SEGMENTS", "1")), n_rows))
    bounds, start = [], 0
    for i in range(k):
        size = (n_rows - start) // (k - i)
        bounds.append((start, start + size))
        start += size
    return bounds


def run(match: dict, args) -> str:
    slug = slugify(f'{match["home"]}-vs-{match["away"]}-{match.get("date", "")}')
    out_dir = os.path.join(args.out, slug)
    os.makedirs(out_dir, exist_ok=True)
    log.info("=== %s vs %s -> %s", match["home"], match["away"], out_dir)

    # 1. прогнозы
    if match.get("scores"):
        pairs = _parse_scores(match["scores"])
        rows = [{"label": s[0], "icon": s[1], "model": "manual", "home": h, "away": a, "reason": ""}
                for s, (h, a) in zip(predictions.SLOTS, pairs)]
    else:
        rows = predictions.predict_all(match)
    with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump({"match": match, "rows": rows}, f, ensure_ascii=False, indent=2)

    # 2. кадры
    m = poster.Match(
        home=match["home"], away=match["away"],
        home_flag=load_flag(match.get("home_flag", ""), match["home"]),
        away_flag=load_flag(match.get("away_flag", ""), match["away"]),
        rows=[poster.Row(r["label"], r["home"], r["away"], r["icon"]) for r in rows],
    )
    table = os.environ.get("TABLE_IMAGE", os.path.join(HERE, "assets", "table.jpg"))
    segs = _segments(len(rows))
    keyframes = [poster.compose_frame(poster.render_paper(m, 0), table)]
    for _, end in segs:
        keyframes.append(poster.compose_frame(poster.render_paper(m, end), table))
    keyframes[0].save(os.path.join(out_dir, "blank.jpg"), quality=93)
    keyframes[-1].save(os.path.join(out_dir, "filled.jpg"), quality=93)

    prompts = [video.build_prompt(rows, a, b) for a, b in segs]
    with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(prompts))

    # 3. видео
    video_path = ""
    if not args.no_video:
        parts = []
        for i, ((a, b), prompt) in enumerate(zip(segs, prompts)):
            p = os.path.join(out_dir, f"_seg{i}.mp4")
            video.generate_segment(keyframes[i], keyframes[i + 1], prompt, p)
            parts.append(p)
        video_path = video.assemble(parts, os.path.join(out_dir, "video.mp4"),
                                    hold_sec=float(os.environ.get("HOLD_SEC", "2")),
                                    music=os.environ.get("MUSIC_FILE", ""))

    # 4. кит + Telegram
    zpath = build_kit(out_dir, slug, match, rows, video_path, os.path.join(out_dir, "filled.jpg"))
    log.info("Кит: %s", zpath)
    if not args.no_send:
        send_telegram(zpath, match, video_path)
    return zpath


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home")
    ap.add_argument("--away")
    ap.add_argument("--home-flag", default="", help="ISO-код (es), путь или URL эмблемы")
    ap.add_argument("--away-flag", default="")
    ap.add_argument("--competition", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--scores", default="", help='свои счета без API: "1-2,2-1,2-2,1-0,2-1"')
    ap.add_argument("--match-file", default="", help="JSON-список матчей с теми же полями")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--no-video", action="store_true", help="только кадры, промпт и кит с картинкой")
    ap.add_argument("--no-send", action="store_true", help="не отправлять в Telegram")
    args = ap.parse_args()

    if args.match_file:
        with open(args.match_file, encoding="utf-8") as f:
            matches = json.load(f)
    elif args.home and args.away:
        matches = [{"home": args.home, "away": args.away, "home_flag": args.home_flag,
                    "away_flag": args.away_flag, "competition": args.competition,
                    "date": args.date, "scores": args.scores}]
    else:
        ap.error("нужны --home/--away или --match-file")

    failed = 0
    for mt in matches:
        mt.setdefault("competition", "")
        mt.setdefault("date", "")
        try:
            run(mt, args)
        except Exception as e:
            failed += 1
            log.exception("Матч %s vs %s не собран: %s", mt.get("home"), mt.get("away"), e)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
