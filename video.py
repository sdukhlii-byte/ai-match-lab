"""Видео «рука вписывает счёт» через fal.ai (first-frame + last-frame).

Почему два кадра, а не один: text-to-video / image-to-video модель сама
не напишет нужные цифры — она выдумает свои. Поэтому мы отдаём ей
первый кадр (пустой бланк) и последний (бланк с нашими цифрами), а модель
дорисовывает между ними руку с маркером. Итоговые цифры гарантированно
совпадают с прогнозами и подписью поста.

Провайдеры (VIDEO_PROVIDER):
  kling25 — fal-ai/kling-video/v2.5-turbo/pro/image-to-video + tail_image_url
            (по умолчанию): ~$0.70 за 10-секундный сегмент, без звука —
            в разы дешевле veo31 при похожем качестве самой руки/письма;
  veo31   — fal-ai/veo3.1/first-last-frame-to-video: самый фотореалистичный,
            8 сек, 9:16, со звуком маркера, но $0.40/сек — на SEGMENTS=2
            это ~$6.4 за ролик.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import random
import shutil
import subprocess
import time

import requests
from PIL import Image

from config import env_bool, env_float, env_int, env_str, require_env

log = logging.getLogger("video")

QUEUE = "https://queue.fal.run"

PROVIDERS = {
    "veo31": "fal-ai/veo3.1/first-last-frame-to-video",
    "kling25": "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
}

NEGATIVE = ("text changes on the poster, distorted letters, extra boxes, moving paper, "
            "camera movement, zoom, blur, extra fingers, deformed hands, watermark, "
            "hand touching multiple boxes at once, fingers resting on or pointing at boxes, "
            "flags or icons, ghost or duplicate digits, faint digits appearing in boxes before "
            "they are written, ink appearing in the wrong box, two boxes being filled in at the "
            "same time, digits bleeding or overlapping between rows, black or dark ink, ink color "
            "that does not match the marker tip, hand freezing or holding still mid-action, idle "
            "pauses with no writing happening, the marker hovering without touching the paper, "
            "dead time, slow motion, the action stopping before the video ends")


class VideoError(RuntimeError):
    pass


# ------------------------------------------------------------- требования ---

def ensure_tools() -> None:
    """Без ffmpeg склейка падает с FileNotFoundError из глубины subprocess —
    проверяем заранее и один раз, с внятным текстом."""
    missing = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if missing:
        raise VideoError(
            f"Не найдены {', '.join(missing)} — установи ffmpeg "
            "(в Docker-образе это уже сделано; локально: apt install ffmpeg / brew install ffmpeg) "
            "или запусти с --no-video.")


# ---------------------------------------------------------------- промпт ---

def build_prompt(rows: list, first_row: int, last_row: int) -> str:
    lines = []
    for i, r in enumerate(rows[first_row:last_row], start=1):
        lines.append(f'  {i}. Row "{r["label"].upper()}": write the single digit "{r["home"]}" inside '
                     f'its left score box. Only after it is fully drawn, move to the right score box '
                     f'of the SAME row and write the single digit "{r["away"]}" inside it.')
    order = "\n".join(lines)
    already = ("The rows above this one are already filled in with ink and stay completely unchanged — "
               "the hand does not touch, cross over, or hover near them.\n") if first_row else ""
    return (
        "Static top-down smartphone video, locked-off camera, absolutely no camera movement or zoom. "
        "A printed \"COINPLAY AI LAB\" prediction sheet lies flat and never moves on a wooden table.\n"
        "A person's left hand rests still the entire time, holding only the far left margin of the "
        "paper — clearly outside any printed box, flag, icon or text — and never moves, never points, "
        "and never hovers over the table.\n"
        "The right hand holds a bright yellow/gold paint marker — the ink it lays down is the exact "
        "same bright yellow color as the marker's own tip and cap, never black or dark — and writes "
        "ONLY the digits listed below, filling ONE empty box at a time, strictly in this order:\n"
        f"{already}{order}\n"
        "Hard rules: at every moment the marker tip touches at most one single box — the one currently "
        "being written — and nothing else; it never crosses, brushes, or lingers over any other box, "
        "flag, icon, or the title. No two boxes are ever filled in at the same time, and no digit "
        "appears anywhere until the marker has actually drawn it there. Each digit is a single, "
        "confident, continuous stroke with realistic pen pressure, ink appearing exactly where the pen "
        "tip touches — nothing more, nothing less, and always in that same bright yellow ink. The hand "
        "keeps moving smoothly from one box straight to the next with NO idle pause, no freezing, and no "
        "holding still in between — the writing action fills the entire clip continuously from the first "
        "frame to the last, finishing exactly when all the listed digits are done. Every other printed "
        "element (title, flags, model names, icons, boxes not yet reached) stays perfectly sharp and "
        "unchanged throughout. Soft natural daylight, realistic skin and hands, subtle marker squeak "
        "sound. At the very end both hands lift and move out of frame, leaving the completed sheet lying still."
    )


# ------------------------------------------------------------------ fal ----

def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _headers() -> dict:
    key = require_env("FAL_KEY", "Ключ fal.ai нужен для рендера видео. "
                                 "Без него запускай с --no-video.")
    return {"Authorization": f"Key {key}", "Content-Type": "application/json"}


def _get_json(url: str, timeout: int) -> dict:
    r = requests.get(url, headers=_headers(), timeout=timeout)
    if r.status_code >= 400:
        raise VideoError(f"fal {url} -> {r.status_code}: {r.text[:300]}")
    try:
        return r.json()
    except ValueError as e:
        raise VideoError(f"fal {url}: ответ не JSON: {r.text[:200]}") from e


def _run(endpoint: str, payload: dict, timeout: int | None = None) -> dict:
    timeout = timeout or env_int("FAL_TIMEOUT_SEC", 900, lo=60, hi=3600)
    r = requests.post(f"{QUEUE}/{endpoint}", headers=_headers(), json=payload, timeout=120)
    if r.status_code >= 400:
        raise VideoError(f"fal submit {endpoint} -> {r.status_code}: {r.text[:500]}")
    try:
        job = r.json()
    except ValueError as e:
        raise VideoError(f"fal submit {endpoint}: ответ не JSON: {r.text[:200]}") from e

    request_id = job.get("request_id")
    status_url = job.get("status_url")
    response_url = job.get("response_url")
    if not (status_url and response_url):
        if not request_id:
            raise VideoError(f"fal: в ответе нет ни request_id, ни ссылок: {str(job)[:300]}")
        status_url = status_url or f"{QUEUE}/{endpoint}/requests/{request_id}/status"
        response_url = response_url or f"{QUEUE}/{endpoint}/requests/{request_id}"
    log.info("fal: задача %s поставлена (%s)", request_id, endpoint)

    deadline = time.time() + timeout
    delay, last_status = 3.0, ""
    while time.time() < deadline:
        time.sleep(delay)
        delay = min(delay * 1.3, 15.0)  # мягкий backoff: не долбим статус каждые 6 сек час подряд
        try:
            s = _get_json(status_url, timeout=60)
        except (VideoError, requests.RequestException) as e:
            log.warning("fal: статус недоступен (%s) — повторю", e)
            continue
        st = (s.get("status") or "").upper()
        if st != last_status:
            log.info("fal: %s", st or "?")
            last_status = st
        if st == "COMPLETED":
            break
        if st in ("FAILED", "ERROR", "CANCELLED"):
            raise VideoError(f"fal: задача упала: {str(s)[:400]}")
    else:
        raise VideoError(f"fal: не дождался результата за {timeout} сек (request_id={request_id})")

    return _get_json(response_url, timeout=120)


def _video_url(result: dict) -> str:
    """У разных эндпоинтов fal результат лежит то в `video`, то в `videos[0]`."""
    node = result.get("video")
    if isinstance(node, dict) and node.get("url"):
        return node["url"]
    if isinstance(node, str) and node:
        return node
    for item in result.get("videos") or []:
        if isinstance(item, dict) and item.get("url"):
            return item["url"]
        if isinstance(item, str) and item:
            return item
    raise VideoError(f"fal: в ответе нет видео: {str(result)[:300]}")


def _download(url: str, out_path: str) -> None:
    tmp = f"{out_path}.part"
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
    if os.path.getsize(tmp) < 10_000:  # пустышка вместо ролика — лучше узнать сразу
        os.remove(tmp)
        raise VideoError(f"fal: скачанный файл подозрительно мал ({url})")
    os.replace(tmp, out_path)


def _payload(provider: str, first: Image.Image, last: Image.Image, prompt: str) -> dict:
    if provider == "veo31":
        return {
            "prompt": prompt,
            "first_frame_url": _data_uri(first),
            "last_frame_url": _data_uri(last),
            "duration": env_str("VEO_DURATION", "8s"),
            "aspect_ratio": "9:16",
            "resolution": env_str("VEO_RESOLUTION", "1080p"),
            "generate_audio": env_bool("VEO_AUDIO", True),
            "negative_prompt": NEGATIVE,
        }
    return {
        "prompt": prompt,
        "image_url": _data_uri(first),
        "tail_image_url": _data_uri(last),
        "duration": env_str("KLING_DURATION", "10"),
        "negative_prompt": NEGATIVE,
        # выше 0.6 — жёстче следует тексту промпта (меньше пауз/отсебятины по
        # цвету чернил и т.п.), но при слишком высоком значении движение может
        # стать более дёрганым/менее естественным
        "cfg_scale": env_float("KLING_CFG", 0.8, lo=0.0, hi=1.0),
    }


def generate_segment(first: Image.Image, last: Image.Image, prompt: str, out_path: str) -> str:
    provider = env_str("VIDEO_PROVIDER", "kling25").lower()
    if provider not in PROVIDERS:
        raise VideoError(f"VIDEO_PROVIDER={provider!r} — известны только "
                         f"{', '.join(sorted(PROVIDERS))}")
    endpoint = PROVIDERS[provider]
    payload = _payload(provider, first, last, prompt)

    attempts = env_int("FAL_ATTEMPTS", 2, lo=1, hi=5)
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = _run(endpoint, payload)
            _download(_video_url(result), out_path)
            log.info("Сегмент сохранён: %s", out_path)
            return out_path
        except (VideoError, requests.RequestException) as e:
            last_err = e
            log.warning("fal: попытка %d/%d не удалась — %s", attempt, attempts, e)
            if attempt < attempts:
                time.sleep(5 * attempt + random.uniform(0, 2))
    raise VideoError(f"Не удалось сгенерировать сегмент за {attempts} попыт(ки): {last_err}")


# ------------------------------------------------------------- ffmpeg ------

def _ff(*args) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # раньше здесь был check=True, и настоящая причина (сообщение ffmpeg)
        # просто терялась — в логе оставался только код возврата
        raise VideoError(f"ffmpeg завершился с кодом {proc.returncode}:\n"
                         f"{(proc.stderr or '').strip()[:800]}")


def _has_audio(path: str) -> bool:
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                           "-show_entries", "stream=index", "-of", "csv=p=0", path],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        log.warning("ffprobe не смог прочитать %s — считаю, что звука нет", path)
        return False
    return bool(proc.stdout.strip())


def _normalize(src: str, dst: str) -> None:
    """Один формат для склейки: 1080x1920, 30 fps, H.264 + AAC (тишина, если звука нет)."""
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,format=yuv420p"
    common = ["-c:v", "libx264", "-preset", "medium", "-crf", "19",
              "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
              "-video_track_timescale", "90000"]  # одинаковый timebase — иначе concat рассинхронит звук
    if _has_audio(src):
        _ff("-i", src, "-vf", vf, "-map", "0:v:0", "-map", "0:a:0", *common, dst)
    else:
        _ff("-i", src, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-vf", vf, "-map", "0:v:0", "-map", "1:a:0", "-shortest", *common, dst)


def assemble(segments: list, out_path: str, hold_sec: float = 2.0, music: str = "") -> str:
    """Склейка сегментов + стоп-кадр в конце, чтобы прогнозы успели прочитать."""
    ensure_tools()
    segments = [s for s in segments if s and os.path.exists(s) and os.path.getsize(s) > 0]
    if not segments:
        raise VideoError("Нечего склеивать: ни одного готового сегмента")

    work = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(work, exist_ok=True)
    temp: list[str] = []
    try:
        norm = []
        for i, s in enumerate(segments):
            n = os.path.join(work, f"_norm{i}.mp4")
            _normalize(s, n)
            norm.append(n)
        temp += norm

        if len(norm) == 1:
            joined = norm[0]
        else:
            lst = os.path.join(work, "_concat.txt")
            with open(lst, "w", encoding="utf-8") as f:
                for n in norm:
                    # в concat-листе кавычка внутри пути экранируется как '\''
                    safe = os.path.abspath(n).replace("'", r"'\''")
                    f.write(f"file '{safe}'\n")
            joined = os.path.join(work, "_joined.mp4")
            temp += [lst, joined]
            _ff("-f", "concat", "-safe", "0", "-i", lst, "-fflags", "+genpts", "-c", "copy", joined)

        hold_sec = max(0.0, hold_sec)
        if music and os.path.exists(music):
            _ff("-i", joined, "-stream_loop", "-1", "-i", music,
                "-filter_complex",
                f"[0:v]tpad=stop_mode=clone:stop_duration={hold_sec}[v];"
                f"[0:a]apad=pad_dur={hold_sec}[a0];"
                f"[1:a]volume={env_float('MUSIC_VOLUME', 0.25, lo=0.0, hi=1.0)}[a1];"
                f"[a0][a1]amix=inputs=2:duration=first:normalize=0[a]",
                "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "medium",
                "-crf", "19", "-c:a", "aac", "-b:a", "160k",
                "-movflags", "+faststart", out_path)
        else:
            _ff("-i", joined, "-vf", f"tpad=stop_mode=clone:stop_duration={hold_sec}",
                "-af", f"apad=pad_dur={hold_sec}", "-c:v", "libx264", "-preset", "medium",
                "-crf", "19", "-c:a", "aac", "-b:a", "160k",
                "-movflags", "+faststart", out_path)
    finally:
        for p in temp:
            if p == out_path:
                continue
            try:
                os.remove(p)
            except OSError:
                pass
    return out_path
