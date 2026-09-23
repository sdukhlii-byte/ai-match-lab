"""Видео «рука вписывает счёт» через fal.ai (first-frame + last-frame).

Почему два кадра, а не один: text-to-video / image-to-video модель сама
не напишет нужные цифры — она выдумает свои. Поэтому мы отдаём ей
первый кадр (пустой бланк) и последний (бланк с нашими цифрами), а модель
дорисовывает между ними руку с маркером. Итоговые цифры гарантированно
совпадают с прогнозами и подписью поста.

Провайдеры (VIDEO_PROVIDER):
  veo31  — fal-ai/veo3.1/first-last-frame-to-video (по умолчанию):
           самый фотореалистичный, 8 сек, 9:16, со звуком маркера;
  kling25 — fal-ai/kling-video/v2.5-turbo/pro/image-to-video + tail_image_url:
           10 сек, дешевле, без звука.
"""

import base64
import io
import logging
import os
import subprocess
import time

import requests
from PIL import Image

log = logging.getLogger("video")

QUEUE = "https://queue.fal.run"

PROVIDERS = {
    "veo31": "fal-ai/veo3.1/first-last-frame-to-video",
    "kling25": "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
}

NEGATIVE = ("text changes on the poster, distorted letters, extra boxes, moving paper, "
            "camera movement, zoom, blur, extra fingers, deformed hands, watermark, "
            "hand touching multiple boxes at once, fingers resting on or pointing at boxes, flags or icons, "
            "ghost or duplicate digits, faint digits appearing in boxes before they are written, "
            "ink appearing in the wrong box, two boxes being filled in at the same time, "
            "digits bleeding or overlapping between rows")


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
        "The right hand holds a black marker and writes ONLY the digits listed below, filling ONE "
        "empty box at a time, strictly in this order:\n"
        f"{already}{order}\n"
        "Hard rules: at every moment the marker tip touches at most one single box — the one currently "
        "being written — and nothing else; it never crosses, brushes, or lingers over any other box, "
        "flag, icon, or the title. No two boxes are ever filled in at the same time, and no digit "
        "appears anywhere until the marker has actually drawn it there. Each digit is a single, "
        "confident, continuous stroke with realistic pen pressure, ink appearing exactly where the pen "
        "tip touches — nothing more, nothing less. Every other printed element (title, flags, model "
        "names, icons, boxes not yet reached) stays perfectly sharp and unchanged throughout. Soft "
        "natural daylight, realistic skin and hands, subtle marker squeak sound. At the very end both "
        "hands lift and move out of frame, leaving the completed sheet lying still."
    )


def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _headers():
    return {"Authorization": f"Key {os.environ['FAL_KEY']}", "Content-Type": "application/json"}


def _run(endpoint: str, payload: dict, timeout: int = 900) -> dict:
    r = requests.post(f"{QUEUE}/{endpoint}", headers=_headers(), json=payload, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"fal submit {endpoint} -> {r.status_code}: {r.text[:500]}")
    job = r.json()
    status_url, response_url = job["status_url"], job["response_url"]
    log.info("fal: задача %s поставлена (%s)", job.get("request_id"), endpoint)

    deadline = time.time() + timeout
    while time.time() < deadline:
        s = requests.get(status_url, headers=_headers(), timeout=60).json()
        st = s.get("status")
        if st == "COMPLETED":
            break
        if st in ("FAILED", "ERROR"):
            raise RuntimeError(f"fal: задача упала: {s}")
        time.sleep(6)
    else:
        raise RuntimeError(f"fal: не дождался результата за {timeout} сек")

    res = requests.get(response_url, headers=_headers(), timeout=120)
    if res.status_code >= 400:
        raise RuntimeError(f"fal result -> {res.status_code}: {res.text[:500]}")
    return res.json()


def generate_segment(first: Image.Image, last: Image.Image, prompt: str, out_path: str) -> str:
    provider = os.environ.get("VIDEO_PROVIDER", "veo31").lower()
    endpoint = PROVIDERS[provider]
    if provider == "veo31":
        payload = {
            "prompt": prompt,
            "first_frame_url": _data_uri(first),
            "last_frame_url": _data_uri(last),
            "duration": os.environ.get("VEO_DURATION", "8s"),
            "aspect_ratio": "9:16",
            "resolution": os.environ.get("VEO_RESOLUTION", "1080p"),
            "generate_audio": os.environ.get("VEO_AUDIO", "true").lower() == "true",
            "negative_prompt": NEGATIVE,
        }
    else:
        payload = {
            "prompt": prompt,
            "image_url": _data_uri(first),
            "tail_image_url": _data_uri(last),
            "duration": os.environ.get("KLING_DURATION", "10"),
            "negative_prompt": NEGATIVE,
            "cfg_scale": 0.6,
        }
    result = _run(endpoint, payload)
    url = (result.get("video") or {}).get("url")
    if not url:
        raise RuntimeError(f"fal: в ответе нет видео: {str(result)[:300]}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    log.info("Сегмент сохранён: %s", out_path)
    return out_path


# ------------------------------------------------------------- ffmpeg ------

def _ff(*args):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *args]
    subprocess.run(cmd, check=True)


def _has_audio(path: str) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                          "stream=index", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return bool(out)


def _normalize(src: str, dst: str):
    """Один формат для склейки: 1080x1920, 30 fps, H.264 + AAC (тишина, если звука нет)."""
    vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,format=yuv420p"
    if _has_audio(src):
        _ff("-i", src, "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", dst)
    else:
        _ff("-i", src, "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest",
            "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-c:a", "aac", "-b:a", "160k", dst)


def assemble(segments: list, out_path: str, hold_sec: float = 2.0, music: str = ""):
    """Склейка сегментов + стоп-кадр в конце, чтобы прогнозы успели прочитать."""
    work = os.path.dirname(out_path)
    norm = []
    for i, s in enumerate(segments):
        n = os.path.join(work, f"_norm{i}.mp4")
        _normalize(s, n)
        norm.append(n)

    lst = os.path.join(work, "_concat.txt")
    with open(lst, "w") as f:
        for n in norm:
            f.write(f"file '{os.path.abspath(n)}'\n")
    joined = os.path.join(work, "_joined.mp4")
    _ff("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined)

    af = f"apad=pad_dur={hold_sec}"
    if music and os.path.exists(music):
        _ff("-i", joined, "-stream_loop", "-1", "-i", music,
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={hold_sec}[v];"
            f"[0:a]{af}[a0];[1:a]volume=0.25[a1];[a0][a1]amix=inputs=2:duration=first[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "19", "-c:a", "aac",
            "-movflags", "+faststart", out_path)
    else:
        _ff("-i", joined, "-vf", f"tpad=stop_mode=clone:stop_duration={hold_sec}",
            "-af", af, "-c:v", "libx264", "-crf", "19", "-c:a", "aac",
            "-movflags", "+faststart", out_path)

    for p in norm + [lst, joined]:
        try:
            os.remove(p)
        except OSError:
            pass
    return out_path
