"""Рендер бланка AI Match Lab: брендовая карточка Coinplay на столе, кадр 9:16.

Два кадра на матч:
  * blank  — пустые боксы (первый кадр видео);
  * filled — боксы заполнены «от руки» (последний кадр видео).
Видео-модель интерполирует между ними — рука дописывает цифры, и в конце
на бланке гарантированно стоят именно наши прогнозы, а не то, что модель
«придумала» сама.

Если нужно несколько сегментов (см. SEGMENTS в generate.py), промежуточные
кадры рендерятся с частично заполненными строками (filled_rows=N).

Оформление — по брендбуку Coinplay: глубокий фиолетовый фон с мягким
неоновым свечением, шрифт Roboto Condensed, акцентный жёлтый, лого-знак
из двух пересекающихся «play»-кругов.
"""

from __future__ import annotations

import functools
import math
import os
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "assets", "fonts")
ICONS = os.path.join(HERE, "assets", "icons")
LOGO_MARK = os.path.join(HERE, "assets", "logo", "mark.png")  # настоящий знак Coinplay

# Кадр — как у референса (720x1280), рендерим в 1080x1920.
FRAME_W, FRAME_H = 1080, 1920
# Карточка (1:1.414, как A4) в пикселях самой карточки.
PAPER_W, PAPER_H = 1240, 1754

# --- Coinplay brand palette -------------------------------------------------
BG = (9, 0, 27)            # #09001B  фон
BG_2 = (26, 19, 56)        # #1A1338  тон фона
BG_3 = (28, 5, 73)         # #1C0549  тон фона
PANEL = (47, 11, 118)      # #2F0B76  тёмная фиолетовая панель
SECONDARY = (81, 53, 138)  # #51358A  secondary
VIOLET = (131, 93, 253)    # #835DFD  tint secondary
LILAC = (162, 133, 253)    # #A285FD  tint secondary
PRIMARY = (201, 105, 255)  # #C969FF  primary
YELLOW = (255, 225, 69)    # #FFE145  accent
WHITE = (247, 247, 250)
GREY = (198, 186, 224)     # приглушённый лиловый для второстепенного текста
INK = YELLOW               # цвет «маркера» на тёмной карточке

# Обратная совместимость со старыми именами.
NAVY = BG
NAVY_2 = PANEL
LIME = YELLOW
CYAN = VIOLET
PAPER = BG


class AssetMissing(RuntimeError):
    pass


@functools.lru_cache(maxsize=64)
def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Шрифты кэшируются: подбор кегля в _table() создавал новый FreeTypeFont
    на каждой итерации цикла, по десятку объектов на строку таблицы."""
    path = os.path.join(FONTS, name)
    if not os.path.exists(path):
        raise AssetMissing(f"Нет файла шрифта {path} — проверь папку assets/fonts")
    return ImageFont.truetype(path, size)


@dataclass
class Row:
    name: str                      # "CHATGPT"
    home: int | None = None
    away: int | None = None
    icon: str = ""                 # ключ иконки: assets/icons/<icon>.png


@dataclass
class Match:
    home: str
    away: str
    home_flag: Image.Image
    away_flag: Image.Image
    rows: list = field(default_factory=list)
    title: str = "COINPLAY AI LAB"
    subtitle: str = "5 AI MODELS PREDICT"


def _overlay(base: Image.Image, tile: Image.Image, xy: tuple[int, int]) -> None:
    """Корректное наложение RGBA поверх RGBA.

    `base.paste(tile, xy, tile)` смешивает и альфа-канал тоже, из-за чего
    у непрозрачной подложки в месте вставки альфа проседала ниже 255.
    alpha_composite делает то, что нужно.
    """
    if base.mode != "RGBA":
        base.paste(tile, xy, tile)
        return
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(tile, xy)
    base.alpha_composite(layer)


# ---------------------------------------------------------------- иконки ---

def _glyph(draw: ImageDraw.ImageDraw, key: str, cx: int, cy: int, r: int) -> None:
    """Нейтральные абстрактные значки, если своей иконки модели нет."""
    w = max(3, r // 7)
    col = WHITE
    if key == "chatgpt":        # шестигранный «узел»
        for i in range(6):
            a = math.radians(60 * i)
            b = math.radians(60 * i + 60)
            draw.line([(cx + r * .55 * math.cos(a), cy + r * .55 * math.sin(a)),
                       (cx + r * .55 * math.cos(b), cy + r * .55 * math.sin(b))],
                      fill=col, width=w)
        draw.ellipse([cx - r * .18, cy - r * .18, cx + r * .18, cy + r * .18], outline=col, width=w)
    elif key == "claude":       # лучистая звезда
        for i in range(12):
            a = math.radians(30 * i)
            draw.line([(cx + r * .15 * math.cos(a), cy + r * .15 * math.sin(a)),
                       (cx + r * .6 * math.cos(a), cy + r * .6 * math.sin(a))],
                      fill=YELLOW, width=w)
    elif key == "gemini":       # четырёхлучевая искра
        pts = []
        for i in range(8):
            a = math.radians(45 * i - 90)
            rr = r * .62 if i % 2 == 0 else r * .16
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        draw.polygon(pts, fill=LILAC)
    elif key == "perplexity":   # «решётка»
        s = r * .45
        draw.rectangle([cx - s, cy - s * .5, cx + s, cy + s * .5], outline=col, width=w)
        draw.line([(cx, cy - s * 1.2), (cx, cy + s * 1.2)], fill=col, width=w)
        draw.line([(cx - s, cy - s), (cx + s, cy + s)], fill=col, width=w)
        draw.line([(cx + s, cy - s), (cx - s, cy + s)], fill=col, width=w)
    else:                       # кольцо с диагональю
        draw.ellipse([cx - r * .45, cy - r * .45, cx + r * .45, cy + r * .45], outline=col, width=w)
        draw.line([(cx - r * .55, cy + r * .55), (cx + r * .55, cy - r * .55)], fill=col, width=w)


def _paste_icon(img: Image.Image, key: str, cx: int, cy: int, r: int) -> None:
    d = ImageDraw.Draw(img)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PANEL, outline=PRIMARY, width=4)
    path = os.path.join(ICONS, f"{key}.png") if key else ""
    if path and os.path.exists(path):
        try:
            with Image.open(path) as raw:
                ic = raw.convert("RGBA")
            side = max(1, int(r * 1.3))
            ic.thumbnail((side, side), Image.LANCZOS)
            _overlay(img, ic, (cx - ic.width // 2, cy - ic.height // 2))
            return
        except OSError:
            pass  # битый PNG — не повод ронять весь рендер, рисуем глиф
    _glyph(d, key, cx, cy, r)


# ------------------------------------------------------- фоновая подложка ---

def _vertical_gradient(w: int, h: int, top, bottom) -> Image.Image:
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    top_a = np.array(top, dtype=np.float32)
    bot_a = np.array(bottom, dtype=np.float32)
    row = top_a * (1 - t) + bot_a * t
    arr = np.repeat(row, w, axis=1)
    return Image.fromarray(arr.clip(0, 255).astype("uint8"))


def _glow_blob(canvas: Image.Image, cx: int, cy: int, r: int, color, alpha: int, blur: int) -> None:
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(color) + (alpha,))
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    canvas.alpha_composite(layer)


def _background(w: int, h: int) -> Image.Image:
    """Слоистый фон в духе брендбука: глубокий фиолетовый + неоновые «пузыри»
    мягкого света — никогда не плоский цвет."""
    base = _vertical_gradient(w, h, BG, BG_3).convert("RGBA")
    _glow_blob(base, int(w * 0.18), int(h * 0.12), int(w * 0.42), PRIMARY, 70, 140)
    _glow_blob(base, int(w * 0.86), int(h * 0.30), int(w * 0.34), VIOLET, 60, 120)
    _glow_blob(base, int(w * 0.10), int(h * 0.92), int(w * 0.36), SECONDARY, 70, 130)
    _glow_blob(base, int(w * 0.80), int(h * 0.96), int(w * 0.30), PANEL, 80, 110)
    return base


# ----------------------------------------------------------------- лист ----

def _text_c(d, xy, text, font, fill) -> None:
    d.text(xy, text, font=font, fill=fill, anchor="mm")


def _frame_border(d: ImageDraw.ImageDraw) -> None:
    m = 34  # отступ
    r = 56  # скругление углов — как в карточках/кнопках брендбука
    W, H = PAPER_W, PAPER_H
    d.rounded_rectangle([m, m, W - m, H - m], radius=r, outline=PRIMARY, width=8)
    inset = 16
    d.rounded_rectangle([m + inset, m + inset, W - m - inset, H - m - inset],
                        radius=r - 10, outline=YELLOW, width=3)
    # маленькие акцентные «чипы» по углам
    for x in (m + 70, W - m - 70 - 46):
        d.rounded_rectangle([x, m + 54, x + 46, m + 78], radius=12, fill=YELLOW)


def _logo(d: ImageDraw.ImageDraw, cx: int, cy: int, img: Image.Image | None = None) -> None:
    """Знак Coinplay. Если рядом лежит настоящий файл лого (assets/logo/mark.png,
    белый знак на прозрачном фоне) — вставляем его; иначе рисуем приблизительную
    версию (два пересекающихся «play»-круга)."""
    r = 56
    if img is not None and os.path.exists(LOGO_MARK):
        try:
            with Image.open(LOGO_MARK) as raw:
                mark = raw.convert("RGBA")
            side = r * 2
            mark.thumbnail((side, side), Image.LANCZOS)
            _overlay(img, mark, (cx - mark.width // 2, cy - mark.height // 2))
            return
        except OSError:
            pass
    d.ellipse([cx - r - 20, cy - r, cx - 20 + r, cy + r], outline=PRIMARY, width=8)
    fx = cx + 20
    d.ellipse([fx - r, cy - r, fx + r, cy + r], fill=PRIMARY)
    t = r * 0.55
    d.polygon([(fx - t * 0.45, cy - t), (fx - t * 0.45, cy + t), (fx + t * 0.85, cy)], fill=WHITE)


def _flag_card(img: Image.Image, flag: Image.Image, box) -> None:
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=22, fill=WHITE, outline=PRIMARY, width=6)
    pad = 20
    fw, fh = x1 - x0 - 2 * pad, y1 - y0 - 2 * pad
    if fw <= 0 or fh <= 0:
        return
    src = flag.convert("RGBA")
    if not src.width or not src.height:
        return
    if abs(src.width / src.height - fw / fh) > 0.25:
        # эмблема клуба, а не флаг — вписываем без растяжения на белом
        fl = Image.new("RGB", (fw, fh), WHITE)
        fitted = src.copy()
        fitted.thumbnail((max(1, fw - 20), max(1, fh - 20)), Image.LANCZOS)
        fl.paste(fitted, ((fw - fitted.width) // 2, (fh - fitted.height) // 2), fitted)
    else:
        white = Image.new("RGBA", src.size, (255, 255, 255, 255))
        fl = Image.alpha_composite(white, src).convert("RGB").resize((fw, fh), Image.LANCZOS)
    mask = Image.new("L", (fw, fh), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, fw, fh], radius=12, fill=255)
    img.paste(fl, (x0 + pad, y0 + pad), mask)


def _vs(d, cx, cy) -> None:
    r = 86
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PANEL, outline=PRIMARY, width=7)
    d.ellipse([cx - r + 12, cy - r + 12, cx + r - 12, cy + r - 12], outline=YELLOW, width=3)
    _text_c(d, (cx, cy - 4), "VS", _font("RobotoCondensed-Bold.ttf", 78), YELLOW)


# Геометрия таблицы — нужна и для рендера, и для рукописных цифр.
TABLE_X0, TABLE_X1 = 70, PAPER_W - 70
TABLE_Y0 = 880
ROW_H = 160
ICON_CELL = 170
BOX_W, BOX_H = 180, 112
HOME_BOX_CX = 760
AWAY_BOX_CX = 1040
NAME_X = TABLE_X0 + ICON_CELL + 56
NAME_MAX_W = HOME_BOX_CX - BOX_W // 2 - 24 - NAME_X


def _box(cx: int, cy: int):
    return (cx - BOX_W // 2, cy - BOX_H // 2, cx + BOX_W // 2, cy + BOX_H // 2)


def _name_font(d: ImageDraw.ImageDraw, rows: list) -> ImageFont.FreeTypeFont:
    """Один размер шрифта на все строки: самый длинный ник должен влезть до бокса."""
    size = 58
    while size > 30:
        font = _font("RobotoCondensed-SemiBold.ttf", size)
        if max(d.textlength(r.name.upper(), font=font) for r in rows) <= NAME_MAX_W:
            return font
        size -= 2
    return _font("RobotoCondensed-SemiBold.ttf", 30)


def _table(img: Image.Image, rows: list) -> None:
    if not rows:
        return
    d = ImageDraw.Draw(img)
    name_font = _name_font(d, rows)
    y = TABLE_Y0
    # карточка-панель под всей таблицей
    panel_fill = (*PANEL, 255) if img.mode == "RGBA" else PANEL
    d.rounded_rectangle([TABLE_X0, y - 18, TABLE_X1, y + ROW_H * len(rows) + 18],
                        radius=28, fill=panel_fill, outline=PRIMARY, width=5)
    mid_x = (HOME_BOX_CX + AWAY_BOX_CX) // 2
    for i, row in enumerate(rows):
        top, bot = y + i * ROW_H, y + (i + 1) * ROW_H
        cy = (top + bot) // 2
        _paste_icon(img, row.icon, TABLE_X0 + 84, cy, 54)
        d = ImageDraw.Draw(img)  # _paste_icon мог подменить содержимое img
        d.text((NAME_X, cy), row.name.upper(), font=name_font, fill=WHITE, anchor="lm")
        for bx in (HOME_BOX_CX, AWAY_BOX_CX):
            d.rounded_rectangle(_box(bx, cy), radius=18, fill=BG_2, outline=YELLOW, width=5)
        d.rectangle([mid_x - 16, cy - 4, mid_x + 16, cy + 5], fill=YELLOW)
        if i < len(rows) - 1:
            d.line([(TABLE_X0 + ICON_CELL, bot), (TABLE_X1 - 6, bot)], fill=VIOLET, width=2)
        for k in range(3):
            d.ellipse([TABLE_X1 - 26, cy - 20 + k * 18, TABLE_X1 - 20, cy - 14 + k * 18], fill=VIOLET)


def _handwrite(img: Image.Image, rows: list, filled_rows: int, seed: int) -> None:
    """Цифры «маркером»: рукописный шрифт + лёгкий разброс угла/размера/позиции.

    Разброс детерминирован (общий seed + индекс строки), поэтому кадр с 2
    заполненными строками и кадр с 5 рисуют первые две цифры ОДИНАКОВО.
    Раньше состояние Random зависело от количества уже нарисованных цифр, и
    при SEGMENTS>1 цифры между сегментами слегка «прыгали» на стыке.
    """
    for i, row in enumerate(rows[:max(0, filled_rows)]):
        cy = TABLE_Y0 + i * ROW_H + ROW_H // 2
        for j, (cx, val) in enumerate(((HOME_BOX_CX, row.home), (AWAY_BOX_CX, row.away))):
            if val is None:
                continue
            rnd = random.Random(f"{seed}:{i}:{j}")
            size = rnd.randint(112, 124)
            f = _font("Kalam-Bold.ttf", size)
            tile = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((110, 118), str(val), font=f, fill=INK + (255,), anchor="mm")
            tile = tile.rotate(rnd.uniform(-7, 5), resample=Image.BICUBIC)
            tile = tile.filter(ImageFilter.GaussianBlur(0.7))  # маркер слегка расплывается
            _overlay(img, tile, (cx - 110 + rnd.randint(-8, 8), cy - 110 + rnd.randint(-5, 5)))


def render_paper(m: Match, filled_rows: int = 0, seed: int = 7) -> Image.Image:
    img = _background(PAPER_W, PAPER_H)
    d = ImageDraw.Draw(img)
    _frame_border(d)
    _logo(d, PAPER_W // 2, 150, img)
    d = ImageDraw.Draw(img)
    _text_c(d, (PAPER_W // 2, 330), m.title, _font("RobotoCondensed-Bold.ttf", 128), WHITE)
    sub_font = _font("RobotoCondensed-SemiBold.ttf", 50)
    _text_c(d, (PAPER_W // 2, 440), m.subtitle, sub_font, PRIMARY)
    tw = d.textlength(m.subtitle, font=sub_font)
    for side in (-1, 1):
        x_in = PAPER_W // 2 + side * (tw / 2 + 24)
        x_out = PAPER_W // 2 + side * (tw / 2 + 170)
        d.line([(x_in, 440), (x_out, 440)], fill=YELLOW, width=6)
        d.line([(x_in, 454), (x_in + side * 70, 454)], fill=VIOLET, width=4)

    _flag_card(img, m.home_flag, (90, 540, 480, 800))
    _flag_card(img, m.away_flag, (PAPER_W - 480, 540, PAPER_W - 90, 800))
    _vs(ImageDraw.Draw(img), PAPER_W // 2, 670)

    _table(img, m.rows)
    _handwrite(img, m.rows, filled_rows, seed)
    return _card_texture(img.convert("RGB"))


def _card_texture(img: Image.Image) -> Image.Image:
    """Лёгкое зерно премиального картона + мягкая виньетка."""
    arr = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(11)
    arr += rng.normal(0, 2.6, arr.shape[:2])[..., None]
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 1.0 - 0.05 * (xx / w) - 0.04 * (yy / h)
    arr *= light[..., None]
    return Image.fromarray(arr.clip(0, 255).astype("uint8"))


# ---------------------------------------------------------------- стол ----

def _procedural_wood(w: int, h: int, seed: int = 3) -> Image.Image:
    """Тёмная деревянная столешница без внешних файлов (если нет TABLE_IMAGE)."""
    rng = np.random.default_rng(seed)
    y = np.arange(h)[:, None].astype(np.float32)
    x = np.arange(w)[None, :].astype(np.float32)
    grain = np.zeros((h, w), np.float32)
    for k in range(6):
        freq = rng.uniform(0.004, 0.02)
        amp = rng.uniform(4, 30)
        phase = rng.uniform(0, 6.28)
        grain += np.sin(x * freq * (k + 1) * 0.4 + np.sin(y * 0.002 * (k + 1) + phase) * amp) * (1 / (k + 1))
    span = float(grain.max() - grain.min())
    grain = (grain - grain.min()) / span if span > 1e-6 else np.zeros_like(grain)
    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    base = np.array([70, 56, 58], np.float32)
    dark = np.array([38, 28, 34], np.float32)
    t = (grain * 0.75 + 0.25 * (noise * 0.15 + 0.5)).clip(0, 1)[..., None]
    arr = base * (1 - t) + dark * t
    return Image.fromarray(arr.clip(0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1.2))


def compose_frame(paper: Image.Image, table_image: str = "", seed: int = 3) -> Image.Image:
    """Кладёт карточку на стол: небольшой поворот, мягкая тень, сверху — как на референсе."""
    bg = None
    if table_image and os.path.exists(table_image):
        try:
            with Image.open(table_image) as raw:
                src = raw.convert("RGB")
            scale = max(FRAME_W / src.width, FRAME_H / src.height)
            src = src.resize((int(src.width * scale) + 1, int(src.height * scale) + 1), Image.LANCZOS)
            left, top = (src.width - FRAME_W) // 2, (src.height - FRAME_H) // 2
            bg = src.crop((left, top, left + FRAME_W, top + FRAME_H))
        except OSError as e:
            # битый/нечитаемый TABLE_IMAGE не должен ронять прогон
            import logging
            logging.getLogger("poster").warning("Не открыл %s (%s) — рисую дерево", table_image, e)
    if bg is None:
        bg = _procedural_wood(FRAME_W, FRAME_H, seed)

    target_w = int(FRAME_W * 0.955)
    target_h = int(target_w * PAPER_H / PAPER_W)
    sheet = paper.resize((target_w, target_h), Image.LANCZOS).convert("RGBA")
    sheet = sheet.rotate(-0.6, resample=Image.BICUBIC, expand=True)

    x = (FRAME_W - sheet.width) // 2
    y = int(FRAME_H * 0.10)

    shadow = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    alpha = sheet.split()[-1].point(lambda a: 90 if a else 0)
    shadow.paste((0, 0, 0, 255), (x + 10, y + 16), alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    out = Image.alpha_composite(bg.convert("RGBA"), shadow)
    out.alpha_composite(sheet, (x, y))
    # лёгкая виньетка/неравномерный свет — ближе к фото с телефона
    vign = Image.new("L", out.size, 0)
    ImageDraw.Draw(vign).ellipse([-300, -200, FRAME_W + 300, FRAME_H + 300], fill=255)
    vign = vign.filter(ImageFilter.GaussianBlur(220))
    dark = Image.new("RGBA", out.size, (0, 0, 0, 255))
    dark.putalpha(vign.point(lambda v: int((255 - v) * 0.22)))
    return Image.alpha_composite(out, dark).convert("RGB")
