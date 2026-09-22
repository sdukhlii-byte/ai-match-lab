"""Рендер бланка AI Match Lab: бумажный постер на столе, кадр 9:16.

Два кадра на матч:
  * blank  — пустые боксы (первый кадр видео);
  * filled — боксы заполнены «от руки» (последний кадр видео).
Видео-модель интерполирует между ними — рука дописывает цифры, и в конце
на бланке гарантированно стоят именно наши прогнозы, а не то, что модель
«придумала» сама.

Если нужно несколько сегментов (см. SEGMENTS в generate.py), промежуточные
кадры рендерятся с частично заполненными строками (filled_rows=N).
"""

import math
import os
import random
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "assets", "fonts")
ICONS = os.path.join(HERE, "assets", "icons")

# Кадр — как у референса (720x1280), рендерим в 1080x1920.
FRAME_W, FRAME_H = 1080, 1920
# Лист A4 (1:1.414) в пикселях самого листа.
PAPER_W, PAPER_H = 1240, 1754

NAVY = (20, 30, 58)
NAVY_2 = (33, 46, 82)
LIME = (178, 222, 38)
CYAN = (70, 200, 230)
INK = (22, 22, 26)          # цвет «маркера»
PAPER = (247, 247, 244)
GREY = (205, 210, 218)


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(os.path.join(FONTS, name), size)


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
    title: str = "AI MATCH LAB"
    subtitle: str = "5 AI MODELS PREDICT"


# ---------------------------------------------------------------- иконки ---

def _glyph(draw: ImageDraw.ImageDraw, key: str, cx: int, cy: int, r: int):
    """Нейтральные абстрактные значки, если своей иконки модели нет."""
    w = max(3, r // 7)
    col = (235, 240, 250)
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
                      fill=(240, 120, 70), width=w)
    elif key == "gemini":       # четырёхлучевая искра
        pts = []
        for i in range(8):
            a = math.radians(45 * i - 90)
            rr = r * .62 if i % 2 == 0 else r * .16
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        draw.polygon(pts, fill=(90, 140, 250))
    elif key == "perplexity":   # «решётка»
        s = r * .45
        draw.rectangle([cx - s, cy - s * .5, cx + s, cy + s * .5], outline=col, width=w)
        draw.line([(cx, cy - s * 1.2), (cx, cy + s * 1.2)], fill=col, width=w)
        draw.line([(cx - s, cy - s), (cx + s, cy + s)], fill=col, width=w)
        draw.line([(cx + s, cy - s), (cx - s, cy + s)], fill=col, width=w)
    else:                       # кольцо с диагональю
        draw.ellipse([cx - r * .45, cy - r * .45, cx + r * .45, cy + r * .45], outline=col, width=w)
        draw.line([(cx - r * .55, cy + r * .55), (cx + r * .55, cy - r * .55)], fill=col, width=w)


def _paste_icon(img: Image.Image, key: str, cx: int, cy: int, r: int):
    d = ImageDraw.Draw(img)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=NAVY_2, outline=CYAN, width=4)
    path = os.path.join(ICONS, f"{key}.png")
    if os.path.exists(path):
        ic = Image.open(path).convert("RGBA")
        side = int(r * 1.3)
        ic.thumbnail((side, side), Image.LANCZOS)
        img.paste(ic, (cx - ic.width // 2, cy - ic.height // 2), ic)
    else:
        _glyph(d, key, cx, cy, r)


# ----------------------------------------------------------------- лист ----

def _text_c(d, xy, text, font, fill):
    d.text(xy, text, font=font, fill=fill, anchor="mm")


def _frame_border(d: ImageDraw.ImageDraw):
    m, c = 34, 60  # отступ и срез углов
    W, H = PAPER_W, PAPER_H
    poly = [(m + c, m), (W - m - c, m), (W - m, m + c), (W - m, H - m - c),
            (W - m - c, H - m), (m + c, H - m), (m, H - m - c), (m, m + c)]
    d.polygon(poly, outline=NAVY, width=14)
    # «технологичные» насечки
    for x in (m + 120, W - m - 220):
        d.line([(x, m + 26), (x + 100, m + 26)], fill=NAVY, width=5)
    d.line([(W // 2 - 160, m), (W // 2 - 110, m + 44), (W // 2 + 110, m + 44), (W // 2 + 160, m)],
           fill=NAVY, width=10)
    for i in range(3):
        d.rectangle([W - m - 40, 260 + i * 22, W - m - 26, 268 + i * 22], fill=GREY)
        d.rectangle([m + 26, 260 + i * 22, m + 40, 268 + i * 22], fill=GREY)


def _logo(d: ImageDraw.ImageDraw, cx: int, cy: int):
    r = 50
    pts = [(cx + r * math.cos(math.radians(60 * i + 30)), cy + r * math.sin(math.radians(60 * i + 30)))
           for i in range(6)]
    d.polygon(pts, fill=NAVY, outline=LIME, width=5)
    d.ellipse([cx - 24, cy - 24, cx + 24, cy + 24], outline=(240, 240, 240), width=5)
    for a in (0, 72, 144, 216, 288):
        x, y = cx + 12 * math.cos(math.radians(a - 90)), cy + 12 * math.sin(math.radians(a - 90))
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(240, 240, 240))


def _flag_card(img, flag: Image.Image, box):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=22, fill=(255, 255, 255), outline=NAVY, width=6)
    pad = 20
    fw, fh = x1 - x0 - 2 * pad, y1 - y0 - 2 * pad
    src = flag.convert("RGBA")
    if abs(src.width / src.height - fw / fh) > 0.25:
        # эмблема клуба, а не флаг — вписываем без растяжения на белом
        fl = Image.new("RGB", (fw, fh), (255, 255, 255))
        src.thumbnail((fw - 20, fh - 20), Image.LANCZOS)
        fl.paste(src, ((fw - src.width) // 2, (fh - src.height) // 2), src)
    else:
        bg = Image.new("RGBA", src.size, (255, 255, 255, 255))
        fl = Image.alpha_composite(bg, src).convert("RGB").resize((fw, fh), Image.LANCZOS)
    mask = Image.new("L", (fw, fh), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, fw, fh], radius=12, fill=255)
    img.paste(fl, (x0 + pad, y0 + pad), mask)


def _vs(d, cx, cy):
    r = 92
    pts = [(cx + r * math.cos(math.radians(60 * i)), cy + r * math.sin(math.radians(60 * i))) for i in range(6)]
    d.polygon(pts, fill=NAVY, outline=LIME, width=7)
    _text_c(d, (cx, cy - 6), "VS", _font("RussoOne-Regular.ttf", 84), (255, 255, 255))
    for i in range(4):
        d.line([(cx - 34 + i * 20, cy + 58), (cx - 24 + i * 20, cy + 48)], fill=CYAN, width=5)


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


def _table(img, rows):
    d = ImageDraw.Draw(img)
    # один размер шрифта на все строки: самый длинный ник должен влезть до бокса
    size = 58
    while size > 30 and max(d.textlength(r.name.upper(), font=_font("ChakraPetch-Bold.ttf", size))
                            for r in rows) > NAME_MAX_W:
        size -= 2
    name_font = _font("ChakraPetch-Bold.ttf", size)
    y = TABLE_Y0
    d.rounded_rectangle([TABLE_X0, y - 18, TABLE_X1, y + ROW_H * len(rows) + 18],
                        radius=18, outline=NAVY, width=6)
    for i, row in enumerate(rows):
        top, bot = y + i * ROW_H, y + (i + 1) * ROW_H
        cy = (top + bot) // 2
        # тёмная ячейка с иконкой + «стрелка»
        d.polygon([(TABLE_X0, top), (TABLE_X0 + ICON_CELL, top), (TABLE_X0 + ICON_CELL + 34, cy),
                   (TABLE_X0 + ICON_CELL, bot), (TABLE_X0, bot)], fill=NAVY)
        d.line([(TABLE_X0 + ICON_CELL + 8, top + 8), (TABLE_X0 + ICON_CELL + 38, cy)], fill=LIME, width=4)
        _paste_icon(img, row.icon, TABLE_X0 + 84, cy, 54)
        d = ImageDraw.Draw(img)
        d.text((NAME_X, cy), row.name.upper(), font=name_font, fill=NAVY, anchor="lm")
        for bx in (HOME_BOX_CX, AWAY_BOX_CX):
            d.rounded_rectangle(_box(bx, cy), radius=14, fill=(255, 255, 255), outline=NAVY, width=5)
        d.rectangle([(HOME_BOX_CX + AWAY_BOX_CX) // 2 - 16, cy - 4,
                     (HOME_BOX_CX + AWAY_BOX_CX) // 2 + 16, cy + 5], fill=NAVY)
        if i < len(rows) - 1:
            d.line([(TABLE_X0 + ICON_CELL, bot), (TABLE_X1 - 6, bot)], fill=GREY, width=3)
        # точки-декор справа
        for k in range(3):
            d.ellipse([TABLE_X1 - 26, cy - 20 + k * 18, TABLE_X1 - 20, cy - 14 + k * 18], fill=GREY)


def _handwrite(img, rows, filled_rows: int, seed: int):
    """Цифры «маркером»: рукописный шрифт + лёгкий разброс угла/размера/позиции."""
    rnd = random.Random(seed)
    for i, row in enumerate(rows[:filled_rows]):
        cy = TABLE_Y0 + i * ROW_H + ROW_H // 2
        for cx, val in ((HOME_BOX_CX, row.home), (AWAY_BOX_CX, row.away)):
            if val is None:
                continue
            size = rnd.randint(112, 124)
            f = _font("Kalam-Bold.ttf", size)
            tile = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((110, 118), str(val), font=f, fill=INK + (255,), anchor="mm")
            tile = tile.rotate(rnd.uniform(-7, 5), resample=Image.BICUBIC)
            # маркер слегка «расплывается» по бумаге
            tile = tile.filter(ImageFilter.GaussianBlur(0.7))
            img.paste(tile, (cx - 110 + rnd.randint(-8, 8), cy - 110 + rnd.randint(-5, 5)), tile)


def render_paper(m: Match, filled_rows: int = 0, seed: int = 7) -> Image.Image:
    img = Image.new("RGB", (PAPER_W, PAPER_H), PAPER)
    d = ImageDraw.Draw(img)
    _frame_border(d)
    _logo(d, PAPER_W // 2, 150)
    title_font = _font("RussoOne-Regular.ttf", 150)
    _text_c(d, (PAPER_W // 2, 330), m.title, title_font, NAVY)
    sub_font = _font("ChakraPetch-Bold.ttf", 54)
    _text_c(d, (PAPER_W // 2, 440), m.subtitle, sub_font, NAVY_2)
    tw = d.textlength(m.subtitle, font=sub_font)
    for side in (-1, 1):
        x_in = PAPER_W // 2 + side * (tw / 2 + 24)
        x_out = PAPER_W // 2 + side * (tw / 2 + 170)
        d.line([(x_in, 440), (x_out, 440)], fill=LIME, width=6)
        d.line([(x_in, 454), (x_in + side * 70, 454)], fill=CYAN, width=4)

    _flag_card(img, m.home_flag, (90, 540, 480, 800))
    _flag_card(img, m.away_flag, (PAPER_W - 480, 540, PAPER_W - 90, 800))
    d = ImageDraw.Draw(img)
    _vs(d, PAPER_W // 2, 670)

    _table(img, m.rows)
    _handwrite(img, m.rows, filled_rows, seed)
    return _paper_texture(img)


def _paper_texture(img: Image.Image) -> Image.Image:
    """Печать на бумаге, а не «цифровая» картинка: зерно + мягкий перепад света."""
    import numpy as np
    arr = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(11)
    arr += rng.normal(0, 3.2, arr.shape[:2])[..., None]
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 1.0 - 0.05 * (xx / w) - 0.04 * (yy / h)
    arr *= light[..., None]
    return Image.fromarray(arr.clip(0, 255).astype("uint8"))


# ---------------------------------------------------------------- стол ----

def _procedural_wood(w: int, h: int, seed: int = 3) -> Image.Image:
    """Светлая деревянная столешница без внешних файлов (если нет TABLE_IMAGE)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    y = np.arange(h)[:, None].astype(np.float32)
    x = np.arange(w)[None, :].astype(np.float32)
    grain = np.zeros((h, w), np.float32)
    for k in range(6):
        freq = rng.uniform(0.004, 0.02)
        amp = rng.uniform(4, 30)
        phase = rng.uniform(0, 6.28)
        grain += np.sin(x * freq * (k + 1) * 0.4 + np.sin(y * 0.002 * (k + 1) + phase) * amp) * (1 / (k + 1))
    grain = (grain - grain.min()) / (grain.max() - grain.min())
    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    base = np.array([196, 178, 150], np.float32)
    dark = np.array([160, 138, 108], np.float32)
    t = (grain * 0.75 + 0.25 * (noise * 0.15 + 0.5)).clip(0, 1)[..., None]
    arr = base * (1 - t) + dark * t
    return Image.fromarray(arr.clip(0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1.2))


def compose_frame(paper: Image.Image, table_image: str = "", seed: int = 3) -> Image.Image:
    """Кладёт лист на стол: небольшой поворот, мягкая тень, сверху — как на референсе."""
    if table_image and os.path.exists(table_image):
        bg = Image.open(table_image).convert("RGB")
        scale = max(FRAME_W / bg.width, FRAME_H / bg.height)
        bg = bg.resize((int(bg.width * scale) + 1, int(bg.height * scale) + 1), Image.LANCZOS)
        left, top = (bg.width - FRAME_W) // 2, (bg.height - FRAME_H) // 2
        bg = bg.crop((left, top, left + FRAME_W, top + FRAME_H))
    else:
        bg = _procedural_wood(FRAME_W, FRAME_H, seed)

    target_w = int(FRAME_W * 0.955)
    target_h = int(target_w * PAPER_H / PAPER_W)
    sheet = paper.resize((target_w, target_h), Image.LANCZOS).convert("RGBA")
    angle = -0.6
    sheet = sheet.rotate(angle, resample=Image.BICUBIC, expand=True)

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
    out = Image.alpha_composite(out, dark)
    return out.convert("RGB")
