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
# «Карточка» — это экран смартфона, поэтому пропорции телефонные (~9:19.5),
# а не A4. Неон на бумаге физически не светится, на экране — светится по
# определению: та же анимация цифр перестаёт выглядеть фокусом.
PAPER_W, PAPER_H = 1240, 2684

# --- Coinplay brand palette -------------------------------------------------
# Палитра «криптоказино»: чёрный + золото, фиолет остаётся только как
# фирменный подсвет. Сплошной фиолетовый корпус спорил с тёмным столом на
# фоне — карточка читалась как наклейка поверх фотографии.
BG = (7, 6, 12)            # #07060C  почти чёрный корпус
BG_2 = (13, 12, 20)        # #0D0C14  заливка клеток счёта
BG_3 = (20, 15, 34)        # #140F22  низ градиента, чуть тёплее
PANEL = (15, 13, 24)       # #0F0D18  панель таблицы
SECONDARY = (74, 56, 122)  # #4A387A  приглушённый фиолет для свечений
VIOLET = (124, 88, 240)    # #7C58F0  фирменный фиолет (подсветы, разделители)
LILAC = (186, 160, 255)    # #BAA0FF
GOLD = (230, 181, 74)      # #E6B54A  основной акцент: рамки, обводки
PRIMARY = GOLD             # бывший фиолетовый primary теперь золото
YELLOW = (255, 214, 92)    # #FFD65C  яркое золото: счёт, неон цифр
WHITE = (247, 247, 250)
GREY = (150, 146, 170)     # второстепенный текст
INK = YELLOW               # цвет цифр на тёмной карточке

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
    # Показываем на экране то, что и так посчитано пайплайном: лигу с датой и
    # консенсус моделей. Без них верх и низ экрана оставались пустыми, а
    # главный вывод ролика жил только в подписи к посту.
    competition: str = ""
    date: str = ""
    consensus: str = ""       # кого выбрали модели: «SÃO PAULO FC»
    consensus_note: str = ""  # насколько единодушно: «4 OF 5 MODELS AGREE»


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
    """Почти чёрный корпус с редкими тёплыми подсветами: золото сверху,
    фирменный фиолет сбоку. Свечения держим слабыми — карточка должна
    оставаться тёмной, чтобы неон цифр был самым ярким пятном."""
    base = _vertical_gradient(w, h, BG, BG_3).convert("RGBA")
    _glow_blob(base, int(w * 0.18), int(h * 0.10), int(w * 0.46), GOLD, 26, 60)
    _glow_blob(base, int(w * 0.88), int(h * 0.28), int(w * 0.38), VIOLET, 30, 72)
    _glow_blob(base, int(w * 0.10), int(h * 0.94), int(w * 0.40), SECONDARY, 30, 70)
    _glow_blob(base, int(w * 0.82), int(h * 0.96), int(w * 0.32), GOLD, 20, 50)
    return base


# ----------------------------------------------------------------- лист ----

def _text_c(d, xy, text, font, fill) -> None:
    d.text(xy, text, font=font, fill=fill, anchor="mm")


def _frame_border(d: ImageDraw.ImageDraw) -> None:
    """Рамка интерфейса.

    Начинается НИЖЕ строки состояния: это экран приложения, а строка состояния
    принадлежит системе и рамкой не обводится. Декоративные «чипы» по верхним
    углам убраны — на карточке они читались как типографика, а на экране
    налезали на часы и индикатор батареи.
    """
    m = 34                      # отступ по бокам и снизу
    top = STATUS_Y + 56         # под строкой состояния
    r = 56                      # скругление — как в карточках/кнопках брендбука
    W, H = PAPER_W, PAPER_H
    d.rounded_rectangle([m, top, W - m, H - m], radius=r, outline=PRIMARY, width=8)
    inset = 16
    d.rounded_rectangle([m + inset, top + inset, W - m - inset, H - m - inset],
                        radius=r - 10, outline=YELLOW, width=3)


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
TABLE_Y0 = 1180
ROW_H = 218
ICON_CELL = 170
BOX_W, BOX_H = 212, 152
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


DIM_NAME = (104, 100, 122)   # модель ещё «думает»
DIM_BOX = (52, 44, 30)       # клетка до ответа — холодная, без золота
# Вердикт до финала: те же буквы на тех же местах, но контраст почти нулевой —
# прочитать нельзя, а видео-модели не приходится выдумывать новый текст, она
# лишь «проявляет» уже имеющийся.
HIDDEN = (17, 15, 23)


def _table(img: Image.Image, rows: list, filled_rows: int = 0) -> None:
    """Строка «загорается», когда её модель ответила.

    Пока модель считает, её имя и клетки приглушены; как только счёт появился —
    имя становится белым, а рамки клеток золотыми. Для видео это идеальный
    сигнал: меняется только яркость одних и тех же элементов, никаких новых
    объектов и текста, которые генеративная модель начала бы выдумывать.
    """
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
        done = i < filled_rows
        top, bot = y + i * ROW_H, y + (i + 1) * ROW_H
        cy = (top + bot) // 2
        _paste_icon(img, row.icon, TABLE_X0 + 84, cy, 54)
        d = ImageDraw.Draw(img)  # _paste_icon мог подменить содержимое img
        d.text((NAME_X, cy), row.name.upper(), font=name_font,
               fill=WHITE if done else DIM_NAME, anchor="lm")
        for bx in (HOME_BOX_CX, AWAY_BOX_CX):
            d.rounded_rectangle(_box(bx, cy), radius=18, fill=BG_2,
                                outline=YELLOW if done else DIM_BOX, width=5)
        d.rectangle([mid_x - 16, cy - 4, mid_x + 16, cy + 5],
                    fill=YELLOW if done else DIM_BOX)
        if i < len(rows) - 1:
            d.line([(TABLE_X0 + ICON_CELL, bot), (TABLE_X1 - 6, bot)], fill=VIOLET, width=2)
        for k in range(3):
            d.ellipse([TABLE_X1 - 26, cy - 20 + k * 18, TABLE_X1 - 20, cy - 14 + k * 18], fill=VIOLET)


def digit_style() -> str:
    """Тот же VIDEO_STYLE, что и у ролика: цифры на финальном кадре обязаны
    совпадать с тем, как их «проявляет» видео. Если на кадре-якоре рукописный
    маркер, кибер-промпт бесполезен — модель воспроизведёт почерк."""
    s = (os.environ.get("VIDEO_STYLE") or "cyber").strip().lower()
    return s if s in ("cyber", "marker") else "cyber"


def _neon_digits(img: Image.Image, rows: list, filled_rows: int, seed: int) -> None:
    """Кибер-цифры: техно-шрифт, ровно по центру клетки, с неоновым свечением.

    Никакого разброса угла/позиции — цифра «встала» в табло, а не написана
    рукой. Свечение собирается слоями: широкий ореол, ближний ореол и
    светлое ядро — так неон читается и на тёмной карточке, и после сжатия
    видео, где тонкий контур просто размылся бы.
    """
    for i, row in enumerate(rows[:max(0, filled_rows)]):
        cy = TABLE_Y0 + i * ROW_H + ROW_H // 2
        for cx, val in ((HOME_BOX_CX, row.home), (AWAY_BOX_CX, row.away)):
            if val is None:
                continue
            f = _font("ChakraPetch-Bold.ttf", 104)
            mask = Image.new("L", (220, 220), 0)
            ImageDraw.Draw(mask).text((110, 110), str(val), font=f, fill=255, anchor="mm")

            tile = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
            # Неон — источник света, а не краска: широкий ореол даёт «засветку»
            # вокруг цифры, ближний — плотность, ядро — раскалённый центр.
            for blur, alpha in ((28, 80), (14, 140), (5, 210)):
                halo = Image.new("RGBA", tile.size, YELLOW + (0,))
                halo.putalpha(mask.filter(ImageFilter.GaussianBlur(blur)).point(
                    lambda v, a=alpha: v * a // 255))
                tile = Image.alpha_composite(tile, halo)

            core = Image.new("RGBA", tile.size, (255, 252, 226, 0))
            core.putalpha(mask)
            tile = Image.alpha_composite(tile, core)
            _overlay(img, tile, (cx - 110, cy - 110))


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


STATUS_Y = 104          # центр строки состояния
HEADER_Y = 236          # шапка приложения: знак + название
HOOK_Y = 392            # крупный вопрос-хук
SUB_Y = 496             # «5 AI MODELS PREDICT»
META_Y = 566            # лига и дата
FLAG_TOP, FLAG_BOT = 636, 916
VS_Y = 776
NAMES_Y = 978           # подписи команд под эмблемами
PROGRESS_Y = 1094       # центр полосы «AI анализирует»
CONSENSUS_Y0 = 2318     # верх плашки с вердиктом
CONSENSUS_H = 170
CTA_Y0 = 2516           # нижняя строка с призывом и 18+


def _hook(d: ImageDraw.ImageDraw) -> None:
    """Вопрос, который останавливает скролл.

    В ленте у ролика есть полторы секунды: тёмный экран на 0% их проигрывает.
    Короткая фраза работает лучше длинной ещё и технически — чем меньше букв,
    тем меньше шансов, что видео-модель их исказит.
    """
    _text_c(d, (PAPER_W // 2, HOOK_Y), "WHO WINS?",
            _font("RobotoCondensed-Bold.ttf", 118), YELLOW)


def _app_header(d: ImageDraw.ImageDraw, img: Image.Image) -> None:
    """Знак и название бренда одной компактной строкой, как шапка приложения:
    крупный заголовок уступил место хуку, но бренд из кадра не исчезает."""
    font = _font("RobotoCondensed-Bold.ttf", 52)
    text = "COINPLAY AI LAB"
    tw = d.textlength(text, font=font)
    total = 112 + 22 + tw
    x0 = (PAPER_W - total) / 2
    _logo(d, int(x0 + 56), HEADER_Y, img)
    ImageDraw.Draw(img).text((x0 + 112 + 22, HEADER_Y), text, font=font, fill=WHITE, anchor="lm")


def _cta_bar(img: Image.Image) -> None:
    """Призыв и возрастная пометка.

    Статичны на всех кадрах — видео-модель их не трогает, а зритель получает
    призыв внутри самого ролика, а не только в подписи поста, которую в ленте
    часто не разворачивают.
    """
    d = ImageDraw.Draw(img)
    x0, x1 = TABLE_X0, TABLE_X1
    y0, y1 = CTA_Y0, CTA_Y0 + 96
    d.rounded_rectangle([x0, y0, x1, y1], radius=22, fill=(24, 20, 12), outline=GOLD, width=3)
    cy = (y0 + y1) // 2
    d.text((PAPER_W // 2 - 44, cy), "LINK IN BIO",
           font=_font("RobotoCondensed-Bold.ttf", 50), fill=YELLOW, anchor="mm")
    badge = _font("RobotoCondensed-Bold.ttf", 34)
    bx = PAPER_W // 2 + 120
    d.ellipse([bx - 34, cy - 34, bx + 34, cy + 34], outline=GREY, width=3)
    d.text((bx, cy + 1), "18+", font=badge, fill=GREY, anchor="mm")


def _progress(img: Image.Image, done: int, total: int) -> None:
    """Полоса прогресса «сколько моделей уже ответили».

    Ради неё всё и затевалось: между двумя кадрами модель сама дорисовывает
    рост полосы, и ролик читается как «ИИ считает → выдал результат», а не
    как набор цифр, возникших из ниоткуда. Растёт ровно как заполняется
    таблица, поэтому на стыке сегментов ничего не прыгает.
    """
    if total <= 0:
        return
    d = ImageDraw.Draw(img)
    frac = max(0.0, min(1.0, done / total))
    label = _font("RobotoCondensed-SemiBold.ttf", 34)
    d.text((TABLE_X0 + 6, PROGRESS_Y - 44), "AI ANALYSIS", font=label, fill=GREY, anchor="lm")
    d.text((TABLE_X1 - 6, PROGRESS_Y - 44), f"{int(round(frac * 100))}%",
           font=_font("RobotoCondensed-Bold.ttf", 36),
           fill=YELLOW if frac >= 1 else WHITE, anchor="rm")

    x0, x1 = TABLE_X0 + 6, TABLE_X1 - 6
    y0, y1 = PROGRESS_Y - 8, PROGRESS_Y + 10
    d.rounded_rectangle([x0, y0, x1, y1], radius=9, fill=(24, 22, 32), outline=(46, 42, 58), width=2)
    if frac > 0:
        # img здесь всегда RGBA: _background() отдаёт RGBA, а _card_texture()
        # переводит в RGB уже после всей отрисовки.
        fill_x = max(x0 + int((x1 - x0) * frac), x0 + 18)
        bar = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(bar).rounded_rectangle([x0, y0, fill_x, y1], radius=9, fill=YELLOW + (255,))
        img.alpha_composite(bar.filter(ImageFilter.GaussianBlur(14)))  # свечение заполненной части
        img.alpha_composite(bar)


def _team_names(img: Image.Image, m: Match) -> None:
    """Подписи команд под эмблемами: по гербу не всегда понятно, кто играет,
    а ставка делается именно на команду."""
    d = ImageDraw.Draw(img)
    for name, cx in ((m.home, (90 + 480) // 2), (m.away, (PAPER_W - 480 + PAPER_W - 90) // 2)):
        text = name.upper()
        font = _fit_font(d, text, 400, "RobotoCondensed-Bold.ttf", 52, 26)
        d.text((cx, NAMES_Y), text, font=font, fill=WHITE, anchor="mm")


def _status_bar(d: ImageDraw.ImageDraw) -> None:
    """Строка состояния и «дырка» фронталки.

    Самая дешёвая деталь, которая мгновенно читается как экран телефона, а не
    как картинка: время слева, связь/wi-fi/батарея справа, камера по центру.
    Время фиксированное — это оформление, а не реальные часы устройства.
    """
    d.text((88, STATUS_Y), "20:45", font=_font("RobotoCondensed-Bold.ttf", 46),
           fill=WHITE, anchor="lm")

    cx = PAPER_W // 2
    d.ellipse([cx - 18, STATUS_Y - 18, cx + 18, STATUS_Y + 18], fill=(3, 3, 6))
    d.ellipse([cx - 18, STATUS_Y - 18, cx + 18, STATUS_Y + 18], outline=(30, 30, 38), width=2)

    right = PAPER_W - 88
    # батарея (справа налево): корпус, заряд, контакт
    bw, bh = 64, 30
    bx1 = right - 10
    bx0, by0 = bx1 - bw, STATUS_Y - bh // 2
    d.rounded_rectangle([bx1 + 3, STATUS_Y - 8, bx1 + 9, STATUS_Y + 8], radius=3, fill=WHITE)
    d.rounded_rectangle([bx0, by0, bx1, by0 + bh], radius=9, outline=WHITE, width=3)
    d.rounded_rectangle([bx0 + 6, by0 + 6, bx0 + 6 + int((bw - 12) * 0.78), by0 + bh - 6],
                        radius=5, fill=WHITE)

    # wi-fi: три дуги и точка
    wx = bx0 - 44
    for r in (12, 23, 34):
        d.arc([wx - r, STATUS_Y - r + 10, wx + r, STATUS_Y + r + 10],
              start=215, end=325, fill=WHITE, width=5)
    d.ellipse([wx - 5, STATUS_Y + 10, wx + 5, STATUS_Y + 20], fill=WHITE)

    # уровень сигнала: четыре растущих столбика
    sx = wx - 96
    for i in range(4):
        hgt = 11 + i * 8
        d.rounded_rectangle([sx + i * 15, STATUS_Y + 14 - hgt, sx + i * 15 + 9, STATUS_Y + 16],
                            radius=3, fill=WHITE)


def _fit_font(d: ImageDraw.ImageDraw, text: str, max_w: int,
              name: str, big: int, small: int) -> ImageFont.FreeTypeFont:
    """Подбирает кегль, пока строка не влезет — названия команд бывают длинные
    («Olympique de Marseille»), и жёсткий размер их обрезал бы."""
    for size in range(big, small - 1, -2):
        font = _font(name, size)
        if d.textlength(text, font=font) <= max_w:
            return font
    return _font(name, small)


def _consensus_bar(img: Image.Image, m: Match, lit: bool) -> None:
    """Нижняя плашка с вердиктом моделей — главный вывод ролика.

    Геометрия и текст одинаковы на всех кадрах, различается только яркость:
    пока модели считают, плашка приглушена, а когда ответили все — загорается
    золотом со свечением.
    """
    if not m.consensus:
        return
    d = ImageDraw.Draw(img)
    x0, x1 = TABLE_X0, TABLE_X1
    y0, y1 = CONSENSUS_Y0, CONSENSUS_Y0 + CONSENSUS_H
    cx = (x0 + x1) // 2
    d.rounded_rectangle([x0, y0, x1, y1], radius=26, fill=BG_2,
                        outline=GOLD if lit else DIM_BOX, width=4)
    d.text((cx, y0 + 34), "AI CONSENSUS", font=_font("RobotoCondensed-SemiBold.ttf", 34),
           fill=GREY if lit else DIM_NAME, anchor="mm")

    text = m.consensus.upper()
    font = _fit_font(d, text, x1 - x0 - 70, "RobotoCondensed-Bold.ttf", 64, 32)
    if not lit:
        d.text((cx, y0 + 90), text, font=font, fill=HIDDEN, anchor="mm")
        if m.consensus_note:
            d.text((cx, y0 + 140), m.consensus_note.upper(),
                   font=_font("RobotoCondensed-SemiBold.ttf", 36), fill=HIDDEN, anchor="mm")
        return

    halo = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(halo).text((cx, y0 + 90), text, font=font, fill=YELLOW + (255,), anchor="mm")
    img.alpha_composite(halo.filter(ImageFilter.GaussianBlur(18)))
    d = ImageDraw.Draw(img)
    d.text((cx, y0 + 90), text, font=font, fill=(255, 248, 206), anchor="mm")
    # Сила согласия — главный аргумент вместо коэффициента: своих кф у нас нет,
    # а выдумывать их в iGaming-креативе нельзя.
    if m.consensus_note:
        d.text((cx, y0 + 140), m.consensus_note.upper(),
               font=_font("RobotoCondensed-SemiBold.ttf", 36), fill=GOLD, anchor="mm")


def render_paper(m: Match, filled_rows: int = 0, seed: int = 7) -> Image.Image:
    img = _background(PAPER_W, PAPER_H)
    d = ImageDraw.Draw(img)
    _frame_border(d)
    _status_bar(d)
    _app_header(d, img)
    d = ImageDraw.Draw(img)
    _hook(d)

    sub_font = _font("RobotoCondensed-SemiBold.ttf", 50)
    _text_c(d, (PAPER_W // 2, SUB_Y), m.subtitle, sub_font, PRIMARY)
    tw = d.textlength(m.subtitle, font=sub_font)
    for side in (-1, 1):
        x_in = PAPER_W // 2 + side * (tw / 2 + 24)
        x_out = PAPER_W // 2 + side * (tw / 2 + 170)
        d.line([(x_in, SUB_Y), (x_out, SUB_Y)], fill=YELLOW, width=6)
        d.line([(x_in, SUB_Y + 14), (x_in + side * 70, SUB_Y + 14)], fill=VIOLET, width=4)

    meta = " · ".join(x for x in (m.competition, m.date) if x)
    if meta:
        _text_c(d, (PAPER_W // 2, META_Y), meta.upper(),
                _fit_font(d, meta.upper(), TABLE_X1 - TABLE_X0 - 40,
                          "RobotoCondensed-SemiBold.ttf", 40, 26), GREY)

    _flag_card(img, m.home_flag, (90, FLAG_TOP, 480, FLAG_BOT))
    _flag_card(img, m.away_flag, (PAPER_W - 480, FLAG_TOP, PAPER_W - 90, FLAG_BOT))
    _vs(ImageDraw.Draw(img), PAPER_W // 2, VS_Y)
    _team_names(img, m)
    _progress(img, min(max(0, filled_rows), len(m.rows)), len(m.rows))

    _table(img, m.rows, filled_rows)
    if digit_style() == "marker":
        _handwrite(img, m.rows, filled_rows, seed)
    else:
        _neon_digits(img, m.rows, filled_rows, seed)
    # Вердикт «загорается», когда ответили все модели: сами буквы стоят на
    # месте с первого кадра, меняется только яркость — такое видео-модель
    # отрабатывает чисто, а появись текст в финале, она бы его исказила.
    _consensus_bar(img, m, lit=len(m.rows) > 0 and filled_rows >= len(m.rows))
    _cta_bar(img)
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


def _perspective_coeffs(src_pts, dst_pts):
    """Коэффициенты для Image.PERSPECTIVE (PIL тянет ИЗ dst В src)."""
    rows = []
    for (sx, sy), (dx, dy) in zip(src_pts, dst_pts):
        rows.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        rows.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    a = np.array(rows, dtype=np.float64)
    b = np.array(src_pts, dtype=np.float64).reshape(8)
    return np.linalg.solve(a, b)


def _keystone(sheet: Image.Image, shrink: float = 0.045) -> Image.Image:
    """Лёгкий наклон: верх уже низа, как у листа, снятого чуть сверху-спереди.

    Без этого карточка — идеально осевой прямоугольник поверх фотографии,
    снятой под углом, и глаз сразу читает её как наклейку.
    """
    w, h = sheet.size
    dx = w * shrink
    dst = [(dx, 0), (w - dx, 0), (w, h), (0, h)]          # трапеция в кадре
    src = [(0, 0), (w, 0), (w, h), (0, h)]                 # исходный прямоугольник
    return sheet.transform((w, h), Image.PERSPECTIVE, _perspective_coeffs(src, dst),
                           resample=Image.BICUBIC)


def _relight(sheet: Image.Image, bg: Image.Image, x: int, y: int) -> Image.Image:
    """Переносит светотень сцены на карточку.

    Главная причина эффекта «наложено»: карточка освещена ровно, а фон —
    с направленным светом и почти чёрным центром. Берём яркость фона под
    карточкой, сильно размываем (остаётся только градиент света, без деталей)
    и умножаем на него карточку. Тогда она темнеет там же, где темнеет стол.
    """
    w, h = sheet.size
    patch = bg.convert("L").crop((x, y, x + w, y + h)).filter(ImageFilter.GaussianBlur(90))
    lum = np.asarray(patch, dtype=np.float32) / 255.0
    mean = float(lum.mean()) or 1.0
    # Нормируем вокруг среднего: важна ФОРМА градиента, а не абсолютная
    # темнота фона — иначе на чёрном столе карточка стала бы нечитаемой.
    gain = np.clip(0.88 + 0.34 * (lum - mean) / max(mean, 0.05), 0.74, 1.14)[..., None]
    arr = np.asarray(sheet, dtype=np.float32)
    arr[..., :3] = np.clip(arr[..., :3] * gain, 0, 255)
    return Image.fromarray(arr.astype("uint8"), "RGBA")


def _match_grain(sheet: Image.Image, seed: int, amount: float = 5.0) -> Image.Image:
    """Немного зерна: фотофон шумит, идеально чистая карточка выдаёт себя."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(sheet, dtype=np.float32)
    noise = rng.normal(0.0, amount, arr.shape[:2])[..., None]
    arr[..., :3] = np.clip(arr[..., :3] + noise, 0, 255)
    return Image.fromarray(arr.astype("uint8"), "RGBA")


BEZEL = 26          # рамка вокруг экрана, в пикселях экрана
SCREEN_RADIUS = 96  # скругление самого экрана
BODY_RADIUS = 122   # скругление корпуса


def _phone(screen: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Собирает смартфон вокруг готового экрана.

    Возвращает (устройство, маска-экрана): маска нужна, чтобы потом НЕ гасить
    экран светотенью сцены — он сам источник света — и чтобы посчитать
    засветку, которую экран бросает на стол.
    """
    sw, sh = screen.size
    w, h = sw + BEZEL * 2, sh + BEZEL * 2

    body = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body)
    bd.rounded_rectangle([0, 0, w - 1, h - 1], radius=BODY_RADIUS, fill=(16, 16, 20, 255))
    # тонкий металлический кант по грани корпуса
    bd.rounded_rectangle([0, 0, w - 1, h - 1], radius=BODY_RADIUS, outline=(92, 88, 104, 255), width=5)
    bd.rounded_rectangle([3, 3, w - 4, h - 4], radius=BODY_RADIUS - 3, outline=(34, 32, 40, 255), width=3)

    # экран со скруглёнными углами
    rounded = Image.new("L", (sw, sh), 0)
    ImageDraw.Draw(rounded).rounded_rectangle([0, 0, sw - 1, sh - 1], radius=SCREEN_RADIUS, fill=255)
    screen = screen.convert("RGBA")
    screen.putalpha(rounded)
    body.alpha_composite(screen, (BEZEL, BEZEL))

    screen_mask = Image.new("L", (w, h), 0)
    screen_mask.paste(rounded, (BEZEL, BEZEL))
    return body, screen_mask


def _screen_spill(out: Image.Image, device: Image.Image, screen_mask: Image.Image,
                  x: int, y: int) -> Image.Image:
    """Свет экрана, падающий на стол вокруг телефона.

    Без него телефон — тёмный прямоугольник с картинкой внутри; именно
    засветка вокруг читается как «экран включён».
    """
    glow = Image.new("RGBA", out.size, (0, 0, 0, 0))
    lit = Image.new("RGBA", device.size, (0, 0, 0, 0))
    lit.paste(device, (0, 0), screen_mask)
    glow.alpha_composite(lit, (x, y))
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    alpha = glow.split()[-1].point(lambda a: int(a * 0.55))
    glow.putalpha(alpha)
    return Image.alpha_composite(out, glow)


def compose_frame(paper: Image.Image, table_image: str = "", seed: int = 3) -> Image.Image:
    """Кладёт смартфон с прогнозом на стол: перспектива, светотень сцены,
    контактная тень и засветка от экрана."""
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

    # Раньше лист занимал 0.955 ширины кадра — почти край в край, без стола
    # вокруг. На видео это давало эффект "всё лицо в кадре": рука и маркер на
    # таком масштабе перекрывают половину постера, а любые фоновые предметы
    # (кружка, растение), которых нет на самом кадре-якоре, начинают наезжать
    # на края листа — их просто некуда деть. 0.72 — как на референсе (лист
    # занимает ~70% ширины, сверху и по бокам видно стол).
    device, screen_mask = _phone(paper)
    target_w = int(FRAME_W * 0.60)
    target_h = int(target_w * device.height / device.width)
    device = device.resize((target_w, target_h), Image.LANCZOS)
    screen_mask = screen_mask.resize((target_w, target_h), Image.LANCZOS)

    device = _keystone(device, shrink=0.03)
    screen_mask = _keystone(screen_mask.convert("RGBA"), shrink=0.03).convert("L")
    device = device.rotate(-0.6, resample=Image.BICUBIC, expand=True)
    screen_mask = screen_mask.rotate(-0.6, resample=Image.BICUBIC, expand=True)

    x = (FRAME_W - device.width) // 2
    y = (FRAME_H - device.height) // 2

    # Корпус живёт по законам сцены (светотень + зерно), а экран — нет:
    # он сам источник света, поэтому возвращаем его поверх нетронутым.
    lit_body = _match_grain(_relight(device, bg, x, y), seed)
    lit_body.paste(device, (0, 0), screen_mask)
    device = lit_body

    mask = device.split()[-1]
    out = bg.convert("RGBA")
    for offset, blur, opacity in (((6, 12), 44, 120), ((2, 5), 10, 160)):
        layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
        layer.paste((0, 0, 0, 255), (x + offset[0], y + offset[1]),
                    mask.point(lambda a, o=opacity: o if a else 0))
        out = Image.alpha_composite(out, layer.filter(ImageFilter.GaussianBlur(blur)))
    out = _screen_spill(out, device, screen_mask, x, y)
    out.alpha_composite(device, (x, y))
    # лёгкая виньетка/неравномерный свет — ближе к фото с телефона
    vign = Image.new("L", out.size, 0)
    ImageDraw.Draw(vign).ellipse([-300, -200, FRAME_W + 300, FRAME_H + 300], fill=255)
    vign = vign.filter(ImageFilter.GaussianBlur(220))
    dark = Image.new("RGBA", out.size, (0, 0, 0, 255))
    dark.putalpha(vign.point(lambda v: int((255 - v) * 0.22)))
    return Image.alpha_composite(out, dark).convert("RGB")
