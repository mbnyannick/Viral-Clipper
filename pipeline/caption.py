"""
Step 6 — Caption image rendering.

Renders the DeepSeek hook text as a white rounded card with centered lines.
All-caps words remain uppercase and are drawn in bold for emphasis.
Emoji is appended to the last line for hook-style punctuation.
"""

import logging
import re
from pathlib import Path


from PIL import Image, ImageDraw, ImageFont

from .errors import PipelineError
from .score import Moment
from .text_utils import mask_profanity

logger = logging.getLogger(__name__)

CANVAS_W: int = 720
CANVAS_H: int = 1280

MAIN_FONT_SIZE = 38
SECONDARY_FONT_SIZE = 32
LINE_GAP = 8
MAX_LINES = 2
TEXT_COLOR = (20, 20, 20, 255)
BOX_COLOR = (255, 255, 255, 255)
BOX_RADIUS = 16
BOX_PAD_X = 22
BOX_PAD_Y = 14
SHADOW_COLOR = (0, 0, 0, 115)
EMOJI_SIZE = 38
WORD_SPACING = 8

_SYSTEM_EMOJI_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto-color-emoji/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]

_EMOJI_CACHE: dict[tuple[str, int], Image.Image] = {}


def _clean_caption_text(text: str) -> str:
    clean = re.sub(r"[☰≡•|~]", "", text)
    return mask_profanity(clean.strip())


def _find_emoji_font(assets_dir: Path) -> str | None:
    bundled = assets_dir / "fonts" / "NotoColorEmoji.ttf"
    if bundled.exists():
        return str(bundled)
    for path in _SYSTEM_EMOJI_PATHS:
        if Path(path).exists():
            return path
    return None


def _load_font_paths(assets_dir: Path) -> tuple[str, str, str | None]:
    inter_bold = assets_dir / "fonts" / "Inter-Bold.ttf"
    bold_otf = assets_dir / "fonts" / "Bold.otf"
    bold_ttf = assets_dir / "fonts" / "Bold.ttf"
    medium_otf = assets_dir / "fonts" / "Medium.otf"

    bold_path = inter_bold if inter_bold.exists() else (bold_otf if bold_otf.exists() else bold_ttf)
    normal_path = medium_otf if medium_otf.exists() else bold_path

    if not Path(bold_path).exists():
        raise FileNotFoundError(f"Font file missing at {bold_path}")

    return str(bold_path), str(normal_path), _find_emoji_font(assets_dir)


def _is_emphasis(word: str) -> bool:
    clean = word.strip(".,!?\"'")
    return bool(clean) and clean.isupper() and len(clean) > 1 and any(c.isalpha() for c in clean)


def _measure_text_width(text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _measure_text_width(candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _get_emoji_image(emoji_char: str, size: int, assets_dir: Path) -> Image.Image | None:
    key = (emoji_char, size)
    if key in _EMOJI_CACHE:
        return _EMOJI_CACHE[key]

    emoji_font_path = _find_emoji_font(assets_dir)
    if not emoji_font_path:
        return None

    try:
        f = ImageFont.truetype(emoji_font_path, 109, index=0)
        temp_img = Image.new("RGBA", (240, 240), (0, 0, 0, 0))
        d = ImageDraw.Draw(temp_img)
        d.text((10, 10), emoji_char, font=f, embedded_color=True)
        bbox = temp_img.getbbox()
        if bbox:
            cropped = temp_img.crop(bbox)
            aspect = cropped.width / cropped.height
            target_w = max(1, int(size * aspect))
            resized = cropped.resize((target_w, size), Image.Resampling.LANCZOS)
            _EMOJI_CACHE[key] = resized
            return resized
    except Exception as exc:
        logger.warning("Could not render emoji '%s': %s", emoji_char, exc)
    return None


def _build_tokens(line_text: str, normal_f: ImageFont.FreeTypeFont, bold_f: ImageFont.FreeTypeFont) -> list[tuple[str, ImageFont.FreeTypeFont]]:
    tokens = []
    for w in line_text.split():
        clean_word = w.strip()
        if not clean_word:
            continue
        if _is_emphasis(clean_word):
            tokens.append((clean_word, bold_f))
        else:
            tokens.append((clean_word, normal_f))
    return tokens


def _extract_emojis(text: str) -> tuple[str, list[str]]:
    plain = []
    emojis = []
    for c in text:
        if ord(c) > 255 and not c.isalnum():
            emojis.append(c)
        else:
            plain.append(c)
    return "".join(plain).strip(), emojis


def _draw_white_card(img: Image.Image, lines: list[str], emoji_str: str | None, assets_dir: Path, layout_mode: str = "blurred_frame") -> None:
    bold_path, normal_path, _ = _load_font_paths(assets_dir)
    normal_f = ImageFont.truetype(normal_path, SECONDARY_FONT_SIZE)
    bold_f = ImageFont.truetype(bold_path, SECONDARY_FONT_SIZE)
    headline_f = ImageFont.truetype(bold_path, MAIN_FONT_SIZE)

    plain_lines = []
    emojis_to_append = []
    for ln in lines:
        p, ems = _extract_emojis(ln)
        if p:
            plain_lines.append(p)
        emojis_to_append.extend(ems)
    if emoji_str:
        _, ems = _extract_emojis(emoji_str)
        emojis_to_append.extend(ems)

    plain_lines = [ln for ln in plain_lines if ln.strip()][:MAX_LINES]
    if len(plain_lines) == 3:
        plain_lines = [plain_lines[0], f"{plain_lines[1]} {plain_lines[2]}".strip()]

    rows: list[tuple[list[tuple[str, ImageFont.FreeTypeFont | None, bool]], ImageFont.FreeTypeFont]] = []
    max_line_width = 0
    total_text_height = 0

    SIDE_MARGIN = 36
    MAX_CARD_WIDTH = CANVAS_W - (SIDE_MARGIN * 2)

    for idx, line in enumerate(plain_lines):
        font = headline_f if idx == 0 else normal_f
        wrapped = _wrap_text(line, font, MAX_CARD_WIDTH - BOX_PAD_X * 2)
        for subline_idx, subline in enumerate(wrapped):
            tokens = [(t, f, False) for t, f in _build_tokens(subline, normal_f, bold_f)]
            if not tokens:
                continue

            if idx == len(plain_lines) - 1 and subline_idx == len(wrapped) - 1:
                for em in emojis_to_append:
                    tokens.append((em, None, True))

            line_width = 0
            for tok_text, f, is_em in tokens:
                if is_em:
                    em_img = _get_emoji_image(tok_text, EMOJI_SIZE, assets_dir)
                    line_width += (em_img.width if em_img else EMOJI_SIZE) + WORD_SPACING
                else:
                    line_width += _measure_text_width(tok_text, f) + WORD_SPACING
            line_width = max(0, line_width - WORD_SPACING)

            max_line_width = max(max_line_width, line_width)
            rows.append((tokens, font))
            total_text_height += font.getbbox("Ay")[3] - font.getbbox("Ay")[1]

    if not rows:
        return

    total_text_height += LINE_GAP * (len(rows) - 1)
    SIDE_MARGIN = 36
    MAX_CARD_WIDTH = CANVAS_W - (SIDE_MARGIN * 2)
    box_width = min(MAX_CARD_WIDTH, max_line_width + BOX_PAD_X * 2)
    box_height = total_text_height + BOX_PAD_Y * 2
    box_x = (CANVAS_W - box_width) // 2
    
    # 40px overlap into top edge of 1:1 main video frame (TikTok / Reels reference positioning)
    CAPTION_OVERLAP = 40
    video_top_y = 280
    box_y = max(15, video_top_y - box_height + CAPTION_OVERLAP)

    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        [(box_x + 4, box_y + 4), (box_x + box_width + 4, box_y + box_height + 4)],
        radius=BOX_RADIUS,
        fill=SHADOW_COLOR,
    )
    img.alpha_composite(shadow)

    card = Image.new("RGBA", img.size, (0, 0, 0, 0))
    card_draw = ImageDraw.Draw(card)
    card_draw.rounded_rectangle(
        [(box_x, box_y), (box_x + box_width, box_y + box_height)],
        radius=BOX_RADIUS,
        fill=BOX_COLOR,
    )
    img.alpha_composite(card)

    draw = ImageDraw.Draw(img)
    y = box_y + BOX_PAD_Y
    for tokens, font in rows:
        line_height = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
        row_width = 0
        for tok_text, f, is_em in tokens:
            if is_em:
                em_img = _get_emoji_image(tok_text, EMOJI_SIZE, assets_dir)
                row_width += (em_img.width if em_img else EMOJI_SIZE) + WORD_SPACING
            else:
                row_width += _measure_text_width(tok_text, f) + WORD_SPACING
        row_width = max(0, row_width - WORD_SPACING)

        x = (CANVAS_W - row_width) // 2
        for tok_text, font_obj, is_em in tokens:
            if is_em:
                em_img = _get_emoji_image(tok_text, EMOJI_SIZE, assets_dir)
                if em_img:
                    ey = y + (line_height - em_img.height) // 2
                    img.alpha_composite(em_img, (x, ey))
                    x += em_img.width + WORD_SPACING
                else:
                    x += EMOJI_SIZE + WORD_SPACING
            else:
                draw.text((x, y), tok_text, font=font_obj, fill=TEXT_COLOR)
                x += _measure_text_width(tok_text, font_obj) + WORD_SPACING
        y += line_height + LINE_GAP


def render_caption(
    moment: Moment,
    assets_dir: Path,
    output_path: Path,
    layout_mode: str = "blurred_frame",
    text_color: tuple[int, int, int] | None = None,
    stroke_color: tuple[int, int, int] | None = None,
    stroke_width: int | None = None,
    include_emoji: bool | None = None,
) -> int:
    eff_include_emoji = True if include_emoji is None else include_emoji

    try:
        _load_font_paths(assets_dir)
    except FileNotFoundError as exc:
        raise PipelineError("caption", str(exc)) from exc

    raw_lines = [_clean_caption_text(ln) for ln in (moment.caption_lines or [])[:3]]
    raw_lines = [ln for ln in raw_lines if ln.strip()]
    emoji_str = moment.emoji.strip() if eff_include_emoji and moment.emoji else None

    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    _draw_white_card(img, raw_lines, emoji_str, assets_dir, layout_mode=layout_mode)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    logger.info(
        "  Caption %02d: %s (white hook card, %d input lines, mode=%s)",
        moment.index, output_path.name, len(raw_lines), layout_mode,
    )
    return CANVAS_H


def render_captions(
    moments: list[Moment],
    assets_dir: Path,
    output_dir: Path,
    layout_mode: str = "pillarbox",
) -> list[tuple[Path, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Path, int]] = []
    for m in moments:
        out = output_dir / f"caption_{m.index:02d}.png"
        height = render_caption(m, assets_dir, out, layout_mode=layout_mode)
        results.append((out, height))
    return results
