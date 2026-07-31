"""
Step 6 — Caption image rendering: TOP/BOTTOM split layout.

Non-face-crop layouts:
  - Renders a full-canvas (720×1280) transparent PNG.
  - TOP bar: 1 line (hook/tease), ~56px bold, above the 4:3 video zone.
  - BOTTOM bars: up to 2 lines (payoff), ~46px bold, below the 4:3 video zone.
  - Each line: dark semi-opaque rounded rect (rgba 0,0,0,195) fitted to text width.
  - Emoji prepended to every line.
  - Overlay in composite at (0, 0) — no offset calculation needed.

face_crop layout:
  - Small transparent PNG, white bold text + black stroke (unchanged).
"""

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .errors import PipelineError
from .score import Moment
from .subtitle import mask_profanity

logger = logging.getLogger(__name__)

# ── Canvas ───────────────────────────────────────────────────────────────────────
CANVAS_W: int = 720
CANVAS_H: int = 1280

# 4:3 video on 720×1280: width=720, height=540, centered
_VIDEO_TOP_Y: int = (CANVAS_H - CANVAS_W * 3 // 4) // 2   # = 370
_VIDEO_BOT_Y: int = _VIDEO_TOP_Y + CANVAS_W * 3 // 4       # = 910
_BOTTOM_LIMIT: int = int(CANVAS_H * 0.85)                   # = 1088 (clear of UI zone)

# ── Typography ───────────────────────────────────────────────────────────────────
TOP_FONT_SIZE: int = 56     # ~7.8% of CANVAS_W (hook line)
BOT_FONT_SIZE: int = 46     # ~6.4% of CANVAS_W (payoff lines)
WORD_GAP: int = 8

# ── Pill style ───────────────────────────────────────────────────────────────────
PILL_ALPHA: int = 195       # ~76% opacity
PILL_BG = (0, 0, 0, PILL_ALPHA)
PILL_RADIUS: int = 12
PILL_PAD_X: int = 22
PILL_PAD_Y: int = 10

TEXT_COLOR = (255, 255, 255)

_SYSTEM_EMOJI_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto-color-emoji/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]
_BITMAP_FALLBACK_SIZES = [56, 48, 46, 40, 44, 32, 64]


def _is_emphasis(word: str) -> bool:
    clean = word.strip(".,!?\"'")
    return bool(clean) and clean.isupper() and len(clean) > 1 and any(c.isalpha() for c in clean)


def _find_emoji_font(assets_dir: Path) -> str | None:
    bundled = assets_dir / "fonts" / "NotoColorEmoji.ttf"
    if bundled.exists():
        return str(bundled)
    for path in _SYSTEM_EMOJI_PATHS:
        if Path(path).exists():
            return path
    return None


def _load_fonts(
    assets_dir: Path,
    font_size: int,
) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    inter_bold = assets_dir / "fonts" / "Inter-Bold.ttf"
    bold_otf   = assets_dir / "fonts" / "Bold.otf"
    bold_ttf   = assets_dir / "fonts" / "Bold.ttf"
    medium_otf = assets_dir / "fonts" / "Medium.otf"

    bold_path   = inter_bold if inter_bold.exists() else (bold_otf if bold_otf.exists() else bold_ttf)
    medium_path = medium_otf if medium_otf.exists() else bold_path

    if not bold_path.exists():
        raise FileNotFoundError(f"Font file missing at {bold_path}")

    normal_font   = ImageFont.truetype(str(medium_path), font_size)
    emphasis_font = ImageFont.truetype(str(bold_path), font_size)

    emoji_font = None
    emoji_path = _find_emoji_font(assets_dir)
    if emoji_path:
        for sz in [font_size] + _BITMAP_FALLBACK_SIZES:
            try:
                emoji_font = ImageFont.truetype(emoji_path, sz, index=0)
                break
            except OSError:
                continue
    if emoji_font is None:
        emoji_font = normal_font

    return normal_font, emphasis_font, emoji_font


def _tok_w(text: str, font: ImageFont.FreeTypeFont, is_emoji: bool, em_sz: int) -> int:
    try:
        bb = font.getbbox(text)
        w = bb[2] - bb[0]
        return max(w, em_sz) if is_emoji else w
    except Exception:
        return em_sz if is_emoji else 30


def _line_px_w(tokens: list, em_sz: int) -> int:
    if not tokens:
        return 0
    return (
        sum(_tok_w(t, f, ie, em_sz) for t, f, ie in tokens)
        + WORD_GAP * (len(tokens) - 1)
    )


def _extract_emojis(emoji_str: str) -> list[str]:
    chars = [c for c in emoji_str if not c.isalnum() and not c.isspace()]
    return (chars if chars else [emoji_str])[:3]


def _make_tokens(
    text: str,
    normal_f: ImageFont.FreeTypeFont,
    emph_f: ImageFont.FreeTypeFont,
    emoji_f: ImageFont.FreeTypeFont,
    prepend_emoji: str | None,
) -> list[tuple]:
    toks: list[tuple] = []
    if prepend_emoji:
        toks.append((prepend_emoji, emoji_f, True))
    for word in text.split():
        toks.append((word, emph_f if _is_emphasis(word) else normal_f, False))
    return toks


def _draw_pill_line(
    img: Image.Image,
    tokens: list[tuple],
    font_size: int,
    y: int,
    em_sz: int,
) -> None:
    """Composite a dark rounded-rect pill + text tokens onto *img* at row *y*."""
    lw = _line_px_w(tokens, em_sz)
    x = (CANVAS_W - lw) // 2

    # Draw pill via alpha_composite
    pill = Image.new("RGBA", img.size, (0, 0, 0, 0))
    pd = ImageDraw.Draw(pill)
    pd.rounded_rectangle(
        [
            (max(0, x - PILL_PAD_X), y - PILL_PAD_Y),
            (min(CANVAS_W, x + lw + PILL_PAD_X), y + font_size + PILL_PAD_Y),
        ],
        radius=PILL_RADIUS,
        fill=PILL_BG,
    )
    img.alpha_composite(pill)

    # Draw text tokens
    draw = ImageDraw.Draw(img)
    cx = x
    for text, font, is_emoji in tokens:
        tw = _tok_w(text, font, is_emoji, em_sz)
        if is_emoji:
            try:
                draw.text((cx, y), text, font=font, embedded_color=True)
            except Exception:
                try:
                    draw.text((cx, y), text, font=font, fill=TEXT_COLOR)
                except Exception:
                    pass
        else:
            draw.text((cx, y), text, font=font, fill=TEXT_COLOR)
        cx += tw + WORD_GAP


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
    """
    Render caption PNG for *moment*.

    Non-face-crop: full 720×1280 transparent PNG with TOP bar above and
    BOTTOM bars below the 4:3 video zone. Returns CANVAS_H (1280).

    face_crop: small transparent PNG with white bold text + stroke.
    Returns actual pixel height.
    """
    eff_include_emoji = True if include_emoji is None else include_emoji

    if layout_mode == "face_crop":
        return _render_face_crop(
            moment, assets_dir, output_path,
            text_color=text_color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            include_emoji=eff_include_emoji,
        )

    try:
        top_norm, top_emph, top_emoji = _load_fonts(assets_dir, TOP_FONT_SIZE)
        bot_norm, bot_emph, bot_emoji = _load_fonts(assets_dir, BOT_FONT_SIZE)
    except FileNotFoundError as exc:
        raise PipelineError("caption", str(exc)) from exc

    emoji_chars = _extract_emojis(moment.emoji) if (eff_include_emoji and moment.emoji) else []

    # Line 0 = TOP hook, lines 1-2 = BOTTOM payoff
    clean = [mask_profanity(ln) for ln in (moment.caption_lines or [])[:3]]
    top_text = clean[0] if clean else ""
    bot_texts = clean[1:3]

    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))

    # ── TOP bar ──────────────────────────────────────────────────────────────────
    em0 = emoji_chars[0] if emoji_chars else None
    top_tokens = _make_tokens(top_text, top_norm, top_emph, top_emoji, em0)
    top_line_h = TOP_FONT_SIZE + PILL_PAD_Y * 2
    top_y = max(10, _VIDEO_TOP_Y - top_line_h - 18)
    if top_tokens:
        _draw_pill_line(img, top_tokens, TOP_FONT_SIZE, top_y, TOP_FONT_SIZE)

    # ── BOTTOM bars ──────────────────────────────────────────────────────────────
    LINE_GAP = 14
    bot_y = _VIDEO_BOT_Y + 18
    for i, bt in enumerate(bot_texts):
        em = emoji_chars[i + 1] if (i + 1) < len(emoji_chars) else em0
        bot_tokens = _make_tokens(bt, bot_norm, bot_emph, bot_emoji, em)
        if not bot_tokens:
            continue
        line_h = BOT_FONT_SIZE + PILL_PAD_Y * 2
        if bot_y + line_h > _BOTTOM_LIMIT:
            break
        _draw_pill_line(img, bot_tokens, BOT_FONT_SIZE, bot_y, BOT_FONT_SIZE)
        bot_y += line_h + LINE_GAP

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    logger.info(
        "  Caption %02d: %s (full-canvas 720×1280, mode=%s)",
        moment.index, output_path.name, layout_mode,
    )
    return CANVAS_H


def _render_face_crop(
    moment: Moment,
    assets_dir: Path,
    output_path: Path,
    text_color: tuple[int, int, int] | None = None,
    stroke_color: tuple[int, int, int] | None = None,
    stroke_width: int | None = None,
    include_emoji: bool = True,
) -> int:
    """face_crop: small transparent PNG, white text + stroke (unchanged behavior)."""
    FC_SIZE = 36
    FC_WORD_GAP = 8
    FC_LINE_GAP = 6
    FC_MAX_W = 500

    try:
        norm_f, emph_f, emoji_f = _load_fonts(assets_dir, FC_SIZE)
    except FileNotFoundError as exc:
        raise PipelineError("caption", str(exc)) from exc

    eff_text   = text_color   if text_color   is not None else (255, 255, 255)
    eff_stroke = stroke_color if stroke_color is not None else (0, 0, 0)
    eff_sw     = stroke_width if stroke_width is not None else 3

    clean_text = mask_profanity(" ".join(moment.caption_lines or []))
    tokens: list[tuple] = []
    if include_emoji and moment.emoji:
        for em in _extract_emojis(moment.emoji)[:1]:
            tokens.append((em, emoji_f, True))
    for word in clean_text.split():
        tokens.append((word, emph_f if _is_emphasis(word) else norm_f, False))

    # Wrap into lines
    lines: list[list[tuple]] = []
    cur: list[tuple] = []
    cur_w = 0
    for tok in tokens:
        t, f, ie = tok
        tw = _tok_w(t, f, ie, FC_SIZE)
        added = tw if not cur else FC_WORD_GAP + tw
        if cur and cur_w + added > FC_MAX_W:
            lines.append(cur)
            cur = [tok]
            cur_w = tw
        else:
            cur.append(tok)
            cur_w += added
    if cur:
        lines.append(cur)
    lines = lines[:4]

    LINE_H = FC_SIZE + 4
    total_h = max(60, LINE_H * len(lines) + FC_LINE_GAP * max(0, len(lines) - 1) + 20)

    img = Image.new("RGBA", (CANVAS_W, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = 10
    for line in lines:
        lw = sum(_tok_w(t, f, ie, FC_SIZE) for t, f, ie in line) + FC_WORD_GAP * (len(line) - 1)
        x = (CANVAS_W - lw) // 2
        for text, font, is_emoji in line:
            tw = _tok_w(text, font, is_emoji, FC_SIZE)
            if is_emoji:
                try:
                    draw.text((x, y), text, font=font, embedded_color=True)
                except Exception:
                    pass
            else:
                draw.text((x, y), text, font=font, fill=eff_text,
                          stroke_width=eff_sw, stroke_fill=eff_stroke)
            x += tw + FC_WORD_GAP
        y += LINE_H + FC_LINE_GAP

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    return total_h


def render_captions(
    moments: list[Moment],
    assets_dir: Path,
    output_dir: Path,
    layout_mode: str = "pillarbox",
) -> list[tuple[Path, int]]:
    """Render all captions. Returns list of (png_path, height_px) in moment order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Path, int]] = []
    for m in moments:
        out = output_dir / f"caption_{m.index:02d}.png"
        height = render_caption(m, assets_dir, out, layout_mode=layout_mode)
        results.append((out, height))
    return results
