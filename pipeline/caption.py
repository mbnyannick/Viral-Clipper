"""
Step 6 — Caption image rendering.

ALL layouts (including face_crop):
  Full-canvas (720×1280) transparent PNG.
  ONE single white rounded-rectangle containing all caption lines,
  overlaid ON the video footage.
  Bold black text, each line centered horizontally inside the box.
  Emoji string appended at the end of the last line.
  Box is at minimum 88% of canvas width so text always fits.
  Composited with enable='between(t,0,5)' — visible 5 seconds, then gone.
"""

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .errors import PipelineError
from .score import Moment
from .subtitle import mask_profanity

logger = logging.getLogger(__name__)

# ── Canvas ────────────────────────────────────────────────────────────────────
CANVAS_W: int = 720
CANVAS_H: int = 1280

# 4:3 video zone on 720×1280 (video = 720×540, centered)
_VIDEO_TOP_Y: int = (CANVAS_H - CANVAS_W * 3 // 4) // 2   # = 370
_VIDEO_BOT_Y: int = _VIDEO_TOP_Y + CANVAS_W * 3 // 4       # = 910

# ── Typography ────────────────────────────────────────────────────────────────
FONT_SIZE: int   = 32        # Clean readable size for top blur bar
LINE_GAP: int    = 8
WORD_GAP: int    = 7

# ── Box style ────────────────────────────────────────────────────────────────
BOX_BG       = (255, 255, 255, 245)  # crisp near-opaque white
BOX_RADIUS   = 14
BOX_PAD_X    = 22            # padding around text inside box
BOX_PAD_Y    = 14
TEXT_COLOR   = (10, 10, 10)  # dark text

# ── Box center positions ──────────────────────────────────────────────────────
# Non-face-crop: Center of top blurry bar (Y=0..370 → center Y=185)
_BOX_CENTER_PILLARBOX: int = 185
# face_crop: Upper safe zone (Y=140)
_BOX_CENTER_FACECROP: int  = 140

_SYSTEM_EMOJI_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto-color-emoji/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]
_BITMAP_FALLBACK_SIZES = [42, 40, 44, 38, 36, 32, 48]


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
    emphasis_font = ImageFont.truetype(str(bold_path),   font_size)

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


def _tok_w(text: str, font: ImageFont.FreeTypeFont, is_emoji: bool = False) -> int:
    try:
        bb = font.getbbox(text)
        w  = bb[2] - bb[0]
        return max(w, FONT_SIZE) if is_emoji else w
    except Exception:
        return FONT_SIZE if is_emoji else 30


def _build_line_tokens(
    line_text: str,
    normal_f: ImageFont.FreeTypeFont,
    emph_f:   ImageFont.FreeTypeFont,
) -> list[tuple]:
    """Word tokens for one line. ALL CAPS words get the bold emphasis font."""
    return [
        (w, emph_f if _is_emphasis(w) else normal_f, False)
        for w in line_text.split()
        if w.strip()
    ]


def _draw_white_card(
    img: Image.Image,
    line_token_rows: list[list[tuple]],
    emoji_str: str | None,
    emoji_f: ImageFont.FreeTypeFont,
    box_center_y: int,
) -> None:
    """Render ONE white rounded-rect card containing all lines onto *img*."""

    # ── Measure row widths (emojis appended individually to last row) ───────
    rows = [list(row) for row in line_token_rows]  # copy
    if emoji_str and rows:
        # Split emoji string into individual emoji characters
        emoji_chars = [c for c in emoji_str if not c.isalnum() and not c.isspace()]
        if not emoji_chars and emoji_str.strip():
            emoji_chars = [emoji_str.strip()]
        for em in emoji_chars:
            rows[-1].append((em, emoji_f, True))

    row_widths = [
        sum(_tok_w(t, f, ie) for t, f, ie in row) + WORD_GAP * max(0, len(row) - 1)
        for row in rows
    ]
    n_lines    = len(rows)
    max_row_w  = max(row_widths) if row_widths else 0

    # ── Box geometry (tight shrinkwrap around text + padding) ──────────────────
    box_inner_w = max_row_w
    box_inner_h = FONT_SIZE * n_lines + LINE_GAP * max(0, n_lines - 1)

    box_w = box_inner_w + BOX_PAD_X * 2
    box_h = box_inner_h + BOX_PAD_Y * 2

    # Cap at canvas width with small margin
    box_w = min(box_w, CANVAS_W - 24)
    box_inner_w = box_w - BOX_PAD_X * 2

    # Centered horizontally
    box_x0 = (CANVAS_W - box_w) // 2
    box_y0 = box_center_y - box_h // 2
    # Clamp so box stays within top area
    box_y0 = max(15, min(box_y0, _VIDEO_TOP_Y - box_h - 10 if box_center_y < _VIDEO_TOP_Y else CANVAS_H - box_h - 20))
    box_x1 = box_x0 + box_w
    box_y1 = box_y0 + box_h

    # ── Draw white card ───────────────────────────────────────────────────────
    card = Image.new("RGBA", img.size, (0, 0, 0, 0))
    cd   = ImageDraw.Draw(card)
    cd.rounded_rectangle([(box_x0, box_y0), (box_x1, box_y1)], radius=BOX_RADIUS, fill=BOX_BG)
    img.alpha_composite(card)

    # ── Draw text ────────────────────────────────────────────────────────────
    draw   = ImageDraw.Draw(img)
    text_y = box_y0 + BOX_PAD_Y

    for row, row_w in zip(rows, row_widths):
        # Center each line horizontally inside the box interior
        text_x = box_x0 + BOX_PAD_X + (box_inner_w - row_w) // 2
        cx = text_x
        for token_text, font, is_emoji in row:
            tw = _tok_w(token_text, font, is_emoji)
            ty = text_y
            if is_emoji:
                try:
                    draw.text((cx, ty), token_text, font=font, embedded_color=True)
                except Exception:
                    try:
                        draw.text((cx, ty), token_text, font=font, fill=TEXT_COLOR)
                    except Exception:
                        logger.warning("Emoji render failed: '%s'", token_text)
            else:
                draw.text((cx, ty), token_text, font=font, fill=TEXT_COLOR)
            cx += tw + WORD_GAP
        text_y += FONT_SIZE + LINE_GAP


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
    Render ONE white card PNG (720×1280 transparent canvas) for *moment*.
    Works for ALL layout modes including face_crop.
    Card is centered on the video at chest level and visible for 5s via FFmpeg.
    Returns CANVAS_H always.
    """
    eff_include_emoji = True if include_emoji is None else include_emoji

    try:
        norm_f, emph_f, emoji_f = _load_fonts(assets_dir, FONT_SIZE)
    except FileNotFoundError as exc:
        raise PipelineError("caption", str(exc)) from exc

    # Sanitize lines
    raw_lines = [mask_profanity(ln) for ln in (moment.caption_lines or [])[:3]]
    raw_lines = [ln for ln in raw_lines if ln.strip()]

    # Build token rows (no emoji in rows — emoji appended separately at the end)
    line_token_rows = [_build_line_tokens(ln, norm_f, emph_f) for ln in raw_lines]
    line_token_rows = [r for r in line_token_rows if r]

    # Emoji string for the end of the last line
    emoji_str: str | None = None
    if eff_include_emoji and moment.emoji:
        emoji_str = moment.emoji.strip() or None

    # Choose box center y based on layout
    if layout_mode == "face_crop":
        box_center_y = _BOX_CENTER_FACECROP
    else:
        box_center_y = _BOX_CENTER_PILLARBOX

    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))

    if line_token_rows or emoji_str:
        _draw_white_card(img, line_token_rows, emoji_str, emoji_f, box_center_y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    logger.info(
        "  Caption %02d: %s (white card, %d lines, center_y=%d, mode=%s)",
        moment.index, output_path.name, len(line_token_rows), box_center_y, layout_mode,
    )
    return CANVAS_H


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
