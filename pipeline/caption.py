"""
Step 6 — Caption image rendering.

ALL layouts (including face_crop):
  Full-canvas (720×1280) transparent PNG.
  ONE single white rounded-rectangle containing all caption lines,
  overlaid ON the video footage.
  Bold black text, each line centered horizontally inside the box.
  Color emojis rendered cleanly using NotoColorEmoji.
  Box fits tightly around text and sits just above the top of the video frame.
"""

import logging
import re
from pathlib import Path


from PIL import Image, ImageDraw, ImageFont

from .errors import PipelineError
from .score import Moment
from .subtitle import mask_profanity

logger = logging.getLogger(__name__)

# ── 1080p Full HD Canvas (1080×1920) ───────────────────────────────────────────
CANVAS_W: int = 1080
CANVAS_H: int = 1920

# 4:3 video zone on 1080×1920 (video = 1080×810, centered)
_VIDEO_TOP_Y: int = (CANVAS_H - CANVAS_W * 3 // 4) // 2   # = 555
_VIDEO_BOT_Y: int = _VIDEO_TOP_Y + CANVAS_W * 3 // 4       # = 1365

# ── Typography & 1080p Scaling ────────────────────────────────────────────────
FONT_SIZE: int   = 51        # Crisp readable 1080p size for top blur bar
LINE_GAP: int    = 12
WORD_GAP: int    = 10

# ── Box style (1080p Full HD rounded white card) ─────────────────────────────
BOX_BG       = (255, 255, 255, 255)  # crisp pure white
BOX_RADIUS   = 24                    # smooth 24px rounded corners @ 1080p
BOX_PAD_X    = 36                    # spacious side padding @ 1080p
BOX_PAD_Y    = 21                    # top/bottom padding @ 1080p
TEXT_COLOR   = (15, 15, 15)          # bold dark text

# ── Box center positions ──────────────────────────────────────────────────────
_BOX_CENTER_PILLARBOX: int = 450
_BOX_CENTER_FACECROP: int  = 210

_SYSTEM_EMOJI_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto-color-emoji/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]

_EMOJI_CACHE: dict[tuple[str, int], Image.Image] = {}


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


def _get_emoji_image(emoji_char: str, size: int, assets_dir: Path) -> Image.Image | None:
    key = (emoji_char, size)
    if key in _EMOJI_CACHE:
        return _EMOJI_CACHE[key]

    emoji_font_path = _find_emoji_font(assets_dir)
    if not emoji_font_path:
        return None

    try:
        f = ImageFont.truetype(emoji_font_path, 109, index=0)
        temp_img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
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


def _load_fonts(
    assets_dir: Path,
    font_size: int,
) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, str | None]:
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
    emoji_path    = _find_emoji_font(assets_dir)

    return normal_font, emphasis_font, emoji_path


def _tok_w(text: str, font: ImageFont.FreeTypeFont, is_emoji: bool = False, assets_dir: Path | None = None) -> int:
    if is_emoji and assets_dir:
        em_img = _get_emoji_image(text, FONT_SIZE, assets_dir)
        if em_img:
            return em_img.width
        return FONT_SIZE
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0]
    except Exception:
        return 30


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


def _clean_caption_text(text: str) -> str:
    """Filter out weird non-standard menu symbols like ☰ ≡ • ~ | and sanitize profanity."""
    clean = re.sub(r"[☰≡•|~]", "", text)
    return mask_profanity(clean.strip())


def _draw_white_card(
    img: Image.Image,
    line_token_rows: list[list[tuple]],
    emoji_str: str | None,
    normal_f: ImageFont.FreeTypeFont,
    assets_dir: Path,
    box_center_y: int,
    layout_mode: str = "pillarbox",
    custom_video_top_y: int = _VIDEO_TOP_Y,
) -> None:
    """Render ONE white rounded-rect card containing all lines onto *img*."""

    # ── Measure row widths (emojis appended individually to last row) ───────
    rows = [list(row) for row in line_token_rows]  # copy
    if emoji_str and rows:
        emoji_chars = [c for c in emoji_str if not c.isalnum() and not c.isspace()]
        if not emoji_chars and emoji_str.strip():
            emoji_chars = [emoji_str.strip()]
        for em in emoji_chars:
            rows[-1].append((em, normal_f, True))

    row_widths = [
        sum(_tok_w(t, f, ie, assets_dir) for t, f, ie in row) + WORD_GAP * max(0, len(row) - 1)
        for row in rows
    ]
    n_lines    = len(rows)
    max_row_w  = max(row_widths) if row_widths else 0

    # ── Box geometry (Yesterday's exact rounded white card layout) ─────────────
    box_inner_w = max_row_w
    box_inner_h = FONT_SIZE * n_lines + LINE_GAP * max(0, n_lines - 1)

    box_w = max(690, min(CANVAS_W - 120, box_inner_w + BOX_PAD_X * 2))
    box_inner_w = box_w - BOX_PAD_X * 2
    box_h = box_inner_h + BOX_PAD_Y * 2

    # Centered horizontally, positioned nicely above top of video frame
    box_x0 = (CANVAS_W - box_w) // 2
    if layout_mode == "face_crop":
        box_y0 = box_center_y - box_h // 2
    else:
        box_y1 = custom_video_top_y - 12
        box_y0 = box_y1 - box_h

    box_y0 = max(15, min(box_y0, CANVAS_H - box_h - 20))
    box_x1 = box_x0 + box_w
    box_y1 = box_y0 + box_h

    # ── Draw white card ───────────────────────────────────────────────────────
    card = Image.new("RGBA", img.size, (0, 0, 0, 0))
    cd   = ImageDraw.Draw(card)
    cd.rounded_rectangle([(box_x0, box_y0), (box_x1, box_y1)], radius=BOX_RADIUS, fill=BOX_BG)
    img.alpha_composite(card)


    # ── Draw text (justified / centered row by row inside 660px box) ──────────
    draw   = ImageDraw.Draw(img)
    text_y = box_y0 + BOX_PAD_Y

    for row, row_w in zip(rows, row_widths):
        # Center each line horizontally inside the box interior
        text_x = box_x0 + BOX_PAD_X + (box_inner_w - row_w) // 2
        cx = text_x
        for token_text, font, is_emoji in row:
            tw = _tok_w(token_text, font, is_emoji, assets_dir)
            ty = text_y
            if is_emoji:
                em_img = _get_emoji_image(token_text, FONT_SIZE, assets_dir)
                if em_img:
                    img.alpha_composite(em_img, (cx, ty))
                else:
                    try:
                        draw.text((cx, ty), token_text, font=font, fill=TEXT_COLOR)
                    except Exception:
                        pass
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
    Returns CANVAS_H always.
    """
    eff_include_emoji = True if include_emoji is None else include_emoji

    try:
        norm_f, emph_f, _ = _load_fonts(assets_dir, FONT_SIZE)
    except FileNotFoundError as exc:
        raise PipelineError("caption", str(exc)) from exc

    # Sanitize lines and clean weird unicode symbols like ☰ ≡ • ~
    raw_lines = [_clean_caption_text(ln) for ln in (moment.caption_lines or [])[:3]]
    raw_lines = [ln for ln in raw_lines if ln.strip()]


    # Build token rows
    line_token_rows = [_build_line_tokens(ln, norm_f, emph_f) for ln in raw_lines]
    line_token_rows = [r for r in line_token_rows if r]

    # Emoji string for the end of the last line
    emoji_str: str | None = None
    if eff_include_emoji and moment.emoji:
        emoji_str = moment.emoji.strip() or None

    # Choose box center y and video top position based on layout
    if "1_1" in layout_mode or "square" in layout_mode:
        video_top_y = (CANVAS_H - CANVAS_W) // 2  # = 280px top for 1:1 square video
    else:
        video_top_y = _VIDEO_TOP_Y  # = 370px top for 4:3 video

    if layout_mode == "face_crop":
        # Face crop is full 9:16 — treat top of video as ~370px (same as 4:3 blurred)
        # so the white card appears at same height as blurred/black canvas modes
        box_center_y = _VIDEO_TOP_Y - 70
    else:
        box_center_y = video_top_y - 70

    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))

    if line_token_rows or emoji_str:
        _draw_white_card(img, line_token_rows, emoji_str, norm_f, assets_dir, box_center_y, layout_mode=layout_mode, custom_video_top_y=video_top_y)

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
