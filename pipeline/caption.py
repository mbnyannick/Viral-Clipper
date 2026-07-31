"""
Step 6 — Caption image rendering.

Non-face-crop:
  Full-canvas (720×1280) transparent PNG.
  ONE single white rounded-rectangle containing all caption lines (1-3),
  positioned in the lower-center of the 4:3 video zone (overlaid on the footage).
  Black bold text, centered per line. Emoji at the end of the last line.
  Composited with enable='between(t,0,5)' so it appears for 5 seconds then vanishes.

face_crop:
  Small transparent PNG, white bold text + black stroke.
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
FONT_SIZE: int = 38          # Bold text size — readable on video
LINE_GAP: int = 10           # Vertical gap between lines inside the box
WORD_GAP: int = 7

# ── Box style (single white card) ────────────────────────────────────────────
BOX_BG       = (255, 255, 255, 242)  # near-opaque white
BOX_RADIUS   = 14
BOX_PAD_X    = 28            # horizontal padding inside box
BOX_PAD_Y    = 18            # vertical padding inside box
TEXT_COLOR   = (15, 15, 15)  # near-black text

# ── Box vertical anchor: lower-center of the 4:3 video zone ──────────────────
# Places the center of the white card at ~72% down the video (chest area)
_BOX_CENTER_Y: int = _VIDEO_TOP_Y + int((_VIDEO_BOT_Y - _VIDEO_TOP_Y) * 0.72)

_SYSTEM_EMOJI_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto-color-emoji/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]
_BITMAP_FALLBACK_SIZES = [38, 40, 36, 32, 44, 48]


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


def _tok_w(text: str, font: ImageFont.FreeTypeFont, is_emoji: bool) -> int:
    try:
        bb = font.getbbox(text)
        w  = bb[2] - bb[0]
        return max(w, FONT_SIZE) if is_emoji else w
    except Exception:
        return FONT_SIZE if is_emoji else 30


def _extract_emoji(emoji_str: str) -> str | None:
    """Return the first emoji character from the field, or None."""
    chars = [c for c in emoji_str if not c.isalnum() and not c.isspace()]
    return chars[0] if chars else (emoji_str.strip() or None)


def _build_line_tokens(
    line_text: str,
    normal_f: ImageFont.FreeTypeFont,
    emph_f:   ImageFont.FreeTypeFont,
) -> list[tuple]:
    return [
        (w, emph_f if _is_emphasis(w) else normal_f, False)
        for w in line_text.split()
    ]


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

    Non-face-crop: Full 720×1280 transparent PNG with ONE white card containing
    all caption lines, positioned on the video. Returns CANVAS_H (1280).

    face_crop: Small transparent PNG, white text + stroke. Returns actual height.
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
        norm_f, emph_f, emoji_f = _load_fonts(assets_dir, FONT_SIZE)
    except FileNotFoundError as exc:
        raise PipelineError("caption", str(exc)) from exc

    # Sanitize and split into up to 3 lines
    raw_lines = [mask_profanity(ln) for ln in (moment.caption_lines or [])[:3]]
    raw_lines = [ln for ln in raw_lines if ln.strip()]

    # Build token rows (one list per line)
    line_token_rows: list[list[tuple]] = []
    for i, ln in enumerate(raw_lines):
        toks = _build_line_tokens(ln, norm_f, emph_f)
        # Append the emoji to the LAST line only
        if i == len(raw_lines) - 1 and eff_include_emoji and moment.emoji:
            em = _extract_emoji(moment.emoji)
            if em:
                toks.append((em, emoji_f, True))
        line_token_rows.append(toks)

    if not line_token_rows:
        # Nothing to render — save blank canvas
        Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0)).save(str(output_path))
        return CANVAS_H

    # ── Measure each line width ───────────────────────────────────────────────
    row_widths = [
        sum(_tok_w(t, f, ie) for t, f, ie in row) + WORD_GAP * max(0, len(row) - 1)
        for row in line_token_rows
    ]
    max_row_w = max(row_widths)
    n_lines   = len(line_token_rows)

    # ── Compute box dimensions ────────────────────────────────────────────────
    box_inner_w = max_row_w
    box_inner_h = FONT_SIZE * n_lines + LINE_GAP * max(0, n_lines - 1)
    box_w = box_inner_w + BOX_PAD_X * 2
    box_h = box_inner_h + BOX_PAD_Y * 2

    # Cap box width to canvas (with 20px margin each side)
    box_w = min(box_w, CANVAS_W - 40)

    # ── Position: centered horizontally, anchored at _BOX_CENTER_Y ───────────
    box_x0 = (CANVAS_W - box_w) // 2
    box_y0 = _BOX_CENTER_Y - box_h // 2
    # Keep inside the 4:3 video zone with margin
    box_y0 = max(_VIDEO_TOP_Y + 20, min(box_y0, _VIDEO_BOT_Y - box_h - 20))
    box_x1, box_y1 = box_x0 + box_w, box_y0 + box_h

    # ── Render full-canvas transparent image ──────────────────────────────────
    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))

    # White card
    card = Image.new("RGBA", img.size, (0, 0, 0, 0))
    cd   = ImageDraw.Draw(card)
    cd.rounded_rectangle([(box_x0, box_y0), (box_x1, box_y1)], radius=BOX_RADIUS, fill=BOX_BG)
    img.alpha_composite(card)

    # Text
    draw  = ImageDraw.Draw(img)
    text_y = box_y0 + BOX_PAD_Y

    for row, row_w in zip(line_token_rows, row_widths):
        # Center each line horizontally inside the box
        text_x = box_x0 + BOX_PAD_X + (box_inner_w - row_w) // 2
        cx = text_x
        for token_text, font, is_emoji in row:
            tw = _tok_w(token_text, font, is_emoji)
            if is_emoji:
                try:
                    draw.text((cx, text_y), token_text, font=font, embedded_color=True)
                except Exception:
                    try:
                        draw.text((cx, text_y), token_text, font=font, fill=TEXT_COLOR)
                    except Exception:
                        pass
            else:
                draw.text((cx, text_y), token_text, font=font, fill=TEXT_COLOR)
            cx += tw + WORD_GAP
        text_y += FONT_SIZE + LINE_GAP

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    logger.info(
        "  Caption %02d: %s (white card y=%d–%d, %d lines, mode=%s)",
        moment.index, output_path.name, box_y0, box_y1, n_lines, layout_mode,
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
    """face_crop: small transparent PNG, white text + stroke."""
    FC_SIZE    = 36
    FC_WORD_GAP = 8
    FC_LINE_GAP = 6
    FC_MAX_W   = 500

    try:
        norm_f, emph_f, emoji_f = _load_fonts(assets_dir, FC_SIZE)
    except FileNotFoundError as exc:
        raise PipelineError("caption", str(exc)) from exc

    eff_text   = text_color   if text_color   is not None else (255, 255, 255)
    eff_stroke = stroke_color if stroke_color is not None else (0, 0, 0)
    eff_sw     = stroke_width if stroke_width is not None else 3

    clean_text = mask_profanity(" ".join(moment.caption_lines or []))
    tokens: list[tuple] = []
    for word in clean_text.split():
        tokens.append((word, emph_f if _is_emphasis(word) else norm_f, False))
    if include_emoji and moment.emoji:
        em = _extract_emoji(moment.emoji)
        if em:
            tokens.append((em, emoji_f, True))

    lines: list[list[tuple]] = []
    cur: list[tuple] = []
    cur_w = 0
    for tok in tokens:
        t, f, ie = tok
        tw = _tok_w(t, f, ie)
        added = tw if not cur else FC_WORD_GAP + tw
        if cur and cur_w + added > FC_MAX_W:
            lines.append(cur)
            cur  = [tok]
            cur_w = tw
        else:
            cur.append(tok)
            cur_w += added
    if cur:
        lines.append(cur)
    lines = lines[:4]

    LINE_H  = FC_SIZE + 4
    total_h = max(60, LINE_H * len(lines) + FC_LINE_GAP * max(0, len(lines) - 1) + 20)

    img  = Image.new("RGBA", (CANVAS_W, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = 10
    for line in lines:
        lw = sum(_tok_w(t, f, ie) for t, f, ie in line) + FC_WORD_GAP * (len(line) - 1)
        x  = (CANVAS_W - lw) // 2
        for token_text, font, is_emoji in line:
            tw = _tok_w(token_text, font, is_emoji)
            if is_emoji:
                try:
                    draw.text((x, y), token_text, font=font, embedded_color=True)
                except Exception:
                    pass
            else:
                draw.text((x, y), token_text, font=font, fill=eff_text,
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
