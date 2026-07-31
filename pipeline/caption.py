"""
Step 6 — Caption image rendering with Inter Font & Strict Text Width Calibration.

Features:
- Font: Inter-Medium for normal text, Inter-Bold for ALL CAPS punch word.
- Calibration: MAX_LINE_WIDTH = 800px (140px side margins) on a 1080px canvas.
  Text can NEVER exceed 800px width or touch screen edges.
- Word Wrapping: Automatically wraps long captions into up to 5 short, well-balanced
  centered lines (max 4-5 words per line).
- Safe Emoji Fallback: Gracefully falls back to normal font if Noto Color Emoji is unavailable in Linux.
"""

import logging
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .errors import PipelineError
from .score import Moment
from .subtitle import mask_profanity

logger = logging.getLogger(__name__)

# ── Layout constants ────────────────────────────────────────────────────────────
CANVAS_W: int = 720
MAX_LINE_WIDTH: int = 580     # Expanded boundary for bold, high-impact text
PADDING_TOP: int = 15
PADDING_BOTTOM: int = 15
LINE_GAP: int = 8
WORD_GAP: int = 10

NORMAL_SIZE: int = 36        # Sleek, compact font size for normal text
EMPHASIS_SIZE: int = 40      # Compact bold font size for punch words
EMOJI_SIZE: int = 40         # Emoji size

TEXT_COLOR = (255, 255, 255)     # Pure white text
STROKE_COLOR = (0, 0, 0)         # Stroke disabled
STROKE_WIDTH = 0                 # No black stroke outline
BG_COLOR = (0, 0, 0, 0)          # 100% transparent background

_SYSTEM_EMOJI_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto-color-emoji/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]

_BITMAP_FALLBACK_SIZES = [40, 32, 48, 64, 20, 96, 109, 128, 160]


def _is_emphasis(word: str) -> bool:
    """Return True if *word* is the ALL CAPS punch word."""
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
) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    """Load Inter Medium for normal text and Inter Bold for emphasis."""
    inter_bold = assets_dir / "fonts" / "Inter-Bold.ttf"
    bold_otf = assets_dir / "fonts" / "Bold.otf"
    bold_ttf = assets_dir / "fonts" / "Bold.ttf"
    medium_otf = assets_dir / "fonts" / "Medium.otf"

    bold_path = inter_bold if inter_bold.exists() else (bold_otf if bold_otf.exists() else bold_ttf)
    medium_path = medium_otf if medium_otf.exists() else bold_path

    if not bold_path.exists():
        raise FileNotFoundError(f"Font file missing at {bold_path}")

    normal_font = ImageFont.truetype(str(medium_path), NORMAL_SIZE)
    emphasis_font = ImageFont.truetype(str(bold_path), EMPHASIS_SIZE)

    emoji_font = None
    emoji_path = _find_emoji_font(assets_dir)
    if emoji_path:
        try:
            emoji_font = ImageFont.truetype(emoji_path, EMOJI_SIZE, index=0)
        except OSError:
            for sz in _BITMAP_FALLBACK_SIZES:
                try:
                    emoji_font = ImageFont.truetype(emoji_path, sz, index=0)
                    break
                except OSError:
                    continue

    if emoji_font is None:
        emoji_font = normal_font

    return normal_font, emphasis_font, emoji_font


def _token_size(text: str, font: ImageFont.FreeTypeFont, is_emoji: bool = False) -> tuple[int, int]:
    try:
        bb = font.getbbox(text)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if is_emoji:
            w = max(w, EMOJI_SIZE)
            h = max(h, EMOJI_SIZE)
        return w, h
    except Exception:
        return (EMOJI_SIZE, EMOJI_SIZE) if is_emoji else (30, 30)


def _token_bbox(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    return _token_size(text, font, is_emoji=False)


def _wrap_tokens(
    tokens: list[tuple[str, ImageFont.FreeTypeFont, bool]],
    max_w: int,
    max_words_per_line: int = 4,
) -> list[list[tuple[str, ImageFont.FreeTypeFont, bool]]]:
    """
    Wrap tokens into lines such that no line exceeds max_w or max_words_per_line.
    """
    lines: list[list[tuple[str, ImageFont.FreeTypeFont, bool]]] = []
    current_line: list[tuple[str, ImageFont.FreeTypeFont, bool]] = []
    current_w = 0

    for token in tokens:
        text, font, is_emoji = token
        tw, _ = _token_size(text, font, is_emoji)

        word_count = sum(1 for _, _, ie in current_line if not ie)
        added_w = tw if not current_line else WORD_GAP + tw

        if current_line and ((current_w + added_w > max_w) or (word_count >= max_words_per_line and not is_emoji)):
            lines.append(current_line)
            current_line = [token]
            current_w = tw
        else:
            current_line.append(token)
            current_w += added_w

    if current_line:
        lines.append(current_line)

    return lines[:5]


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
    - blurred_frame (or default): Solid white card container with rounded corners and Inter-Bold black text.
    - face_crop: Clean white text with 3px black stroke, transparent background (no card box).
    Returns the height of the generated image (pixels).
    """
    is_face_crop = layout_mode == "face_crop"
    
    eff_text_color = text_color if text_color is not None else (0, 0, 0)
    eff_stroke_color = stroke_color if stroke_color is not None else STROKE_COLOR
    eff_stroke_width = stroke_width if stroke_width is not None else STROKE_WIDTH
    eff_include_emoji = True if include_emoji is None else include_emoji

    try:
        normal_font, emphasis_font, emoji_font = _load_fonts(assets_dir)
    except FileNotFoundError as exc:
        raise PipelineError("caption", str(exc)) from exc

    tokens: list[tuple[str, ImageFont.FreeTypeFont, bool]] = []
    full_text = mask_profanity(" ".join(moment.caption_lines))

    for word in full_text.split():
        # Gen Z style: Semi-bold for normal words, Heavy Inter-Bold for ALL CAPS punch words
        font = emphasis_font if _is_emphasis(word) else normal_font
        tokens.append((word, font, False))

    if moment.emoji and eff_include_emoji:
        tokens.append((moment.emoji, emoji_font, True))

    line_tokens = _wrap_tokens(tokens, max_w=MAX_LINE_WIDTH, max_words_per_line=4)

    FIXED_LINE_H = 40
    LINE_GAP = 10
    pad_x = 32
    pad_y = 20

    line_dims: list[tuple[int, int]] = []
    for line in line_tokens:
        w = sum(
            _token_size(t, f, ie)[0]
            for t, f, ie in line
        ) + WORD_GAP * (len(line) - 1)
        line_dims.append((w, FIXED_LINE_H))

    text_total_h = sum(h for _, h in line_dims) + LINE_GAP * max(0, len(line_dims) - 1)
    max_line_w = max(w for w, _ in line_dims)

    card_w = max_line_w + (pad_x * 2)
    card_h = text_total_h + (pad_y * 2)

    total_h = card_h + (PADDING_TOP * 2)
    total_h = max(total_h, 120)

    img = Image.new("RGBA", (CANVAS_W, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    card_left = (CANVAS_W - card_w) // 2
    card_top = PADDING_TOP

    # 1. Draw ONE single clean rectangular white card container box (radius=24)
    draw.rounded_rectangle(
        [(card_left, card_top), (card_left + card_w, card_top + card_h)],
        radius=24,
        fill=(255, 255, 255)
    )

    # 2. Render all text lines centered inside the single white card box
    y = card_top + pad_y
    for (line_w, line_h), line in zip(line_dims, line_tokens):
        x = (CANVAS_W - line_w) // 2   # strictly centered horizontally

        for text, font, is_emoji in line:
            token_w, token_h = _token_size(text, font, is_emoji)
            token_y = y + (line_h - min(token_h, 36)) // 2

            if is_emoji:
                try:
                    draw.text((x, token_y), text, font=font, embedded_color=True)
                except Exception:
                    try:
                        draw.text((x, token_y), text, font=font, fill=(0, 0, 0))
                    except Exception:
                        logger.warning("Emoji render failed for '%s'", text)
            else:
                draw.text(
                    (x, token_y),
                    text,
                    font=font,
                    fill=(0, 0, 0),  # Crisp solid black text on white container
                    stroke_width=0,
                )
            x += token_w + WORD_GAP

        y += line_h + LINE_GAP

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    logger.info("  Caption %02d: %s (%dpx tall, %d lines, mode=%s)", moment.index, output_path.name, total_h, len(line_tokens), layout_mode)
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
