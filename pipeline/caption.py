"""
Step 6 — Caption image rendering with Inter Font & Strict Text Width Calibration.

Features:
- Font: Inter-Medium for normal text, Inter-Bold for ALL CAPS punch word.
- Line layout: Character-limit based wrapping (~22 chars/line). The last leftover
  line goes at the BOTTOM with 2-3 emojis prepended in front of it.
- Smooth organic background: GaussianBlur radius=12 + threshold=60 fuses all
  line pills into one crack-free continuous silhouette. Pills also overlap by 2px
  on top/bottom so there are zero gaps between adjacent lines.
- Safe Emoji Fallback: Gracefully falls back to normal font if Noto Color Emoji
  is unavailable on Linux.
"""

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .errors import PipelineError
from .score import Moment
from .subtitle import mask_profanity

logger = logging.getLogger(__name__)

# ── Layout constants ────────────────────────────────────────────────────────────
CANVAS_W: int = 720
PADDING_TOP: int = 18
PADDING_BOTTOM: int = 18
LINE_GAP: int = 8            # Tighter gap so blurred pills fuse with no cracks
WORD_GAP: int = 9

NORMAL_SIZE: int = 28
EMPHASIS_SIZE: int = 30
EMOJI_SIZE: int = 28

CHARS_PER_LINE: int = 22    # Character limit per line (triggers wrap)

STROKE_COLOR = (0, 0, 0)
STROKE_WIDTH = 0

_SYSTEM_EMOJI_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto-color-emoji/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]

_BITMAP_FALLBACK_SIZES = [28, 32, 24, 20, 40, 48, 64]


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
) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
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


def _extract_emojis(emoji_str: str) -> list[str]:
    """Extract up to 3 individual emoji characters from the emoji field."""
    chars = [c for c in emoji_str if not c.isalnum() and not c.isspace()]
    if not chars:
        chars = [emoji_str] if emoji_str.strip() else []
    return chars[:3]


def _split_text_lines(words: list[str], chars_per_line: int = CHARS_PER_LINE) -> tuple[list[list[str]], list[str]]:
    """
    Wrap words into lines by character count.
    Returns (main_lines, remainder) where remainder is the last short line
    that will have emojis prepended.
    """
    lines: list[list[str]] = []
    current: list[str] = []
    current_chars = 0

    for word in words:
        added = len(word) if not current else len(word) + 1
        if current and current_chars + added > chars_per_line:
            lines.append(current)
            current = [word]
            current_chars = len(word)
        else:
            current.append(word)
            current_chars += added

    if current:
        lines.append(current)

    if len(lines) > 1:
        remainder = lines.pop()
    else:
        remainder = lines.pop() if lines else []

    return lines, remainder


def _line_pixel_width(tokens: list[tuple]) -> int:
    if not tokens:
        return 0
    return sum(_token_size(t, f, ie)[0] for t, f, ie in tokens) + WORD_GAP * (len(tokens) - 1)


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
    Returns the height of the generated image (pixels).
    """
    eff_include_emoji = True if include_emoji is None else include_emoji

    try:
        normal_font, emphasis_font, emoji_font = _load_fonts(assets_dir)
    except FileNotFoundError as exc:
        raise PipelineError("caption", str(exc)) from exc

    full_text = mask_profanity(" ".join(moment.caption_lines))
    words = full_text.split()

    # ── Build character-limit lines + remainder ─────────────────────────────────
    main_word_lines, remainder_words = _split_text_lines(words, CHARS_PER_LINE)

    def words_to_tokens(word_list: list[str]) -> list[tuple]:
        return [
            (w, emphasis_font if _is_emphasis(w) else normal_font, False)
            for w in word_list
        ]

    line_groups: list[list[tuple]] = [words_to_tokens(wl) for wl in main_word_lines]

    # Remainder line: 2-3 emojis first, then leftover words
    remainder_line: list[tuple] = []
    if eff_include_emoji and moment.emoji:
        for em in _extract_emojis(moment.emoji):
            remainder_line.append((em, emoji_font, True))
    remainder_line.extend(words_to_tokens(remainder_words))
    if remainder_line:
        line_groups.append(remainder_line)

    line_groups = line_groups[:5]

    # ── Measure ─────────────────────────────────────────────────────────────────
    FIXED_LINE_H = 34
    LINE_STEP = 8
    bg_pad_x = 20
    bg_pad_y = 8     # Generous vertical pad so adjacent pills overlap before blur

    line_widths = [_line_pixel_width(tg) for tg in line_groups]
    n = len(line_groups)

    text_total_h = FIXED_LINE_H * n + LINE_STEP * max(0, n - 1)
    total_h = max(PADDING_TOP + bg_pad_y + text_total_h + bg_pad_y + PADDING_BOTTOM, 80)

    # ── 1. Grayscale pill mask with slight overlap so blur fuses them fully ─────
    mask_img = Image.new("L", (CANVAS_W, total_h), 0)
    mask_draw = ImageDraw.Draw(mask_img)

    y = PADDING_TOP + bg_pad_y
    for lw in line_widths:
        x = (CANVAS_W - lw) // 2
        mask_draw.rounded_rectangle(
            [
                (x - bg_pad_x, y - bg_pad_y - 3),                       # 3px extra top
                (x + lw + bg_pad_x, y + FIXED_LINE_H + bg_pad_y + 3),   # 3px extra bottom
            ],
            radius=16,
            fill=255,
        )
        y += FIXED_LINE_H + LINE_STEP

    # ── 2. Large blur + low threshold → single crack-free organic silhouette ───
    blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=12))
    smooth_mask = blurred.point(lambda p: 255 if p > 60 else 0)

    # ── 3. White background through smooth mask ─────────────────────────────────
    img = Image.new("RGBA", (CANVAS_W, total_h), (0, 0, 0, 0))
    white_layer = Image.new("RGBA", (CANVAS_W, total_h), (255, 255, 255, 255))
    img.paste(white_layer, (0, 0), smooth_mask)

    draw = ImageDraw.Draw(img)

    # ── 4. Render tokens ────────────────────────────────────────────────────────
    y = PADDING_TOP + bg_pad_y
    for lw, line_tokens in zip(line_widths, line_groups):
        x = (CANVAS_W - lw) // 2
        for text, font, is_emoji in line_tokens:
            token_w, token_h = _token_size(text, font, is_emoji)
            token_y = y + (FIXED_LINE_H - min(token_h, FIXED_LINE_H)) // 2

            if is_emoji:
                try:
                    draw.text((x, token_y), text, font=font, embedded_color=True)
                except Exception:
                    try:
                        draw.text((x, token_y), text, font=font, fill=(0, 0, 0))
                    except Exception:
                        logger.warning("Emoji render failed for '%s'", text)
            else:
                draw.text((x, token_y), text, font=font, fill=(0, 0, 0), stroke_width=0)
            x += token_w + WORD_GAP

        y += FIXED_LINE_H + LINE_STEP

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    logger.info(
        "  Caption %02d: %s (%dpx tall, %d lines, mode=%s)",
        moment.index, output_path.name, total_h, len(line_groups), layout_mode,
    )
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
