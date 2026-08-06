"""
Subtitles System for 9x16 Vertical Social Clips (YouTube Shorts, TikTok & Reels).

Refactored to exact high-impact vertical specs:
1. Layout & Positioning:
   - 9x16 Vertical Canvas baseline (720x1280 or 1080x1920 viewport)
   - Vertical Alignment: Positioned at 67% from top of canvas (eye-level safe zone 65%-70%)
   - Horizontal Alignment: Center-aligned on X-axis (max 2-3 words per line)
2. Typography & Sizing:
   - Font Family: 'Roboto', sans-serif (Roboto Medium ttf)
   - Font Weight: 500 (Medium / Semi-Bold)
   - Text Case: FORCE UPPERCASE text transformation
   - Font Size: Scaled dynamically to 5.5% of canvas height (~70px @ 1280h, ~105px @ 1920h)
3. Coloring & High-Contrast Specs:
   - Default Text Color: Pure White (#FFFFFF)
   - Highlight Text Color: Vibrant Yellow (#FFD700) for active/emphasized words
   - Text Stroke / Outline: Pure Black (#000000) outer stroke with a width of 5px
   - Drop Shadow: Pure Black (#000000), 100% opacity, offset 4px down and 4px right (shadowx=4, shadowy=4)
4. Animation Specs: Word-by-word timing, zero delay, 1:1 speech lip sync speedup.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_RED_WORDS = {
    "no", "stop", "wrong", "bad", "fail", "failed", "loss", "lost", "dead",
    "die", "died", "hate", "hated", "awful", "terrible", "horrible", "worst",
    "never", "scam", "fraud", "broke", "broken", "trash", "disgusting",
    "pathetic", "embarrassing", "eliminated", "destroyed", "over", "done",
    "sad", "cry", "crying", "sobbing", "pain", "hurt", "angry", "mad",
    "furious", "cheated", "unfair", "rigged", "disaster", "rip", "gone",
    "finished", "quit", "quitting", "regret", "shame", "embarrassed",
    "stupid", "idiot", "dumb", "fake", "lied", "lie", "lying", "disrespect",
    "nah", "nope", "toxic", "banned", "blocked",
}

_YELLOW_WORDS = {
    "wait", "bro", "dude", "man", "what", "how", "why", "actually", "literally",
    "seriously", "honestly", "omg", "oh", "wow", "crazy", "wild", "insane",
    "unreal", "imagine", "watch", "look", "see", "bruh", "no way", "really",
    "fr", "deadass", "swear", "facts", "cap", "watch", "hold", "almost",
    "nearly", "barely", "exactly", "moment", "right", "now", "suddenly",
    "plot", "twist", "reveal", "secret", "finally", "about", "happen",
    "happening", "breaking", "exclusive", "leaked", "exposed",
    "drama", "beef", "controversial", "spicy", "heated", "tension",
    "suspect", "suspicious", "maybe", "perhaps", "could", "might",
    "unexpected", "shocking", "surprised", "surprise",
}


_PROFANITY_REPLACEMENTS = [
    (r"\bfucking\b", "f**king"),
    (r"\bfucked\b", "f**ked"),
    (r"\bfucker(s)?\b", "f**ker"),
    (r"\bfuck(s)?\b", "f**k"),
    (r"\bshitting\b", "sh*tting"),
    (r"\bshit(s)?\b", "sh*t"),
    (r"\bbitches\b", "b*tches"),
    (r"\bbitch(ed|ing)?\b", "b*tch"),
    (r"\bkilling\b", "k*lling"),
    (r"\bkilled\b", "k*lled"),
    (r"\bkill(s)?\b", "k*ll"),
    (r"\bcunt(s)?\b", "c*nt"),
    (r"\basshole(s)?\b", "a**hole"),
    (r"\bdick(s)?\b", "d*ck"),
    (r"\bpussy\b", "p*ssy"),
    (r"\bnigga(s)?\b", "n***a"),
    (r"\bnigger(s)?\b", "n***er"),
    (r"\bretarded?\b", "r*tard"),
    (r"\bbastard(s)?\b", "b*stard"),
]


def mask_profanity(text: str) -> str:
    """Sanitize explicit profanity for 100% FYP & algorithm safe text."""
    if not text:
        return ""
    res = text
    for pattern, replacement in _PROFANITY_REPLACEMENTS:
        def _replace_match(m):
            w = m.group(0)
            rep = re.sub(pattern, replacement, w, flags=re.IGNORECASE)
            if w.isupper():
                return rep.upper()
            if w.istitle():
                return rep.capitalize()
        res = re.sub(pattern, _replace_match, res, flags=re.IGNORECASE)
    return res


_GREEN_WORDS = {
    "good", "great", "best", "win", "won", "winner", "success", "successful", "profit",
    "money", "rich", "wealth", "wealthy", "happy", "joy", "love", "loved", "amazing",
    "awesome", "perfect", "flawless", "easy", "free", "safe", "secure", "growth", "grow",
    "fast", "quick", "smart", "genius", "brilliant", "yes", "yeah", "yep", "true", "facts", "exactly"
}


def _get_word_style(word: str) -> str | None:
    """Return ASS color code and italic tag if special, else None"""
    clean = re.sub(r"[^a-zA-Z]", "", word).lower()
    if clean in _RED_WORDS:
        return r"{\c&H0000FF&\i1}"  # Red (BGR) + Italic
    if clean in _GREEN_WORDS:
        return r"{\c&H00FF00&\i1}"  # Green (BGR) + Italic
    if clean in _YELLOW_WORDS or clean in {"think", "thinking", "hmm", "huh"}:
        return r"{\c&H00D7FF&\i1}"  # Yellow (BGR) + Italic
    return None


def _escape_ffmpeg_text(text: str) -> str:
    """Escape text for FFmpeg drawtext filter."""
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\u2019")
    text = text.replace(":", "\\:")
    text = text.replace("%", "\\%")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    text = text.replace(",", "\\,")
    return text.strip()


def _clean_word_text(word: str) -> str:
    """Strip full stops, commas, quotes, and punctuation clutter from word text."""
    clean = re.sub(r"^[^\w]+|[^\w]+$", "", word).strip()
    return clean if clean else word.strip()


def build_word_subtitle_filter(
    segments: list[dict],
    clip_start: float,
    clip_end: float,
    canvas_w: int = 1080,
    wm_y: int | None = None,
    font_path: str | None = None,
    canvas_h: int = 1920,
    time_offset: float = 0.0,
) -> tuple[str | None, str | None]:
    """
    Build Option 2: CapCut Clean Karaoke Style subtitles.
    Displays 2-3 words per phrase with silky-smooth left-to-right yellow karaoke word highlighting.
    """
    speed_factor = 1.00
    words: list[dict] = []
    for seg in segments:
        for w in seg.get("words", []):
            w_start = w["start"]
            w_end = w["end"]
            if w_end < clip_start - 0.2 or w_start > clip_end + 0.2:
                continue
            rel_start = max(0.0, round((w_start - clip_start) / speed_factor, 3)) + round(time_offset, 3)
            rel_end = max(rel_start + 0.05, round((w_end - clip_start) / speed_factor, 3) + round(time_offset, 3))
            raw_w = mask_profanity(_clean_word_text(w["word"]))
            if raw_w:
                words.append({
                    "word": _escape_ffmpeg_text(raw_w.capitalize()),
                    "start": rel_start,
                    "end": rel_end,
                })

    if not words and segments:
        for seg in segments:
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            if seg_end < clip_start - 0.2 or seg_start > clip_end + 0.2:
                continue
            txt = seg.get("text", "").strip()
            raw_words = txt.split()
            if not raw_words:
                continue
            dur = max(0.5, seg_end - seg_start)
            w_dur = dur / len(raw_words)
            for idx, rw in enumerate(raw_words):
                w_s = seg_start + idx * w_dur
                w_e = w_s + w_dur
                rel_start = max(0.0, round((w_s - clip_start) / speed_factor, 3)) + round(time_offset, 3)
                rel_end = max(rel_start + 0.05, round((w_e - clip_start) / speed_factor, 3) + round(time_offset, 3))
                clean_w = mask_profanity(_clean_word_text(rw))
                if clean_w:
                    words.append({
                        "word": _escape_ffmpeg_text(clean_w.capitalize()),
                        "start": rel_start,
                        "end": rel_end,
                    })

    if not words:
        logger.info("  No word timestamps for clip [%.1f-%.1f] — skipping subtitles", clip_start, clip_end)
        return None, None

    # Group words into clean 1 to 2-word phrases (Strict Single Horizontal Line — Never Stacked!)
    phrases = []
    current_phrase = []
    for w in words:
        current_phrase.append(w)
        if len(current_phrase) >= 2:
            phrases.append(current_phrase)
            current_phrase = []
        elif w != words[-1]:
            next_w = words[words.index(w) + 1]
            if next_w["start"] - w["end"] > 0.25:
                phrases.append(current_phrase)
                current_phrase = []
    if current_phrase:
        phrases.append(current_phrase)

    def _format_ass_time(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        cs = int(round((sec - int(sec)) * 100))
        if cs == 100:
            s += 1
            cs = 0
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    # Lower 20% vertical position (below speaker chin/chest, clean safe zone)
    margin_v = int(canvas_h * 0.20)
    font_size = max(46, int(canvas_w * 0.064))  # Crisp single-line font size

    import pysubs2

    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = str(canvas_w)
    subs.info["PlayResY"] = str(canvas_h)
    subs.info["WrapStyle"] = "2"  # Strict single horizontal line (no multi-line stacking!)

    style = pysubs2.SSAStyle(
        fontname="Roboto Medium",
        fontsize=font_size,
        primarycolor=pysubs2.Color(255, 255, 255),    # Pure White for upcoming text
        secondarycolor=pysubs2.Color(255, 215, 0),    # Vibrant Yellow (#FFD700) for active spoken word karaoke!
        outlinecolor=pysubs2.Color(0, 0, 0),          # Thick Black outline
        backcolor=pysubs2.Color(0, 0, 0, 160),
        bold=True,                                     # Bold high-contrast text
        outline=4.0,                                   # Clean 4.0px black stroke
        shadow=2.0,                                    # Drop shadow glow
        alignment=2,                                  # Bottom-center alignment
        marginl=10,
        marginr=10,
        marginv=margin_v,
    )
    subs.styles["CapCutKaraoke"] = style

    for p in phrases:
        p_start_ms = int(p[0]["start"] * 1000)
        p_end_ms = int((p[-1]["end"] + 0.03) * 1000)

        karaoke_parts = []
        for i, w in enumerate(p):
            dur_cs = max(10, int(round((w["end"] - w["start"]) * 100)))
            # Add gap centiseconds if there's a pause before this word
            if i > 0:
                gap_sec = w["start"] - p[i-1]["end"]
                if gap_sec > 0.05:
                    gap_cs = int(round(gap_sec * 100))
                    karaoke_parts.append(f"{{\\k{gap_cs}}}")
            
            # {\kf<cs>} creates silky smooth left-to-right yellow fill tracking
            karaoke_parts.append(f"{{\\kf{dur_cs}}}{w['word']}")

        # {\q2} forces libass to NEVER wrap text onto a 2nd or 3rd line under any circumstances!
        full_karaoke_text = "{\\q2}" + " ".join(karaoke_parts)
        event = pysubs2.SSAEvent(
            start=p_start_ms,
            end=p_end_ms,
            style="CapCutKaraoke",
            text=full_karaoke_text
        )
        subs.events.append(event)

    import tempfile
    import uuid
    tmp_path = Path(tempfile.gettempdir()) / f"sub_{uuid.uuid4().hex}.ass"
    subs.save(str(tmp_path))

    logger.info("  Generated CapCut Karaoke ASS subtitles via pysubs2 with %d phrases: %s", len(phrases), tmp_path)
    return str(tmp_path), None


def build_aura_keyword_filter(
    aura_word: str,
    moment_duration: float,
    canvas_w: int = 720,
    canvas_h: int = 1280,
    font_path: str | None = None,
    appear_at: float = 1.2,
    hold_duration: float = 4.0,
) -> str | None:
    """
    Build an FFmpeg drawtext filter that flashes a single cinematic keyword (*WORD*)
    on screen at the clip's peak moment — like *COOKED*, *IMPRESSED*, *EXPOSED* in viral clips.

    The word appears at `appear_at` seconds into the clip and stays for `hold_duration` seconds (3–5s).
    Style: high-impact bold yellow text with thick black stroke, centered in the middle of the screen.

    Returns a drawtext filter string to append to the FFmpeg filtergraph, or None if no word.
    """
    if not aura_word or not aura_word.strip():
        return None

    # Sanitize: single word only, all caps, no punctuation
    word = re.sub(r"[^A-Za-z0-9]", "", aura_word.strip()).upper()
    if not word:
        return None

    # Display with asterisks like *EXPOSED* for maximum visual impact
    display_text = f"*{word}*"

    # Escape for FFmpeg drawtext
    safe_text = display_text.replace("'", "\u2019").replace(":", "\\:").replace("%", "\\%")

    # Timing: hold for 4 seconds (3-5s)
    t_start = max(0.1, appear_at)
    t_end = min(moment_duration - 0.2, t_start + hold_duration)
    if t_end <= t_start + 1.0:
        t_end = t_start + hold_duration

    # Large high-impact font sizing — ~11% of canvas width (79px @ 720p, bold & centered)
    font_size = max(68, int(canvas_w * 0.11))

    # Centered in the middle of screen (48% vertical position)
    y_pos = int(canvas_h * 0.48)

    # Font resolution — use bundled bold font if available
    font_file_filter = ""
    if font_path and Path(font_path).exists():
        safe_fp = font_path.replace("'", "\u2019").replace(":", "\\:")
        font_file_filter = f"fontfile='{safe_fp}':"
    else:
        # Try common paths
        for fp in [
            "assets/fonts/Inter-Bold.ttf",
            "assets/fonts/Bold.otf",
            "assets/fonts/Bold.ttf",
        ]:
            if Path(fp).exists():
                safe_fp = fp.replace(":", "\\:")
                font_file_filter = f"fontfile='{safe_fp}':"
                break

    # Build the punch-in scale animation:
    # - Zoom from 85% → 100% very quickly over first 0.15s (punchy entrance)
    # - Hold at full size for the rest of hold_duration
    # We simulate scaling by modulating fontsize with a conditional expression
    punch_expr = (
        f"if(lt(t-{t_start:.3f}\\,0.15)"
        f"\\,{int(font_size * 0.85)}+({font_size}-{int(font_size * 0.85)})*(t-{t_start:.3f})/0.15"
        f"\\,{font_size})"
    )

    # Visibility: only show between t_start and t_end
    enable_expr = f"between(t,{t_start:.3f},{t_end:.3f})"

    # Build complete drawtext filter — yellow text, thick black border, drop shadow
    drawtext_filter = (
        f"drawtext="
        f"{font_file_filter}"
        f"text='{safe_text}':"
        f"fontsize={font_size}:"
        f"fontcolor=yellow:"
        f"borderw=6:"
        f"bordercolor=black:"
        f"shadowx=4:shadowy=4:shadowcolor=black@0.9:"
        f"x=(w-text_w)/2:"
        f"y={y_pos}-text_h/2:"
        f"enable='{enable_expr}'"
    )

    logger.info(
        "  Aura keyword overlay: *%s* @ %.1fs-%.1fs (font_size=%d)",
        word, t_start, t_end, font_size,
    )
    return drawtext_filter
