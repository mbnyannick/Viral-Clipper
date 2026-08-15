"""Small text helpers used by the clip pipeline."""

from __future__ import annotations

import re

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
    """Sanitize explicit profanity for safer on-screen text."""
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
            return rep

        res = re.sub(pattern, _replace_match, res, flags=re.IGNORECASE)
    return res


# Regex matching emojis across standard Unicode ranges
_EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u27ff\u2b50\u231a-\u23f3\u2934\u2935\u25aa-\u25fe\u200d\ufe0f]",
    flags=re.UNICODE,
)

_DANGLING_WORDS = {
    "and", "the", "with", "to", "of", "in", "a", "an", "or", "is", "at", "for", "on", "by", "from", "that", "this", "but", "so"
}


def format_seo_title(
    title: str | Sequence[str],
    streamer: str = "",
    default_emoji: str = "🔥😂💀",
    max_chars: int = 55,
) -> str:
    """
    Format a clean, high-CTR, SEO-optimized title for YouTube Shorts, Reels, and TikTok.

    Guarantees:
    - NO slashes ('/', '\\', '//') or pipe symbols ('|').
    - NO raw subtitle dumps or multi-line separators.
    - NO embedded hashtags (#Shorts, #Viral).
    - NO ellipses ('...', '…') or trailing punctuation junk.
    - NO surrounding quotation marks.
    - Clean SEO length (30-55 chars) breaking cleanly at word boundaries.
    - Profanity sanitized via mask_profanity.
    - Clean 2-3 high-energy emojis at the end.
    """
    from collections.abc import Sequence as SeqType
    if isinstance(title, (list, tuple, SeqType)) and not isinstance(title, str):
        raw = " ".join([str(t).strip() for t in title if t and str(t).strip()])
    else:
        raw = str(title or "").strip()

    # Remove slashes, pipes, backslashes, underscores used as separators
    raw = re.sub(r"[\/\\|]+", " ", raw)
    raw = re.sub(r"_+", " ", raw)

    # Strip surrounding quotes
    raw = raw.strip("\"'“”‘’«»`")

    # Remove hashtags from title
    raw = re.sub(r"#\w+", "", raw)

    # Extract existing emojis (support 2-3 emojis)
    emojis = _EMOJI_PATTERN.findall(raw)
    if emojis:
        clean_emojis = "".join(emojis[:3])
    else:
        clean_emojis = default_emoji if default_emoji else "🔥😂"

    # Strip emojis from text body to handle length & phrasing cleanly
    text = _EMOJI_PATTERN.sub("", raw)

    # Clean double spaces, punctuation anomalies, ellipses
    text = re.sub(r"\.{2,}|…", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[:;,\-–—]+$", "", text).strip()

    # Profanity mask
    text = mask_profanity(text)

    # If empty after cleaning, use safe fallback
    if not text:
        clean_streamer = (streamer or "Streamer").strip()
        text = f"{clean_streamer} Craziest Stream Moment"

    # Intelligent length boundary truncation
    words = text.split()
    if len(text) > max_chars:
        accum = []
        curr_len = 0
        for w in words:
            if curr_len + len(w) + (1 if accum else 0) > max_chars:
                break
            accum.append(w)
            curr_len += len(w) + 1

        # Strip trailing dangling words
        while accum and accum[-1].lower() in _DANGLING_WORDS:
            accum.pop()

        if accum:
            text = " ".join(accum)
        else:
            text = words[0] if words else f"{streamer} Highlight"

    # Clean any trailing punctuation leftover
    text = text.rstrip(" .,!?:;\\-–—")

    # Ensure clean final output ending with 2-3 emojis
    if clean_emojis:
        return f"{text} {clean_emojis}"
    return text


def generate_rich_hashtags(
    streamer: str = "",
    topic: str = "",
    aura_word: str = "",
    existing_hashtags: str = "",
    min_count: int = 12,
    max_count: int = 18,
) -> str:
    """
    Generate a comprehensive, high-volume block of 12-18 relevant hashtags.
    Combines creator tags, topic/action tags, viral trends, and platform discovery tags.
    """
    clean_streamer = re.sub(r"[^\w]", "", streamer.strip()) if streamer else "Streamer"

    tags: list[str] = []
    seen: set[str] = set()

    def _add(t: str):
        if not t:
            return
        tag = t if t.startswith("#") else f"#{t}"
        tag = re.sub(r"[^\w#]", "", tag)
        if len(tag) > 1 and tag.lower() not in seen:
            seen.add(tag.lower())
            tags.append(tag)

    # 1. Existing parsed hashtags if provided
    if existing_hashtags:
        for raw in existing_hashtags.split():
            if raw.startswith("#"):
                _add(raw)

    # 2. Creator specific hashtags
    if clean_streamer and clean_streamer.lower() != "streamer":
        _add(f"#{clean_streamer}")
        _add(f"#{clean_streamer}Clips")
        _add(f"#{clean_streamer}Live")
        _add(f"#{clean_streamer}Highlights")
        _add(f"#{clean_streamer}VOD")

    # 3. Topic & Aura word tags
    if aura_word:
        clean_aura = re.sub(r"[^\w]", "", aura_word).capitalize()
        if clean_aura:
            _add(f"#{clean_aura}")
            _add(f"#{clean_aura}Moment")

    if topic and isinstance(topic, str):
        # Extract meaningful topic words
        clean_words = [re.sub(r"[^\w]", "", w).capitalize() for w in topic.split() if len(w) > 3 and not w.startswith("#")]
        for kw in clean_words[:3]:
            if kw and kw.lower() not in ("streamer", "moment", "highlights", "craziest", "video", "shorts"):
                _add(f"#{kw}")

    # 4. Standard high-velocity viral & platform tags
    viral_pool = [
        "#Shorts",
        "#YouTubeShorts",
        "#TikTokViral",
        "#ReelsTrends",
        "#FYP",
        "#ForYouPage",
        "#ViralClips",
        "#StreamerHighlights",
        "#GamingClips",
        "#TwitchClips",
        "#KickClips",
        "#FunnyMoments",
        "#Trending",
        "#ExplorePage",
        "#ViralVideo",
        "#Reaction",
        "#BestMoments",
        "#LiveStreamMoments",
    ]
    for vt in viral_pool:
        _add(vt)
        if len(tags) >= max_count:
            break

    return " ".join(tags[:max_count])


