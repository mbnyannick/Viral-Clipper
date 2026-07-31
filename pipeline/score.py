"""
Step 4 — Moment scoring & master caption generation with Streamer Name awareness.

Sends the full merged transcript to DeepSeek with a structured prompt that returns
a JSON array of the top N moments. Instructs DeepSeek to incorporate the main streamer's
name ("Duke Dennis", "n3on", etc.) for primary actions, and "Bro" for secondary/guest speakers,
plus mandatory viral emoji selection.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

from .errors import PipelineError

logger = logging.getLogger(__name__)


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Moment:
    """One identified highlight moment, ready for cutting and captioning."""
    index: int
    start: float          # seconds from video start
    end: float            # seconds from video start
    caption_lines: list[str]   # 2–4 strings; CAPS words = emphasis
    emoji: str
    score: int = 90       # Viral potential score (0-100)
    reasoning: str = ""   # 1-sentence explanation of why it's viral
    title: str = ""       # YouTube Shorts title (Max 38 characters)
    bgm_track: str = "none" # "hype", "suspense", "funny", "sad", or "none"
    sfx_events: list[dict] = field(default_factory=list) # list of {"type": "boom"|"whoosh"|"ding", "time_offset": float}

    @property
    def duration(self) -> float:
        return self.end - self.start


def clean_streamer_name(raw_name: str) -> str:
    """
    Clean channel/streamer names by stripping suffixes like Live, VODs, Clips, Shorts, Official.
    Examples:
        'Kai Cenat Live' -> 'Kai'
        'N3on Live' -> 'N3on'
        'Adin Live' -> 'Adin'
        'Speed Live' -> 'Speed'
        'Caleb Hammer Official' -> 'Caleb Hammer'
    """
    if not raw_name:
        return "Streamer"

    suffixes = [
        r"\bLive\b", r"\bVODs?\b", r"\bClips?\b", r"\bShorts?\b", r"\bOfficial\b",
        r"\bGaming\b", r"\bReacts?\b", r"\bTV\b", r"\bPodcast\b", r"\bHighlights?\b",
        r"\bExtra\b", r"\bChannel\b", r"\bShow\b", r"\bDaily\b", r"\bReels?\b"
    ]
    cleaned = raw_name
    for s in suffixes:
        cleaned = re.sub(s, "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    words = cleaned.split()
    if words:
        first_word = words[0]
        known_famous = {"kai", "n3on", "neon", "adin", "speed", "ishowspeed", "xqc", "bruce", "fanum", "duke", "agent", "ray", "clix", "tarik", "pokimane", "hassan", "hasan", "sammy"}
        if first_word.lower() in known_famous:
            return first_word.capitalize() if first_word.lower() != "n3on" else "N3on"

    return cleaned if cleaned else raw_name


# ── Prompt ─────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an elite viral clip curator and storytelling master for TikTok, Instagram Reels, and YouTube Shorts.
Analyze the transcript deeply to identify the HIGHEST VIRAL POTENTIAL moments — extreme reactions, hilarious jokes, heated arguments, unexpected plot twists, or internet lore climaxes.

STREAMER & VIDEO CONTEXT:
- Main Streamer / Creator: {streamer}
- Video Title: {video_title}

SPEAKER & CAPTION IDENTIFICATION RULES (CRITICAL):
1. TRUE IDENTITY EXTRACTION (CRITICAL): The Channel Name provided ({streamer}) may just be a brand, aggregator, or network (e.g., 'TVU Networks', 'Daily Clips'). DO NOT blindly use the Channel Name as a person's name! You MUST analyze the conversational context, the Video Title ({video_title}), and your knowledge of famous internet personalities to deduce the TRUE REAL NAMES of the humans speaking.
2. USE CLEAN SHORT NAMES: Once you deduce the true identities, use ONLY their clean short names (e.g., "Kai", "N3on", "Adin", "Speed", "xQc"). NEVER use channel suffixes, brand names, or aggregator network names in the text overlay!
3. NEVER USE CHANNEL NAME AS SPEAKER: If the Channel Name is a brand or network, NEVER write "{streamer} Says X". Figure out exactly who the human speaking is, and use their actual name (e.g., "Ray Calls Out Kai", "Alabama Barker Reacts To Adin").
4. MANDATORY EMOJI: Every single moment MUST include a vibrant, high-energy emoji character in the "emoji" field (e.g. 🔥, 💀, 😱, 😂, 🚨, 🤡, 📈, 🤯). Never leave emoji empty or missing!
5. EVEN TIMELINE DISTRIBUTION: Spread the {top_n} moments across the full timeline of the video (beginning, middle, and end).
6. Number of lines: 2 to 4 short lines per caption.
7. Max words per line: 3 to 5 words per line maximum!
8. Title Case & ALL CAPS Punch Word: Every word starts with a Capital Letter. Exactly ONE key punch word per entire caption MUST be in ALL CAPS (e.g., MESSY, WILD, WRONG, COOKED, CRAZY, UNMATCHED, UNREAL, EXPOSED).
9. DO NOT put the emoji inside caption_lines — it belongs in the "emoji" field only.

NARRATIVE ARCS & CLIP DURATION RULES:
10. PEAK VIRAL SELECTION: Choose only high-energy, emotionally intense, or genuinely hilarious moments. Avoid picking plain filler dialogue.
11. COMPLETE CONVERSATION ARCS: Always capture the FULL context. Start 3–5 seconds BEFORE the dialogue or setup begins so viewers understand what is happening.
12. NO MID-SENTENCE OR PUNCHLINE CUTS: Always extend the clip 4–6 seconds AFTER the punchline, reaction, or laugh lands. NEVER cut a speaker mid-sentence or cut off the conclusion of a story!
13. CLIP DURATION: Each moment MUST be targeted to the requested duration.

EXAMPLE CAPTION LAYOUTS WITH CLEAN SHORT NAMES:
  ["Kai Confronts N3on", "Streamer University Debate", "Things Got MESSY"]
  ["Ray Calls Out Kai", "He Was Left Speechless", "The Argument Is COOKED"]
  ["Alabama Barker & N3on", "Unfiltered Stream Moment", "Reaction Was CRAZY"]

Return ONLY a valid JSON array with exactly {top_n} objects. No markdown, no explanation, just raw JSON.

Each object must have exactly these fields:
  "start"         — float, seconds from start of video
  "end"           — float, seconds from start of video
  "caption_lines" — array of 2–4 strings (Title Case with real speaker names, max 5 words/line, ONE word ALL CAPS total across all lines)
  "emoji"         — single relevant emoji character (e.g. 🔥, 💀, 😱, 😂, 🚨, 🤡)
  "score"         — integer, Viral potential score from 0 to 100
  "reasoning"     — string, a punchy 1-sentence explanation of exactly WHY this moment is highly viral
  "title"         — string, A highly-clickable YouTube Shorts title (MAXIMUM 38 CHARACTERS). It MUST be 38 characters or less or it will get cut off on mobile!
  "bgm_track"     — string, Background music vibe to play throughout the clip. MUST be one of: "hype", "suspense", "funny", "sad", or "none".
  "sfx_events"    — array of dicts, Sound effects to play at specific moments. Each dict MUST have "type" (one of "boom", "whoosh", "ding") and "time_offset" (float, seconds relative to the START of the clip, e.g. 2.5 means 2.5s after the clip begins).\
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_transcript(segments: list[dict]) -> str:
    """Convert segments list to a compact timestamped plain-text block."""
    lines = [f"[{seg['start']:.1f}s] {seg['text'].strip()}" for seg in segments if seg.get("text", "").strip()]
    return "\n".join(lines[:150])


def _parse_moments(raw: str) -> list[dict]:
    """Parse the LLM's response into a list of dicts."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        inner = parts[1]
        if inner.startswith("json"):
            inner = inner[4:]
        text = inner.strip()
    return json.loads(text)


# ── Public API ─────────────────────────────────────────────────────────────────

async def score_moments(
    segments: list[dict],
    api_key: str,
    top_n: int = 10,
    model: str = "deepseek-v4-flash",
    streamer: str = "Streamer",
    video_title: str = "",
    campaign_brief: str = "",
    target_duration: str = "auto",
) -> list[Moment]:
    """Score transcript and return ranked list of Moment objects with optional Campaign Brief directives and target clip duration."""
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    display_streamer = clean_streamer_name(streamer)

    dur_rules = {
        "0_30": "Each moment MUST be between 10 and 30 seconds (end - start MUST be >= 10 and <= 30).",
        "15_30": "Each moment MUST be between 15 and 30 seconds (end - start MUST be >= 15 and <= 30).",
        "30_60": "Each moment MUST be between 30 and 60 seconds (end - start MUST be >= 30 and <= 60).",
        "60_120": "Each moment MUST be between 60 and 120 seconds (1 to 2 minutes) (end - start MUST be >= 60 and <= 120).",
    }
    dur_text = dur_rules.get(target_duration, "Each moment MUST be between 25 and 60 seconds (end - start MUST be >= 25 and <= 60).")

    system = _SYSTEM_PROMPT.format(
        top_n=top_n,
        streamer=display_streamer,
        video_title=video_title,
    )
    # Replace default duration line 12 with selected target duration
    system = system.replace(
        "12. CLIP DURATION: Each moment MUST be between 25 and 60 seconds (end - start MUST be >= 25 and <= 60).",
        f"12. CLIP DURATION: {dur_text}"
    )

    if campaign_brief:
        system += (
            f"\n\nCAMPAIGN BRIEF & COMPLIANCE RULES:\n{campaign_brief}\n\n"
            "CRITICAL COMPLIANCE DIRECTIVE:\n"
            "You MUST ensure all candidate clips strictly follow the Campaign Brief & Rules above.\n"
            "Filter out any moments that violate campaign rules or lack required creator/topic focus.\n"
            "Ensure caption_lines reflect requested creator names, branding, or topic focus."
        )

    transcript_text = _format_transcript(segments)

    models_to_try = [model, "deepseek-chat", "deepseek-reasoner"]
    seen = set()
    models_to_try = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

    last_exc = None
    for m in models_to_try:
        for attempt in range(1, 3):
            try:
                response = await client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": f"Transcript:\n\n{transcript_text}",
                        },
                    ],
                    temperature=0.3,
                    max_tokens=min(16384, max(4096, top_n * 120)),
                )
                raw = response.choices[0].message.content or ""
                data = _parse_moments(raw)

                moments = [
                    Moment(
                        index=i,
                        start=float(item["start"]),
                        end=min(float(item["end"]), float(item["start"]) + 90.0),
                        caption_lines=list(item["caption_lines"]),
                        emoji=item.get("emoji", "🔥"),
                        score=int(item.get("score", 90)),
                        reasoning=item.get("reasoning", "High energy moment."),
                        title=item.get("title", " ".join(item.get("caption_lines", []))[:38]),
                        bgm_track=item.get("bgm_track", "hype"),
                        sfx_events=list(item.get("sfx_events", [])),
                    )
                    for i, item in enumerate(data)
                ]
                logger.info("Scored %d moments via model '%s'", len(moments), m)
                return moments

            except json.JSONDecodeError as exc:
                last_exc = exc
                logger.warning("Model %s attempt %d — JSON parse failed: %s", m, attempt, exc)
            except Exception as exc:
                if last_exc is None:
                    last_exc = exc
                logger.warning("Model %s attempt %d — API request failed: %s", m, attempt, exc)
                break

    logger.warning("All LLM model attempts failed (%s)", last_exc)
    if last_exc is not None:
        raise PipelineError("score", f"LLM scoring failed after retries: {last_exc}")
    return _generate_fallback_moments(segments, top_n, display_streamer)


def _generate_fallback_moments(segments: list[dict], top_n: int = 10, streamer: str = "Streamer") -> list[Moment]:
    """Generate evenly spaced timeline moments as a reliable fallback if LLM scoring fails."""
    if not segments:
        return [Moment(index=0, start=0.0, end=30.0, caption_lines=[f"{streamer} Highlights", "Best Moments", "VIRAL"], emoji="🔥")]
    max_time = max(s.get("end", 30.0) for s in segments)
    step = max(30.0, max_time / max(1, top_n))
    fallback = []
    for i in range(top_n):
        t_start = i * step
        if t_start + 30.0 > max_time and t_start > 0:
            break
        fallback.append(
            Moment(
                index=i,
                start=round(t_start, 1),
                end=round(min(max_time, t_start + 30.0), 1),
                caption_lines=[f"{streamer} Stream", "Best Moments", "UNREAL"],
                emoji="🔥",
            )
        )
    return fallback if fallback else [Moment(index=0, start=0.0, end=30.0, caption_lines=[f"{streamer} Stream", "Best Moments", "UNREAL"], emoji="🔥")]


# ── Master caption ──────────────────────────────────────────────────────────────

_MASTER_CAPTION_PROMPT = """\
You are an expert social media manager for top live streamers.
Write ONE master caption + hashtag block to post alongside all the clips on social media.

Rules:
- Open with a punchy, engaging hook sentence mentioning {streamer} by name.
- Summarise what made this stream special — mention key moments or guests by name.
- End with a question or call-to-action (e.g. "Which moment was your favorite? 👇").
- Then on a new line, add 10–15 relevant hashtags including #{streamer} and relevant topics.
- No markdown formatting, no bullet points — just plain text caption, then hashtags line.
- Keep the whole thing under 250 words.

Return ONLY the caption text + hashtags. No explanation, no extra text."""


async def generate_master_caption(
    moments: list[Moment],
    segments: list[dict],
    api_key: str,
    model: str = "deepseek-v4-flash",
    streamer: str = "Streamer",
    video_title: str = "",
    campaign_brief: str = "",
) -> str:
    """Generate a single master social-media caption + hashtags with streamer awareness and Campaign Brief compliance."""
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    display_streamer = streamer if streamer else "Streamer"

    clip_summaries = "\n".join(
        f"{i+1}. {' / '.join(m.caption_lines)} {m.emoji}"
        for i, m in enumerate(moments)
    )
    transcript_excerpt = _format_transcript(segments[:60])

    user_content = (
        f"Clip captions:\n{clip_summaries}\n\n"
        f"Transcript excerpt:\n{transcript_excerpt}"
    )

    system = _MASTER_CAPTION_PROMPT.format(
        streamer=display_streamer,
        video_title=video_title,
    )
    if campaign_brief:
        system += f"\n\nCAMPAIGN BRIEF & HASHTAG REQUIREMENTS:\n{campaign_brief}\nEnsure mandatory hashtags and caption rules from the brief are applied."

    models_to_try = [model, "deepseek-chat", "deepseek-reasoner"]
    seen = set()
    models_to_try = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

    logger.info("Generating master caption for streamer '%s'", display_streamer)
    for m in models_to_try:
        try:
            response = await client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
                max_tokens=512,
            )
            caption = (response.choices[0].message.content or "").strip()
            logger.info("Master caption generated (%d chars) via model '%s'", len(caption), m)
            return caption
        except Exception as exc:
            logger.warning("Master caption generation failed on model '%s': %s", m, exc)

    return f"🔥 Best highlights from {display_streamer}! Which clip was your favorite? 👇\n\n#{display_streamer.replace(' ', '')} #Viral #StreamHighlights #Shorts #TikTok #Reels"


async def generate_clip_captions(
    moments: list[Moment],
    api_key: str,
    streamer: str = "Streamer",
    video_title: str = "",
    campaign_brief: str = "",
) -> list[str]:
    """Generate high-CTR YouTube titles + video-relevant hashtags for EACH clip (Clip 1 to Clip N)."""
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    display_streamer = streamer if streamer else "Streamer"
    streamer_tag = "#" + re.sub(r"[^\w]", "", display_streamer)

    campaign_text = f"CAMPAIGN BRIEF: {campaign_brief}" if campaign_brief else ""

    prompt = (
        "You are an elite YouTube Shorts & Video title strategist specializing in high Click-Through-Rate (CTR) titles.\n"
        "Generate ONE viral, click-maximizing YouTube title + 5 to 10 video-relevant hashtags for THIS specific clip moment.\n\n"
        f"Streamer: {display_streamer}\n"
        f"Video Title: {video_title}\n\n"
        "Rules:\n"
        "1. HIGH-CTR YOUTUBE TITLE: Create an irresistible, high click-score YouTube title (max 90 chars). Use curiosity gaps, high-CTR hooks, emotional triggers, or suspense. Mention {streamer} if natural.\n"
        "2. RELEVANT HASHTAGS: Append 5 to 10 hashtags directly related to the video content, topic, and streamer ({streamer_tag}).\n"
        f"{campaign_text}\n\n"
        "Return ONLY plain text with the High-CTR YouTube Title on line 1, followed by a blank line, and then the hashtags. No markdown headers, no quotes, no extra explanation."
    )

    async def _gen_one(i: int, m: Moment) -> str:
        headline = " / ".join(m.caption_lines) if hasattr(m, "caption_lines") else f"Clip {i+1}"
        user_content = f"Clip {i+1} Headline: {headline} {getattr(m, 'emoji', '🤯')}"

        try:
            resp = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt.format(streamer=display_streamer, streamer_tag=streamer_tag)},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
                max_tokens=250,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as exc:
            logger.warning("Clip %d YouTube title generation failed: %s", i+1, exc)

        # High-CTR Fallback
        fallback_title = f"{display_streamer} COULD NOT BELIEVE THIS HAPPENED! 😱"
        return f"{fallback_title}\n\n{streamer_tag} #Shorts #YouTube #Viral #Trending #{display_streamer.replace(' ', '')}"

    results = await asyncio.gather(*(_gen_one(i, m) for i, m in enumerate(moments)))
    return list(results)


async def normalize_streamer_name(
    raw_name: str,
    video_title: str,
    api_key: str,
) -> str:
    """
    Use LLM to cleanly normalize raw channel handle/title into the actual 1-word or clean display name of the streamer.
    e.g., 'Suburbbaby Live' -> 'Suburbbaby', 'lacy_official_vods' -> 'Lacy', 'KaiCenatLive' -> 'Kai'.
    """
    if not raw_name or raw_name.lower() in ("streamer", "na", "none", "unknown", "desktop upload"):
        return "Streamer"

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = (
        "You are an expert stream analyst.\n"
        "Extract ONLY the clean, properly capitalized single name or display handle of the primary streamer/creator.\n\n"
        "Rules:\n"
        "1. Return ONLY the clean single name or display name (e.g. 'Lacy', 'Kai', 'Adin', 'Fanum', 'Speed', 'Suburbbaby').\n"
        "2. Strip out words like 'Live', 'VODs', 'Official', 'Twitch', 'Channel', 'Stream', 'Clips'.\n"
        "3. Do NOT add quotes, markdown, or explanation — return ONLY the clean name.\n\n"
        f"Raw Handle/Channel: {raw_name}\n"
        f"Video Title: {video_title}"
    )

    try:
        resp = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=20,
        )
        clean = (resp.choices[0].message.content or "").strip()
        clean = re.sub(r"[^\w\s-]", "", clean).strip()
        if clean and len(clean) < 30:
            logger.info("Normalized streamer name: '%s' -> '%s'", raw_name, clean)
            return clean
    except Exception as exc:
        logger.warning("Streamer name normalization failed for '%s': %s", raw_name, exc)

    # Simple heuristic fallback: first word if raw_name has multiple words
    words = raw_name.strip().split()
    first_word = words[0] if words else raw_name
    if first_word.lower() in ("live", "the", "official", "stream"):
        first_word = words[1] if len(words) > 1 else raw_name
    return first_word.capitalize()
