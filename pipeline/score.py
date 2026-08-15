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
from .text_utils import format_seo_title, mask_profanity, generate_rich_hashtags

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
    title: str = ""       # YouTube Shorts title (≤50 chars, curiosity-gap style)
    hashtags: str = ""    # 6 to 10 highly relevant topic hashtags (e.g. #Streamer #Topic #ViralClips)
    bgm_track: str = "none" # "hype", "suspense", "funny", "sad", or "none"
    sfx_events: list[dict] = field(default_factory=list)
    aura_word: str = ""   # Single cinematic keyword that defines this clip's energy (e.g. COOKED, AURA, IMPRESSED)


    @property
    def duration(self) -> float:
        return self.end - self.start


def clean_streamer_name(raw_name: str) -> str:
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
1. TRUE IDENTITY FROM TRANSCRIPT DIALOGUE (CRITICAL): Channel names or handles (e.g., 'jayyy566', 'user9218', 'Daily VOD Clips', 'Ray Life', 'Kai Live') are frequently handles, aggregators, or suffix-heavy names. You MUST analyze the spoken dialogue in the transcript to deduce the TRUE REAL NAME by which the streamer is addressed in speech (e.g., "Ray", "Carterefe", "Kai", "N3on", "Davido", "Speed", "xQc", "Adin").
2. STRIP CHANNEL SUFFIXES & BRAND WORDS:
   - If channel is named 'Ray Life', 'Ray Live', 'Ray Clips' -> Real Name is ONLY "Ray" (NEVER call them "Life" or "Live"!).
   - If channel is named 'Kai Cenat Live', 'Kai Clips' -> Real Name is ONLY "Kai" or "Kai Cenat".
   - If channel is named 'N3on Live', 'N3on Central' -> Real Name is ONLY "N3on".
3. USE CLEAN SHORT NAMES: Once you deduce the true identity, use ONLY their clean short real names in text overlays, YouTube Shorts titles, and hashtags.
4. SECONDARY/GUEST SPEAKER RULE — "BRO" (MANDATORY): When referencing any person who is NOT the main streamer — whether a guest, chat user, random caller, interviewer, opponent, or anyone else — ALWAYS call them "Bro" in caption_lines and titles. Do NOT try to guess or invent their name. "Bro" is gender-neutral Gen Z language that works for both males and females. Example: "Bro CLAPPED Back Hard 😂", "Bro Said WHAT?! 😱", "Bro Can't HANDLE It 💀".
5. PROFANITY MASKING & GARDEN-FRIENDLY COMPLIANCE (MANDATORY): NEVER output raw un-censored swear words, toxic terms, or explicit language in caption_lines, titles, or descriptions. All content MUST be 100% garden-friendly (family-friendly, advertiser-safe, algorithm-optimized). Always sanitize any profanity using asterisks (e.g. "F**K", "K*LL", "SH*T", "B*TCH").
6. MANDATORY EMOJIS (2 TO 3 EMOJIS): Every single title and caption MUST include 2 to 3 vibrant, high-energy, contextually relevant emoji characters (e.g. "🔥😂💀", "😱🚨🤯", "💀😭🔥").
7. EVEN TIMELINE DISTRIBUTION: Spread the {top_n} moments across the full timeline of the video (beginning, middle, and end).
8. GARDEN-FRIENDLY TOPIC HASHTAGS (12 TO 18 HASHTAGS): For EVERY clip, generate 12 to 18 HIGHLY RELEVANT, garden-friendly hashtags based directly on the video's actual topic, streamer name, action, and subject matter (e.g. "#{streamer} #{streamer}Clips #{streamer}Highlights #TopicName #Gaming #Highlights #ViralShorts #TikTokViral #ReelsTrends #FYP #Trending #Shorts"). Include as many relevant hashtags as possible!

CAPTION FORMAT (3-LINE VIRAL STORYTELLING CARD):
9. EXACTLY 3 LINES in caption_lines — Follow the viral 3-part narrative hook pattern:
   - Line 1 = HOOK / SETUP with 1 ALL CAPS word (e.g. "she asked about VIRGINS")
   - Line 2 = ACTION / RESPONSE (e.g. "he said yes" or "Bro responded instantly")
   - Line 3 = PAYOFF / ESCALATION with 1 ALL CAPS word (e.g. "then she went FURTHER")
   - Max 4–5 words per line. Keep Phrasing casual, natural, and garden-friendly.
10. CAPS EMPHASIS WORDS: Put 1 key emphasis word in ALL CAPS in Line 1 and Line 3 so they automatically render in bold text on the top white card overlay.
11. 100% UNIQUE PHRASING — no repeated words, hooks, or sentence structure across any clip in this batch.
12. DO NOT put emojis directly inside caption_lines — they belong in the "emoji" field only (which is appended to line 3 during PNG rendering).

AURA KEYWORD RULE (100% CONTEXTUALLY RELEVANT):
12. For EVERY clip, pick ONE single uppercase word that directly relates to what is happening in THIS SPECIFIC moment.
    - It MUST be a single word (no spaces, no punctuation)
    - It MUST be ALL CAPS (e.g. PRANK, SPOTTED, KNOCKOUT, RECOGNIZED, BUSTED, ROASTED, EXPOSED, SHOCKED)
    - CRITICAL: Do NOT pick generic random words like "AURA" or "WHO". The word MUST be 100% derived from the actual clip action/topic (e.g., if it's a boxer prank, use `PRANK` or `BOXER`; if a fan recognizes the streamer, use `SPOTTED` or `CAUGHT`; if an insult occurs, use `ROASTED`).

VIRAL CLIP SELECTION — MANDATORY HOOK → EVENT → PAYOFF EVALUATION:
Evaluate every candidate moment strictly based on the HOOK → EVENT → PAYOFF storytelling structure.
The selector's primary objective is: Find self-contained mini-stories that hook the viewer, build tension/curiosity, and deliver a 100% complete, emotionally or visually satisfying payoff.

1. HOOK (0–3s): Identify where viewer curiosity/interest begins (unexpected statement, question, conflict, person approaching, visual surprise, or reaction). Start the clip shortly BEFORE the hook begins.
2. EVENT (Development): Identify the situation unfolding after the hook. Remove dead time and meaningless chatter.
3. PAYOFF (CRITICAL & MANDATORY): Identify the exact moment that rewards the viewer for watching (a punchline, shocking statement, funny response, confrontation, or sudden realization).
   - PAYOFF WEIGHTING (CRITICAL): Give EXTRA WEIGHT to payoff strength in the "score" field.
   - DO NOT select a moment simply because someone says something interesting. If the setup is good but the payoff is weak, LOWER THE VIRAL SCORE (score <= 60).
   - NEVER CUT OFF A CLIP BEFORE THE PAYOFF FINISHES! The viewer MUST receive a complete, satisfying conclusion before the clip ends.
4. START/END OPTIMIZATION:
   - Start timestamp: 2–3 seconds BEFORE the hook begins to provide immediate context.
   - End timestamp: 2–3 seconds AFTER the payoff/reaction completes so the clip feels like a complete mini-story and doesn't cut mid-sentence or mid-laughter.
5. BEST CANDIDATE BEHAVIOR:
   - Prefer (Strong Hook + Clear Event + Strong Payoff) over (Strong Hook + Long Conversation + Weak Payoff).
   - Prefer (Short Setup + Immediate Payoff) over (Long Setup + Slightly Better Payoff).
   - Ensure anyone who has NEVER seen this streamer before can watch the clip and immediately understand: "Oh, this is what is going on!"

Return ONLY a valid JSON array with exactly {top_n} objects. No markdown, no explanation, just raw JSON.

Each object must have exactly these fields:
  "start"         — float, seconds from start of video (shortly BEFORE hook)
  "end"           — float, seconds from start of video (shortly AFTER payoff completes)
  "is_viral_candidate" — boolean (true)
  "viral_score"   — float, score from 0.0 to 10.0 heavily weighted by payoff strength
  "payoff_description" — string, concise 1-sentence description of the exact payoff/resolution
  "caption_lines" — array of EXACTLY 3 strings. Line 1 = hook with 1 ALL CAPS word. Lines 2–3 = payoff with 1 more ALL CAPS word somewhere. Max 5 words/line. NO emoji in this field.
  "emoji"         — 2 to 3 relevant emoji characters (e.g. "🔥😂💀", "😱🚨🤯", "💀😭🔥")
  "score"         — integer, Viral potential score from 0 to 100
  "reasoning"     — string, a punchy 1-sentence explanation of why Hook -> Event -> Payoff is complete
  "title"         — string, HIGH-CTR SEO Title (35–50 chars). Written as a clean, compelling story sentence ending with 2–3 emojis. CRITICAL SEO RULES: NO slashes (/ or \\), NO pipes (|), NO ellipses (...), NO quotation marks, and NO hashtags (#Shorts or #Viral). Examples: "Jason Becomes a Voice Actor 😭🎙️🔥", "Speed Did Not Expect This Reaction 💀😂🔥", "When She Realized Who He Was 😱🔥👀", "Bro Thought He Was Getting Away With It 💀🤣🔥"
  "hashtags"      — string, 12 to 18 garden-friendly, topic-specific viral hashtags separated by spaces (e.g. "#{streamer} #{streamer}Clips #GamingMoments #FunnyClips #StreamerHighlights #ViralShorts #TikTokViral #ReelsTrends #FYP #Trending #ExplorePage #Viral")
  "aura_word"     — string, ONE single ALL CAPS word capturing this clip's energy (e.g. "COOKED", "AURA", "IMPRESSED", "CAUGHT", "DEAD", "WILD", "REAL", "WHO", "NOPE")
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
    model: str = "deepseek-chat",
    streamer: str = "Streamer",
    video_title: str = "",
    campaign_brief: str = "",
    target_duration: str = "auto",
    spike_windows: list[dict] | None = None,
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
    dur_text = dur_rules.get(
        target_duration,
        "Each moment's duration MUST be dynamically determined by the exact natural length of the Hook -> Event -> Payoff story arc (between 10 and 60 seconds). Let the complete sentence and natural conversation payoff determine the exact clip length so the moment never cuts off."
    )

    system = _SYSTEM_PROMPT.format(
        top_n=top_n,
        streamer=display_streamer,
        video_title=video_title,
    )
    # Append the selected clip-duration rule (prompt has no line 12 to replace)
    system += f"\n\n12. CLIP DURATION: {dur_text}"

    if spike_windows:
        spike_text = "\n".join(
            f"• Chart Peak Spike @ {int(w['start_sec']//60):02d}:{int(w['start_sec']%60):02d}–{int(w['end_sec']//60):02d}:{int(w['end_sec']%60):02d} (Spike Density: {int(w['spike_score']*100)}%)"
            for w in spike_windows
        )
        system += (
            f"\n\nLIVE VIEWER DENSITY & CHART REPLAY SPIKES DETECTED:\n{spike_text}\n\n"
            "MANDATORY SPIKE SELECTION DIRECTIVE:\n"
            "The timestamp ranges above represent peak viewer activity, live chat velocity bursts, and replay heatmap climaxes.\n"
            "You MUST prioritize selecting clip candidate moments that align closely with these peak chart spike timestamp ranges!"
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
                        start=max(0.0, float(item["start"]) - 2.0),
                        end=max(float(item["start"]) + 15.0, float(item["end"]) + 3.5),
                        caption_lines=list(item["caption_lines"]),
                        emoji=item.get("emoji", "🔥😂💀"),
                        score=int(float(item.get("viral_score", item.get("score", 9.0))) * 10) if float(item.get("viral_score", 0.0)) > 0 else int(item.get("score", 90)),
                        reasoning=item.get("reason", item.get("reasoning", "Strong Hook -> Event -> Payoff arc.")),
                        title=format_seo_title(item.get("title") or item.get("caption_lines", []), display_streamer, default_emoji=item.get("emoji", "🔥😂💀")),
                        hashtags=generate_rich_hashtags(streamer=display_streamer, topic=str(item.get("title", "")), aura_word=str(item.get("aura_word", "")), existing_hashtags=item.get("hashtags", "")),
                        bgm_track=item.get("bgm_track", "hype"),
                        sfx_events=list(item.get("sfx_events", [])),
                        aura_word=str(item.get("aura_word", "")).upper().strip(),
                    )
                    for i, item in enumerate(data)
                ]
                logger.info("Scored %d moments via model '%s'", len(moments), m)
                return moments[:top_n]

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


def verify_and_clean_visual_moments(moments: list[Moment], video_path: Path | str | None = None) -> list[Moment]:
    """
    Inspect candidate moments using OpenCV keyframe scanner.
    If a moment contains an ugly YouTube 'SUBSCRIBE' banner or end-screen pop-up,
    automatically shift start timestamp or adjust window to keep footage clean.
    """
    if not video_path or not Path(video_path).exists():
        return moments

    try:
        from .crop import analyze_keyframe_visuals, verify_visual_hook_alignment
        cleaned = []
        for m in moments:
            has_overlay, motion = analyze_keyframe_visuals(Path(video_path), m.start)
            if has_overlay:
                logger.info(
                    "  Moment %02d (%.1fs) detected YouTube Subscribe banner overlay — shifting start +4s for clean footage",
                    m.index, m.start,
                )
                m.start += 4.0  # Shift past the Subscribe popup banner
                m.end = max(m.start + 15.0, m.end + 2.0)

            # Optimize clip start timestamp using visual reaction keyframe alignment
            m.start = verify_visual_hook_alignment(Path(video_path), m.start)
            cleaned.append(m)
        return cleaned
    except Exception as exc:
        logger.warning("Visual keyframe overlay check skipped: %s", exc)
        return moments


def _generate_fallback_moments(segments: list[dict], top_n: int = 10, streamer: str = "Streamer") -> list[Moment]:
    """Generate evenly spaced timeline moments as a reliable fallback if LLM scoring fails."""
    if not segments:
        return [Moment(index=0, start=0.0, end=30.0, caption_lines=[f"{streamer} Highlights", "Best Moments", "VIRAL"], emoji="🔥", title=f"{streamer} Craziest Stream Highlights 🔥")]
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
                title=f"{streamer} Best Stream Moments 🔥",
            )
        )
    res = fallback if fallback else [Moment(index=0, start=0.0, end=30.0, caption_lines=[f"{streamer} Stream", "Best Moments", "UNREAL"], emoji="🔥", title=f"{streamer} Best Stream Moments 🔥")]
    return res[:top_n]


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
    model: str = "deepseek-chat",
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

async def generate_clip_captions(
    moments: list[Moment],
    streamer: str,
    video_title: str,
    api_key: str,
    campaign_brief: str = "",
) -> list[str]:
    """Generate high-CTR YouTube titles + rich social captions + 10-15 topic hashtags for EACH clip (TikTok, IG, FB, YT)."""
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    display_streamer = streamer if streamer else "Streamer"
    streamer_tag = "#" + re.sub(r"[^\w]", "", display_streamer)

    campaign_text = f"CAMPAIGN BRIEF: {campaign_brief}" if campaign_brief else ""

    prompt = (
        "You are an elite YouTube Shorts & Social Media caption strategist specializing in high Click-Through-Rate (CTR) titles and long viral captions.\n"
        "Generate a complete social posting package for THIS specific clip moment:\n\n"
        f"Streamer: {display_streamer}\n"
        f"Video Title: {video_title}\n\n"
        "Rules:\n"
        "1. HIGH-CTR SEO YOUTUBE TITLE (LINE 1): Create an irresistible, high curiosity story title (35–50 chars) ending with 1–2 emojis. STRICT SEO RULES: NO slashes (/ or \\), NO pipe symbols (|), NO ellipses (...), NO quotation marks, and NO hashtags in the title! (e.g. 'Jason Becomes a Voice Actor 😭🎙️', 'Speed Did Not Expect This Reaction 💀🔥').\n"
        "2. RICH SOCIAL CAPTION BODY (PARAGRAPH 2): Write a 2 to 3 sentence engaging story summary summarizing what happens in this clip, with high-energy narrative phrasing and 2–3 emojis. This will be used for TikTok, Instagram Reels, and Facebook Reels captions!\n"
        "3. EXTENDED HASHTAG BLOCK (PARAGRAPH 3): Generate 10 to 15 clean, garden-friendly, topic-specific hashtags for TikTok, Instagram, Facebook, and YouTube Shorts (e.g., '{streamer_tag} #Gaming #Viral #Shorts #TikTok #Reels #FYP #Trending #StreamerHighlights #FunnyClips').\n"
        "4. GARDEN-FRIENDLY COMPLIANCE: All text MUST be 100% garden-friendly (advertiser-safe, family-friendly, algorithm-optimized, sanitize any profanity).\n"
        f"{campaign_text}\n\n"
        "Return ONLY plain text with: Line 1 = Title, followed by blank line, followed by Paragraph 2 = Rich Social Caption, followed by blank line, followed by Paragraph 3 = 10 to 15 Hashtags. No markdown headers, no quotes."
    )

    async def _gen_one(i: int, m: Moment) -> str:
        raw_m_title = getattr(m, "title", "")
        m_title = format_seo_title(raw_m_title or getattr(m, "caption_lines", []), display_streamer, default_emoji=getattr(m, "emoji", "🔥"))
        m_tags = getattr(m, "hashtags", "")
        
        if m_title and m_tags:
            lines_body = " ".join(m.caption_lines) if hasattr(m, "caption_lines") else ""
            lines_body = mask_profanity(lines_body)
            social_body = f"{lines_body} {getattr(m, 'emoji', '😱')} — Watch until the end for the full moment!"
            return f"{m_title}\n\n{social_body}\n\n{m_tags}"

        user_content = f"Clip {i+1} Setup: {m_title}"

        try:
            resp = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt.format(streamer=display_streamer, streamer_tag=streamer_tag)},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
                max_tokens=350,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                parts = text.split("\n\n")
                if parts:
                    parts[0] = format_seo_title(parts[0], display_streamer, default_emoji=getattr(m, "emoji", "🔥"))
                    return "\n\n".join(parts)
                return text
        except Exception as exc:
            logger.warning("Clip %d YouTube title generation failed: %s", i+1, exc)

        fallback_title = format_seo_title(f"{display_streamer} Could Not Believe This Happened", display_streamer, default_emoji="😱")
        fallback_tags = f"{streamer_tag} #{display_streamer.replace(' ', '')}Clips #{display_streamer.replace(' ', '')}Live #StreamerHighlights #Shorts #Viral"
        return f"{fallback_title}\n\n{fallback_tags}"

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
