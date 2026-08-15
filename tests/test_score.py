"""
Tests for pipeline/score.py

Tests:
- Valid JSON response is parsed into Moment objects.
- Markdown code fences are stripped before parsing.
- Bad JSON on first attempt triggers a retry on second attempt.
- Two consecutive bad JSON responses raise PipelineError.
- _format_transcript produces the expected text format.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.score import score_moments, Moment, _format_transcript
from pipeline.errors import PipelineError
from pipeline.voiceover import generate_voiceover
from pipeline.composite import voiceover_generation_enabled

# ── Fixtures ────────────────────────────────────────────────────────────────────

SAMPLE_SEGMENTS = [
    {"text": "Hey what is up guys", "start": 0.0, "end": 3.5},
    {"text": "So today we are going to talk about something crazy", "start": 3.5, "end": 8.0},
    {"text": "I cannot believe this happened", "start": 142.5, "end": 146.0},
    {"text": "He denied everything on stream", "start": 146.0, "end": 150.0},
]

VALID_MOMENT_JSON = json.dumps([
    {
        "start": 142.5,
        "end": 167.0,
        "caption_lines": ["PIGFORD OUT HERE DENYING", "everything"],
        "emoji": "😤",
    }
])


def _make_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


def _patch_client(side_effect):
    """Patch AsyncOpenAI and return the mocked create callable."""
    patcher = patch("pipeline.score.AsyncOpenAI")
    mock_cls = patcher.start()
    mock_client = AsyncMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create = AsyncMock(side_effect=side_effect)
    return patcher, mock_client


# ── Tests ───────────────────────────────────────────────────────────────────────

async def test_score_parses_valid_json():
    patcher, _ = _patch_client([_make_response(VALID_MOMENT_JSON)])
    try:
        moments = await score_moments(SAMPLE_SEGMENTS, api_key="test", top_n=1)
        assert len(moments) == 1
        m = moments[0]
        assert isinstance(m, Moment)
        assert m.start == pytest.approx(140.5)
        assert m.end == pytest.approx(170.5)
        assert m.emoji == "😤"
        assert m.caption_lines[0] == "PIGFORD OUT HERE DENYING"
        assert m.caption_lines[1] == "everything"
        assert m.index == 0
    finally:
        patcher.stop()


async def test_score_strips_json_code_fences():
    fenced = f"```json\n{VALID_MOMENT_JSON}\n```"
    patcher, _ = _patch_client([_make_response(fenced)])
    try:
        moments = await score_moments(SAMPLE_SEGMENTS, api_key="test", top_n=1)
        assert len(moments) == 1
    finally:
        patcher.stop()


async def test_score_strips_plain_code_fences():
    fenced = f"```\n{VALID_MOMENT_JSON}\n```"
    patcher, _ = _patch_client([_make_response(fenced)])
    try:
        moments = await score_moments(SAMPLE_SEGMENTS, api_key="test", top_n=1)
        assert len(moments) == 1
    finally:
        patcher.stop()


async def test_score_retries_on_bad_json():
    """First call returns bad JSON; second call returns valid JSON."""
    patcher, _ = _patch_client([
        _make_response("not valid json at all"),
        _make_response(VALID_MOMENT_JSON),
    ])
    try:
        moments = await score_moments(SAMPLE_SEGMENTS, api_key="test", top_n=1)
        assert len(moments) == 1
    finally:
        patcher.stop()


async def test_score_raises_after_two_bad_json():
    """Two consecutive bad JSON responses → PipelineError."""
    patcher, _ = _patch_client([
        _make_response("garbage"),
        _make_response("still garbage"),
    ])
    try:
        with pytest.raises(PipelineError) as exc_info:
            await score_moments(SAMPLE_SEGMENTS, api_key="test", top_n=1)
        assert exc_info.value.step == "score"
    finally:
        patcher.stop()


def test_format_transcript_structure():
    text = _format_transcript(SAMPLE_SEGMENTS)
    lines = text.splitlines()
    assert len(lines) == len(SAMPLE_SEGMENTS)
    assert lines[0].startswith("[0.0s]")
    assert "Hey what is up guys" in lines[0]
    assert "[142.5s]" in lines[2]


async def test_moment_duration_property():
    m = Moment(index=0, start=10.0, end=45.5, caption_lines=["TEST"], emoji="🔥", score=90, reasoning="Test", title="Test", bgm_track="none", sfx_events=[])
    assert m.duration == pytest.approx(35.5)


async def test_score_moments_truncates_to_top_n():
    """Verify that score_moments returns at most top_n moments even if LLM returns more."""
    five_moments_json = json.dumps([
        {"start": float(i * 10), "end": float(i * 10 + 20), "caption_lines": ["LINE1", "LINE2"], "emoji": "🔥"}
        for i in range(5)
    ])
    patcher, _ = _patch_client([_make_response(five_moments_json)])
    try:
        moments = await score_moments(SAMPLE_SEGMENTS, api_key="test", top_n=3)
        assert len(moments) == 3
    finally:
        patcher.stop()


def test_voiceover_generation_disabled_by_default(monkeypatch):
    """The workflow-level policy should default to disabling TTS voiceover synthesis."""
    monkeypatch.delenv("ENABLE_VOICEOVER", raising=False)
    assert voiceover_generation_enabled() is False

async def test_voiceover_generation_disabled_by_default(tmp_path, monkeypatch):
    """The runtime policy should shut off voiceover synthesis unless explicitly re-enabled."""
    monkeypatch.delenv("ENABLE_VOICEOVER", raising=False)
    out = tmp_path / "voiceover.mp3"
    result = await generate_voiceover("This is a hook", out)
    assert result is None
    assert not out.exists()


def test_format_seo_title_strips_slashes_and_pipes():
    from pipeline.text_utils import format_seo_title
    raw = "she asked about VIRGINS / he said yes / then she went FURTHER"
    title = format_seo_title(raw, default_emoji="🔥")
    assert "/" not in title
    assert "\\" not in title
    assert "|" not in title
    assert title.endswith("🔥")


def test_format_seo_title_removes_hashtags_and_quotes():
    from pipeline.text_utils import format_seo_title
    raw = '"Speed Got Caught In 4K #Shorts #Viral #Gaming"'
    title = format_seo_title(raw, default_emoji="😱")
    assert "#" not in title
    assert '"' not in title
    assert title.endswith("😱")


def test_format_seo_title_removes_ellipses_and_dangling_words():
    from pipeline.text_utils import format_seo_title
    raw = "This is an extremely long title that keeps talking about random stuff and things with someone on stream..."
    title = format_seo_title(raw, max_chars=40, default_emoji="💀")
    assert "..." not in title
    assert "…" not in title
    assert len(title) <= 50
    assert title.endswith("💀")


def test_format_seo_title_from_caption_lines():
    from pipeline.text_utils import format_seo_title
    lines = ["PIGFORD OUT HERE DENYING", "everything"]
    title = format_seo_title(lines, default_emoji="😤")
    assert "/" not in title
    assert "PIGFORD OUT HERE DENYING everything 😤" == title


def test_generate_rich_hashtags_generates_large_pool():
    from pipeline.text_utils import generate_rich_hashtags
    tags = generate_rich_hashtags(streamer="Kai Cenat", topic="Prank Call", aura_word="EXPOSED")
    tag_list = tags.split()
    assert len(tag_list) >= 12
    assert "#KaiCenat" in tag_list
    assert "#KaiCenatClips" in tag_list
    assert "#Exposed" in tag_list
    assert "#Shorts" in tag_list
    assert "#TikTokViral" in tag_list


def test_format_seo_title_handles_2_to_3_emojis():
    from pipeline.text_utils import format_seo_title
    title = format_seo_title("Speed Did Not Expect This Reaction", default_emoji="🔥😂💀")
    assert "🔥😂💀" in title


def test_generate_rich_hashtags_strips_suffixes():
    from pipeline.text_utils import generate_rich_hashtags
    tags = generate_rich_hashtags(streamer="PlaqueBoyMaxLive", topic="Flop on Stream")
    tag_list = tags.split()
    assert "#PlaqueBoyMax" in tag_list
    assert "#PlaqueBoyMaxLiveLive" not in tag_list


def test_extract_title_and_caption_parses_cleanly_without_meta_headers():
    from bot.handlers import _extract_title_and_caption
    sample_caption = (
        "🎬 <b>Clip 01/10 • PlaqueBoyMax</b> ⚡ <i>Score: 96/100 (S-Tier)</i>\n"
        "💡 <i>High viral potential.</i>\n\n"
        "🔴 <b>YouTube Title:</b>\n"
        "<code>The Flop He Didn't See Coming 😂💀🔥</code>\n\n"
        "📱 <b>Caption &amp; Hashtags:</b>\n"
        "<code>Bro hits a HIP check he flops hard 😂💀🤣\n\n#PlaqueBoyMax #Shorts</code>"
    )
    title, desc = _extract_title_and_caption(sample_caption, "1")
    assert "YouTube Title" not in title
    assert "YouTube Title" not in desc
    assert "The Flop He Didn't See Coming" in title
    assert "Bro hits a HIP check" in desc
    assert "#PlaqueBoyMax" in desc



