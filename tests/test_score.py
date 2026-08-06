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
        assert m.start == pytest.approx(142.5)
        assert m.end == pytest.approx(167.0)
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
