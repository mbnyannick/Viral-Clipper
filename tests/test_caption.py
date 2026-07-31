"""
Tests for pipeline/caption.py

The full render test requires the actual font files to be present on disk.
It is skipped automatically if the fonts are missing (e.g., in CI without
the assets directory set up). Run it locally after completing the setup
steps in the README.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.score import Moment
from pipeline.errors import PipelineError


def _make_moment(**kwargs) -> Moment:
    defaults = dict(
        index=0,
        start=10.0,
        end=40.0,
        caption_lines=["PIGFORD OUT HERE DENYING", "everything"],
        emoji="😤",
        score=99,
        reasoning="Testing",
        title="Testing Title",
        bgm_track="none",
        sfx_events=[]
    )
    defaults.update(kwargs)
    return Moment(**defaults)


# ── Font-dependent render tests (skipped if assets not present) ─────────────────

@pytest.fixture
def assets_dir() -> Path:
    return Path("assets")


def _fonts_available(assets_dir: Path) -> bool:
    return (assets_dir / "fonts" / "Bold.ttf").exists()


def test_render_caption_produces_correct_dimensions(assets_dir, tmp_path):
    if not _fonts_available(assets_dir):
        pytest.skip("Font assets not present — run setup steps from README first")

    from pipeline.caption import render_caption
    from PIL import Image

    moment = _make_moment()
    out = tmp_path / "caption_00.png"
    height = render_caption(moment, assets_dir, out)

    assert out.exists(), "Caption PNG was not created"
    assert out.stat().st_size > 0, "Caption PNG is empty"
    assert height > 0

    img = Image.open(out)
    assert img.width == 720
    assert img.height == height


def test_render_caption_minimum_height(assets_dir, tmp_path):
    if not _fonts_available(assets_dir):
        pytest.skip("Font assets not present")

    from pipeline.caption import render_caption

    # Single very short line
    moment = _make_moment(caption_lines=["ok"], emoji="")
    out = tmp_path / "caption_short.png"
    height = render_caption(moment, assets_dir, out)
    # Minimum height guard ensures caption is never a sliver
    assert height >= 80


def test_render_caption_multiline(assets_dir, tmp_path):
    if not _fonts_available(assets_dir):
        pytest.skip("Font assets not present")

    from pipeline.caption import render_caption, CANVAS_H

    moment = _make_moment(
        caption_lines=["THIS IS THE FIRST LINE", "and this is the second", "wow a third"],
        emoji="🔥",
    )
    out = tmp_path / "caption_multi.png"
    h3 = render_caption(moment, assets_dir, out)

    moment2 = _make_moment(caption_lines=["ONE LINE"], emoji="")
    out2 = tmp_path / "caption_single.png"
    h1 = render_caption(moment2, assets_dir, out2)

    # Non-face-crop captions are full-canvas PNGs — both return CANVAS_H
    assert h3 == CANVAS_H
    assert h1 == CANVAS_H


# ── Font-not-found raises PipelineError ────────────────────────────────────────

def test_render_caption_missing_bold_font_raises(tmp_path):
    from pipeline.caption import render_caption

    fake_assets = tmp_path / "assets"
    (fake_assets / "fonts").mkdir(parents=True)
    # NotoColorEmoji present but Bold.ttf missing

    moment = _make_moment()
    out = tmp_path / "out.png"
    with pytest.raises(PipelineError) as exc_info:
        render_caption(moment, fake_assets, out)
    assert exc_info.value.step == "caption"
    assert "Bold.ttf" in exc_info.value.reason


# ── _is_emphasis unit tests ─────────────────────────────────────────────────────

def test_is_emphasis_all_caps():
    from pipeline.caption import _is_emphasis
    assert _is_emphasis("DENIED") is True
    assert _is_emphasis("PIGFORD") is True


def test_is_emphasis_lowercase():
    from pipeline.caption import _is_emphasis
    assert _is_emphasis("everything") is False
    assert _is_emphasis("the") is False


def test_is_emphasis_mixed():
    from pipeline.caption import _is_emphasis
    assert _is_emphasis("Pigford") is False


def test_is_emphasis_with_punctuation():
    from pipeline.caption import _is_emphasis
    # Punctuation stripped before check
    assert _is_emphasis("DENIED!") is True
    assert _is_emphasis("WOW,") is True


def test_render_caption_face_crop(assets_dir, tmp_path):
    if not _fonts_available(assets_dir):
        pytest.skip("Font assets not present")

    from pipeline.caption import render_caption
    from PIL import Image

    moment = _make_moment(caption_lines=["FACE CROP TEST"], emoji="🔥")
    out = tmp_path / "caption_face_crop.png"
    height = render_caption(moment, assets_dir, out, layout_mode="face_crop")

    assert out.exists()
    assert height > 0

    img = Image.open(out)
    assert img.width == 720
    assert img.height == height

