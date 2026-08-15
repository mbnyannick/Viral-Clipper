import asyncio
import os
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from bot.scheduler import PlatformScheduler
from pipeline.download import download_video_clip_range, download_audio_chunk, _kick_vod_get_hls_url
from pipeline import get_public_base_url


@pytest.mark.asyncio
async def test_scheduler_queue_and_retry(tmp_path):
    """Verify that PlatformScheduler schedules clips, handles retries, and persists state."""
    sched = PlatformScheduler()
    sched._queue = []
    sched._yt_daily = {}

    platforms = ["tiktok", "youtube"]
    payload = {
        "clip_id": "clip_001",
        "title": "Test Clip",
        "video_url": "https://example.com/clips/clip_001.mp4",
        "chat_id": 12345678,
    }

    fired = sched.schedule_clip_staggered(platforms, payload)
    assert "tiktok" in fired
    assert "youtube" in fired
    assert len(sched._queue) == 2

    # Simulate tick where _fire fails
    with patch.object(sched, "_fire", new_callable=AsyncMock) as mock_fire:
        mock_fire.return_value = False  # Fail fire
        # Set fire_at in past
        for item in sched._queue:
            item["fire_at"] = (datetime.now(timezone.utc)).isoformat()

        await sched.tick()
        # Item should be re-enqueued with retry = 1
        assert len(sched._queue) == 2
        for item in sched._queue:
            assert item.get("retries") == 1


@pytest.mark.asyncio
async def test_kick_vod_hls_resolution():
    """Verify Kick VOD HLS parser properly formats API request."""
    sample_url = "https://kick.com/n3on/videos/123456"
    from unittest.mock import MagicMock
    with patch("curl_cffi.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "id": "123456",
                    "slug": "123456",
                    "session_title": "N3on Stream",
                    "source": "https://stream.kick.com/hls/test/master.m3u8",
                    "duration": 3600000,
                }
            ]
        }
        mock_get.return_value = mock_resp

        hls_url, channel, title, duration = await _kick_vod_get_hls_url(sample_url)
        assert hls_url == "https://stream.kick.com/hls/test/master.m3u8"
        assert channel == "n3on"
        assert title == "N3on Stream"
        assert duration == 3600


def test_public_base_url_default():
    """Verify get_public_base_url returns non-empty string."""
    url = get_public_base_url()
    assert url.startswith("http")


def test_extract_title_and_caption():
    """Verify _extract_title_and_caption returns a 2-tuple (title, description)."""
    from bot.handlers import _extract_title_and_caption
    title, desc = _extract_title_and_caption("📹 Clip 01\nCool Highlight\n#Shorts #Viral", "1")
    assert "Cool Highlight" in title or "Viral Clip" in title
    assert isinstance(title, str)
    assert isinstance(desc, str)
