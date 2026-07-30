"""
Tests for pipeline/download.py
"""

import pytest
from pipeline.download import detect_platform_and_type, YT_CLIENT_CHAINS


def test_detect_platform_youtube():
    platform, content_type, live_status = detect_platform_and_type("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert "YouTube" in platform
    assert "Video" in content_type


def test_yt_client_chains_contains_player_clients():
    assert len(YT_CLIENT_CHAINS) >= 3
    has_web = any("web" in " ".join(opts) for opts in YT_CLIENT_CHAINS) or len(YT_CLIENT_CHAINS[0]) == 0
    assert has_web, "YT_CLIENT_CHAINS should contain web player client fallback"
