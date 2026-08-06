"""
pipeline/pixabay_assets.py — Pixabay API Media & Audio Asset Manager

Fetches high quality royalty-free background music and sound effects
using your active Pixabay API Key (PIXABAY_API_KEY).
"""

import asyncio
import logging
import os
import subprocess
import urllib.parse
import urllib.request
import json
from pathlib import Path

logger = logging.getLogger(__name__)


async def fetch_pixabay_bgm(mood: str, output_dir: Path) -> Path | None:
    """
    Download a high-quality royalty-free background music track matching `mood`
    from Pixabay using PIXABAY_API_KEY.
    """
    key = os.getenv("PIXABAY_API_KEY", "").strip()
    if not key:
        logger.warning("PIXABAY_API_KEY not set in environment.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_mp3 = output_dir / f"pixabay_{mood}.mp3"
    if out_mp3.exists() and out_mp3.stat().st_size > 10000:
        logger.info("Using cached Pixabay track: %s", out_mp3.name)
        return out_mp3

    search_query = f"{mood} beat music"
    try:
        url = f"https://pixabay.com/api/videos/?key={key}&q={urllib.parse.quote(search_query)}&per_page=5"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        def _fetch_track():
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode())
                hits = data.get("hits", [])
                if not hits:
                    return None
                
                # Pick top matching item
                video_item = hits[0]
                videos = video_item.get("videos", {})
                media_url = videos.get("medium", {}).get("url") or videos.get("large", {}).get("url")
                if not media_url:
                    return None

                # Extract audio stream with ffmpeg
                cmd = [
                    "ffmpeg", "-y", "-i", media_url,
                    "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                    str(out_mp3)
                ]
                proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
                if proc.returncode == 0 and out_mp3.exists() and out_mp3.stat().st_size > 5000:
                    return out_mp3
            return None

        result_path = await asyncio.to_thread(_fetch_track)
        if result_path:
            logger.info("Successfully fetched Pixabay track for mood '%s': %s", mood, result_path.name)
            return result_path
    except Exception as exc:
        logger.warning("Pixabay BGM fetch exception for '%s': %s", mood, exc)

    return None
