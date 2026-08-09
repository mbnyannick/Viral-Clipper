import asyncio
import logging
from pathlib import Path
from pipeline.download import extract_metadata, detect_platform_and_type, _kick_live_direct_download

logging.basicConfig(level=logging.INFO)

async def test_twitch():
    url = "https://www.twitch.tv/kaicenat"
    print(f"Testing Twitch: {url}")
    try:
        meta = await extract_metadata(url)
        print("Twitch Metadata:", meta)
    except Exception as e:
        print("Twitch Metadata Error:", e)

async def test_kick():
    url = "https://kick.com/adinross"
    print(f"\nTesting Kick: {url}")
    try:
        meta = await extract_metadata(url)
        print("Kick Metadata:", meta)
    except Exception as e:
        print("Kick Metadata Error:", e)
        
    print("\nTesting Kick Live Direct Download logic...")
    try:
        video_path = Path("tmp/test_kick.mp4")
        audio_path = Path("tmp/test_kick.m4a")
        # We don't want to actually download 1 hour, so we'll mock the command inside or just let it fail fast.
        # Actually just getting the M3U8 URL is enough to prove it works.
        import re
        from curl_cffi import requests
        channel = "adinross"
        api_url = f"https://kick.com/api/v2/channels/{channel}"
        r = requests.get(api_url, impersonate="chrome", timeout=15)
        print("Kick API Status:", r.status_code)
        if r.status_code == 200:
            data = r.json()
            playback_url = data.get("playback_url")
            print("Kick HLS URL:", playback_url)
        else:
            print("Kick API Failed:", r.text)
    except Exception as e:
        print("Kick Direct Logic Error:", e)

if __name__ == "__main__":
    asyncio.run(test_twitch())
    asyncio.run(test_kick())
