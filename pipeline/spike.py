"""
pipeline/spike.py — Multi-Signal Chart Spike Detection Engine

Extracts viewer activity spikes across YouTube, Kick.com, and Twitch VODs using:
1. Live Chat Density & Keyword Velocity (via `chat-downloader`)
2. YouTube Replay Heatmaps (`heatMarkers` via `yt-dlp`)
3. Audio Energy RMS Decibel Spikes (via `scipy` & FFmpeg loudness analysis)

Combines signals into a unified composite spike curve S(t) in [0.0, 1.0]
to pinpoint exact peak viral timestamps.
"""

import asyncio
import logging
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HYPE_KEYWORDS = {
    "kekw", "lmao", "lol", "rofl", "cooked", "snitched", "exposed", "wild",
    "pog", "poggers", "holy", "omg", "cringe", "nooo", "wtf", "damn", "bro",
    "w", "l", "gg", "nahhh", "aintnoway", "dead", "skull", "skull_emoji",
}


def extract_youtube_heatmap_spikes(url: str) -> list[dict[str, float]]:
    """
    Extract YouTube 'Most Replayed' heatmap data (heatMarkers) via yt-dlp.
    Returns a list of dicts: [{'time_sec': float, 'value': float}] (normalized 0.0 - 1.0).
    """
    if "youtube.com" not in url.lower() and "youtu.be" not in url.lower():
        return []

    try:
        import yt_dlp
        cookie_path = Path("cookies.txt")
        ydl_opts = {"skip_download": True, "quiet": True, "no_warnings": True}
        if cookie_path.exists():
            ydl_opts["cookiefile"] = str(cookie_path)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            heatmap = info.get("heatmap") or []
            results = []
            for marker in heatmap:
                start_time = float(marker.get("start_time", 0.0))
                value = float(marker.get("value", 0.0))
                results.append({"time_sec": start_time, "value": value})
            if results:
                logger.info("⚡ [SPIKE DETECTOR] Extracted %d YouTube heatMarkers!", len(results))
            return results
    except Exception as exc:
        logger.warning("YouTube heatmap extraction warning: %s", exc)
        return []


def extract_chat_density_spikes(url: str, duration_sec: float, max_messages: int = 25000) -> list[dict[str, float]]:
    """
    Extract live chat messages via chat-downloader for YouTube, Kick, or Twitch.
    Aggregates messages into 10-second windows and calculates density scores.
    """
    try:
        from chat_downloader import ChatDownloader
        downloader = ChatDownloader()
        chat = downloader.get_chat(url, max_messages=max_messages)
        
        # 10-second window buckets
        window_size = 10.0
        num_windows = max(1, int(math.ceil(duration_sec / window_size)))
        scores = [0.0] * num_windows

        msg_count = 0
        for message in chat:
            msg_count += 1
            time_in_sec = float(message.get("time_in_seconds", 0.0))
            if time_in_sec < 0 or time_in_sec > duration_sec:
                continue

            idx = min(num_windows - 1, int(time_in_sec // window_size))
            text = str(message.get("message", "")).lower()
            
            # Base message weight = 1.0
            weight = 1.0
            words = set(re.findall(r"\w+", text))
            if words.intersection(HYPE_KEYWORDS):
                weight = 1.6  # Hype word multiplier

            scores[idx] += weight

        if msg_count == 0:
            return []

        # Normalize 0.0 to 1.0
        max_score = max(scores) if scores and max(scores) > 0 else 1.0
        results = [
            {"time_sec": i * window_size, "value": round(score / max_score, 3)}
            for i, score in enumerate(scores)
        ]
        logger.info("⚡ [SPIKE DETECTOR] Processed %d chat messages across %d windows", msg_count, len(results))
        return results
    except Exception as exc:
        logger.warning("Chat density extraction warning: %s", exc)
        return []


def extract_audio_rms_spikes(audio_path: Path | str, duration_sec: float) -> list[dict[str, float]]:
    """
    Analyze audio decibel RMS energy envelope using FFmpeg ebur128 filter.
    Returns 2-second decibel RMS energy scores normalized from 0.0 to 1.0.
    """
    path_obj = Path(audio_path)
    if not path_obj.exists():
        return []

    try:
        cmd = [
            "ffmpeg", "-nostats", "-i", str(path_obj),
            "-filter_complex", "ebur128=peak=true",
            "-f", "null", "-"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Parse M: line outputs for loudness peaks
        loudness_samples: list[tuple[float, float]] = []
        for line in res.stderr.splitlines():
            if "t:" in line and "M:" in line:
                try:
                    parts = line.split()
                    t_str = [p for p in parts if p.startswith("t:")][0].replace("t:", "")
                    m_str = [p for p in parts if p.startswith("M:")][0].replace("M:", "")
                    t_val = float(t_str)
                    m_val = float(m_str)
                    loudness_samples.append((t_val, m_val))
                except Exception:
                    pass

        if not loudness_samples:
            return []

        # Map into 5-second windows
        window_size = 5.0
        num_windows = max(1, int(math.ceil(duration_sec / window_size)))
        window_loudness = [-100.0] * num_windows

        for t_val, m_val in loudness_samples:
            idx = min(num_windows - 1, int(t_val // window_size))
            if m_val > window_loudness[idx]:
                window_loudness[idx] = m_val

        # Convert dB (-70 LUFS -> -5 LUFS) to normalized 0.0 -> 1.0 scale
        min_lu, max_lu = -50.0, -10.0
        results = []
        for i, val in enumerate(window_loudness):
            norm_val = max(0.0, min(1.0, (val - min_lu) / (max_lu - min_lu)))
            results.append({"time_sec": i * window_size, "value": round(norm_val, 3)})

        logger.info("⚡ [SPIKE DETECTOR] Extracted %d audio RMS energy windows!", len(results))
        return results
    except Exception as exc:
        logger.warning("Audio RMS extraction warning: %s", exc)
        return []


async def compute_composite_spike_curve(
    url: str,
    audio_path: Path | str | None = None,
    duration_sec: float = 300.0
) -> list[dict[str, Any]]:
    """
    Asynchronously compute composite spike curve blending:
    - YouTube Heatmap (weight 0.45)
    - Chat Velocity (weight 0.35)
    - Audio RMS (weight 0.20)

    Returns top 5 peak timestamp windows:
    [{'start_sec': float, 'end_sec': float, 'spike_score': float, 'reason': str}]
    """
    logger.info("⚡ [SPIKE DETECTOR] Starting multi-signal chart spike scan for VOD duration %.1fs...", duration_sec)

    loop = asyncio.get_event_loop()

    # Run signal extractions concurrently in threadpool
    heatmap_task = loop.run_in_executor(None, extract_youtube_heatmap_spikes, url)
    chat_task = loop.run_in_executor(None, extract_chat_density_spikes, url, duration_sec, 15000)
    audio_task = (
        loop.run_in_executor(None, extract_audio_rms_spikes, audio_path, duration_sec)
        if audio_path else asyncio.sleep(0, result=[])
    )

    heatmap_data, chat_data, audio_data = await asyncio.gather(
        heatmap_task, chat_task, audio_task, return_exceptions=True
    )

    if isinstance(heatmap_data, Exception): heatmap_data = []
    if isinstance(chat_data, Exception): chat_data = []
    if isinstance(audio_data, Exception): audio_data = []

    # Map all signals into unified 10-second buckets
    bucket_size = 10.0
    num_buckets = max(1, int(math.ceil(duration_sec / bucket_size)))
    composite = [0.0] * num_buckets

    has_signals = False

    # 1. Heatmap contribution (0.45 weight)
    if heatmap_data:
        has_signals = True
        for pt in heatmap_data:
            idx = min(num_buckets - 1, int(pt["time_sec"] // bucket_size))
            composite[idx] += pt["value"] * 0.45

    # 2. Chat contribution (0.35 weight)
    if chat_data:
        has_signals = True
        for pt in chat_data:
            idx = min(num_buckets - 1, int(pt["time_sec"] // bucket_size))
            composite[idx] += pt["value"] * 0.35

    # 3. Audio RMS contribution (0.20 weight)
    if audio_data:
        has_signals = True
        for pt in audio_data:
            idx = min(num_buckets - 1, int(pt["time_sec"] // bucket_size))
            composite[idx] += pt["value"] * 0.20

    if not has_signals:
        logger.info("  No live chat or heatmap signals available — using default transcript timeline")
        return []

    # Find top peak windows
    peaks = []
    for idx, score in enumerate(composite):
        if score >= 0.20:
            start_sec = max(0.0, idx * bucket_size - 5.0)
            end_sec = min(duration_sec, (idx + 1) * bucket_size + 25.0)
            peaks.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "spike_score": round(score, 3),
                "reason": f"Chart Spike Density Score: {int(score * 100)}%",
            })

    # Sort descending by score & deduplicate overlapping windows
    peaks.sort(key=lambda x: x["spike_score"], reverse=True)
    
    unique_peaks: list[dict[str, Any]] = []
    for p in peaks:
        overlap = any(
            abs(p["start_sec"] - u["start_sec"]) < 30.0 for u in unique_peaks
        )
        if not overlap:
            unique_peaks.append(p)
        if len(unique_peaks) >= 5:
            break

    logger.info("⚡ [SPIKE DETECTOR] Successfully identified %d top chart spike peaks!", len(unique_peaks))
    for pk in unique_peaks:
        logger.info("   • Spike Peak @ %02d:%02d (Score: %.2f)", int(pk["start_sec"] // 60), int(pk["start_sec"] % 60), pk["spike_score"])

    return unique_peaks
