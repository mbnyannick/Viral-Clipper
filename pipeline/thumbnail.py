"""
HD Cover Thumbnail Generator for YouTube Shorts, TikTok & Reels.

Extracts the exact composited video frame at t = 1.8s containing the Aura Word
Asterisk Overlay (*FAIL*, *COOKED*) with 40% white glow and Storytelling Header Card.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from PIL import Image, ImageEnhance

from .score import Moment

logger = logging.getLogger(__name__)

THUMB_W = 1080
THUMB_H = 1920


async def generate_cover_thumbnail(
    clip_path: Path,
    moment: Moment,
    output_path: Path,
    caption_png_path: Path | None = None,
    frame_ts: float = 1.8,
) -> Path:
    """
    Generate a 1080x1920 HD cover thumbnail card for *clip_path*.
    Extracts exact composited frame at t = 1.8s (Aura Word asterisk overlay + header card).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_frame_path = output_path.parent / f"raw_frame_{moment.index:02d}.jpg"

    # Frame timestamp @ 1.8 seconds (when Aura Word asterisk overlay is 100% visible)
    extract_ts = max(0.5, frame_ts)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(extract_ts),
        "-i", str(clip_path),
        "-vframes", "1",
        "-q:v", "2",
        str(raw_frame_path),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10.0)
    except Exception as exc:
        logger.warning("Frame extraction for thumbnail %02d failed: %s", moment.index, exc)

    if not raw_frame_path.exists():
        # Fallback solid dark thumbnail
        img = Image.new("RGB", (THUMB_W, THUMB_H), (15, 15, 20))
        img.save(str(output_path))
        return output_path

    try:
        frame_img = Image.open(raw_frame_path).convert("RGBA")

        # Scale frame to fill 1080x1920 vertical canvas
        fw, fh = frame_img.size
        aspect = THUMB_W / THUMB_H
        frame_aspect = fw / fh

        if frame_aspect > aspect:
            new_h = THUMB_H
            new_w = int(new_h * frame_aspect)
        else:
            new_w = THUMB_W
            new_h = int(new_w / frame_aspect)

        frame_resized = frame_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        crop_x = (new_w - THUMB_W) // 2
        crop_y = (new_h - THUMB_H) // 2
        canvas = frame_resized.crop((crop_x, crop_y, crop_x + THUMB_W, crop_y + THUMB_H))

        # Enhance saturation & contrast for vibrant thumbnail pop
        enhancer = ImageEnhance.Color(canvas)
        canvas = enhancer.enhance(1.15)
        contrast = ImageEnhance.Contrast(canvas)
        canvas = contrast.enhance(1.08)

        # Save clean 1080x1920 HD Cover snapshot showing *WORD* asterisk overlay
        canvas.convert("RGB").save(str(output_path), "JPEG", quality=95)
        logger.info("  Generated Aura Cover Thumbnail: %s (t=%.1fs, 1080x1920)", output_path.name, extract_ts)

    except Exception as exc:
        logger.warning("Thumbnail composite error (%s) — using raw frame fallback", exc)
        raw_frame_path.rename(output_path)
    finally:
        if raw_frame_path.exists():
            try:
                raw_frame_path.unlink()
            except Exception:
                pass

    return output_path
