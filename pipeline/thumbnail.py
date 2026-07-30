"""
HD Cover Thumbnail Generator for YouTube Shorts, TikTok & Reels.

Extracts peak emotion frame from clip, applies color grade + vignette,
and overlays bold title badge to create ready-to-use cover cards.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

from .score import Moment

logger = logging.getLogger(__name__)

THUMB_W = 1080
THUMB_H = 1920


async def generate_cover_thumbnail(
    clip_path: Path,
    moment: Moment,
    output_path: Path,
    caption_png_path: Path | None = None,
) -> Path:
    """
    Generate a 1080x1920 HD cover thumbnail card for *clip_path*.
    Extracts high-emotion frame from 30% mark of clip, overlays title badge & dark vignette.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_frame_path = output_path.parent / f"raw_frame_{moment.index:02d}.jpg"

    # Frame timestamp @ 30% into the clip (usually setup or peak expression)
    frame_ts = max(0.5, (moment.end - moment.start) * 0.35)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(frame_ts),
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
        canvas = enhancer.enhance(1.25)
        contrast = ImageEnhance.Contrast(canvas)
        canvas = contrast.enhance(1.15)

        # Apply dark vignette overlay on top and bottom for text readability
        overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Top dark gradient (y: 0 to 450)
        for y in range(450):
            alpha = int(180 * (1 - y / 450.0))
            draw.line([(0, y), (THUMB_W, y)], fill=(0, 0, 0, alpha))

        # Bottom dark gradient (y: 1470 to 1920)
        for y in range(1470, THUMB_H):
            alpha = int(200 * ((y - 1470) / 450.0))
            draw.line([(0, y), (THUMB_W, y)], fill=(0, 0, 0, alpha))

        canvas = Image.alpha_composite(canvas, overlay)

        # Overlay caption title badge if provided
        if caption_png_path and caption_png_path.exists():
            try:
                cap_img = Image.open(caption_png_path).convert("RGBA")
                # Scale caption badge to 80% of thumbnail width
                target_w = int(THUMB_W * 0.85)
                cw, ch = cap_img.size
                target_h = int(ch * (target_w / cw))
                cap_resized = cap_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

                cap_x = (THUMB_W - target_w) // 2
                cap_y = 160  # Top position below status bar
                canvas.paste(cap_resized, (cap_x, cap_y), cap_resized)
            except Exception as cap_exc:
                logger.warning("Could not overlay caption badge on thumbnail: %s", cap_exc)

        canvas.convert("RGB").save(str(output_path), "JPEG", quality=92)
        logger.info("  Generated HD Cover Thumbnail: %s (1080x1920)", output_path.name)
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
