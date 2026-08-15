"""
Step 5 — Clip cutting.

For each identified moment, runs ffmpeg to cut that time range from the
single locally-downloaded source file. All cuts run *concurrently* via
asyncio.gather — each reads a different byte range of the same file,
so there is no contention.

Using -ss / -to before -i (input seeking) is faster than output seeking
for large files; -c copy avoids re-encoding (instant, lossless cuts).
"""

import asyncio
import logging
from pathlib import Path

from .errors import PipelineError
from .score import Moment

logger = logging.getLogger(__name__)


from .download import download_video_clip_range


import subprocess


def _is_valid_mp4(p: Path) -> bool:
    """Verify that p exists, has non-zero size, and has a valid moov atom readable by ffprobe."""
    if not p.exists() or p.stat().st_size < 1000:
        return False
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
            capture_output=True, text=True, timeout=5
        )
        return res.returncode == 0 and bool(res.stdout.strip())
    except Exception:
        return False


async def _cut_one(source: Path, moment: Moment, output_dir: Path, url: str = "") -> Path:
    """Cut a single clip from *source* using ffmpeg or fetch HD segment range via yt-dlp."""
    out_path = output_dir / f"clip_{moment.index:02d}.mp4"
    dur = moment.end - moment.start

    if source.exists() and source.suffix.lower() == ".mp4":
        # 1. First attempt: Precision cut with near-lossless master quality (CRF 14, no generation loss)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-ss", str(moment.start),
            "-i", str(source),
            "-t", str(dur),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "14",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "320k",
            "-avoid_negative_ts", "make_zero",
            str(out_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        if not _is_valid_mp4(out_path):
            logger.warning("Fast clip extraction produced invalid MP4 for clip %02d — retrying with safety copy", moment.index)
            if out_path.exists():
                try:
                    out_path.unlink(missing_ok=True)
                except Exception:
                    pass
            proc_fallback = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-ss", str(moment.start),
                "-i", str(source),
                "-t", str(dur),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "16",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "320k",
                "-avoid_negative_ts", "make_zero",
                str(out_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc_fallback.communicate()
    else:
        target_url = url if url else str(source)
        await download_video_clip_range(target_url, moment.start, moment.end, out_path)

    if not _is_valid_mp4(out_path):
        target_url = url if url else str(source)
        logger.warning("Source cut invalid for clip %02d — falling back to direct range stream download", moment.index)
        await download_video_clip_range(target_url, moment.start, moment.end, out_path)

    if not out_path.exists():
        raise PipelineError("clip", f"clip_{moment.index:02d}.mp4 was not produced")

    logger.info(
        "  Cut clip %02d: %.1fs–%.1fs (%.1fs) → %s",
        moment.index,
        moment.start,
        moment.end,
        moment.duration,
        out_path.name,
    )
    return out_path


_CUT_SEMAPHORE = asyncio.Semaphore(4)


async def _cut_one_safe(source: Path, moment: Moment, output_dir: Path, url: str = "") -> Path:
    async with _CUT_SEMAPHORE:
        return await _cut_one(source, moment, output_dir, url=url)


async def cut_clips(
    source: Path,
    moments: list[Moment],
    output_dir: Path,
    url: str = "",
) -> list[Path]:
    """
    Cut all moments concurrently (max 4 parallel FFmpeg tasks) from *source* or fetch HD segments.

    Returns a list of clip paths in the same order as *moments*.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Cutting %d clips from %s", len(moments), source.name)

    paths: list[Path] = await asyncio.gather(
        *(_cut_one_safe(source, m, output_dir, url=url) for m in moments)
    )
    return list(paths)
