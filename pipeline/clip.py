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


async def _cut_one(source: Path, moment: Moment, output_dir: Path, url: str = "") -> Path:
    """Cut a single clip from *source* using ffmpeg or fetch HD segment range via yt-dlp."""
    out_path = output_dir / f"clip_{moment.index:02d}.mp4"
    dur = moment.end - moment.start

    if source.exists() and source.suffix.lower() == ".mp4":
        # Fast, frame-accurate clip extraction via ultrafast re-encode
        # This prevents the lip-sync drift caused by `-c copy` snapping to keyframes
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-ss", str(moment.start),
            "-i", str(source),
            "-t", str(dur),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            str(out_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        # Fallback to ultrafast re-encode if stream copy produced corrupt/empty file
        if not out_path.exists() or out_path.stat().st_size < 1000:
            logger.warning("Stream copy failed for clip %02d — falling back to fast re-encode", moment.index)
            proc_fallback = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-ss", str(moment.start),
                "-i", str(source),
                "-t", str(dur),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
                "-c:a", "aac",
                str(out_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc_fallback.communicate()
    else:
        target_url = url if url else str(source)
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
