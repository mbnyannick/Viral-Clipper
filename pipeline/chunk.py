"""
Step 2 — Audio chunking & silence/AFK filtering.

Splits source audio into fixed-duration segments via ffmpeg's segment muxer.
Filters out silent/AFK chunks (mean_volume < -50 dB) to save API transcription costs
and speed up processing on long streams.
"""

import asyncio
import logging
from pathlib import Path

from .errors import PipelineError

logger = logging.getLogger(__name__)


async def _is_chunk_active(path: Path) -> bool:
    """
    Fast local audio volume check using ffmpeg volumedetect.
    Returns False if the chunk is silent/AFK (mean_volume < -50 dB).
    """
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", str(path),
        "-af", "volumedetect",
        "-f", "null", "-",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    output = stderr.decode(errors="replace")
    for line in output.splitlines():
        if "mean_volume:" in line:
            try:
                val = float(line.split("mean_volume:")[1].split("dB")[0].strip())
                if val < -50.0:
                    logger.info("  Skipping quiet/AFK chunk %s (mean_volume: %.1f dB)", path.name, val)
                    return False
            except ValueError:
                pass
    return True


async def chunk_audio(
    audio_path: Path,
    chunk_duration_minutes: int = 3,
    max_duration_hours: float = 5.0,
) -> list[tuple[Path, float]]:
    """
    Split *audio_path* into sequential fixed-duration M4A chunks and filter quiet/AFK sections.
    Caps total stream processing at max_duration_hours (default 5.0 hours).

    Returns
    -------
    List of (chunk_path, start_offset_seconds) tuples for active audio chunks.
    """
    chunk_dir = audio_path.parent / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    chunk_seconds = chunk_duration_minutes * 60
    pattern = str(chunk_dir / "chunk_%03d.m4a")

    logger.info(
        "Splitting audio into %d-minute chunks → %s", chunk_duration_minutes, chunk_dir
    )

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",
        "-reset_timestamps", "1",
        pattern,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise PipelineError("chunk", stderr.decode(errors="replace")[-600:])

    chunk_files = sorted(chunk_dir.glob("chunk_*.m4a"))
    if not chunk_files:
        raise PipelineError("chunk", "ffmpeg produced zero chunk files")

    max_chunks = int((max_duration_hours * 3600) // chunk_seconds)
    if len(chunk_files) > max_chunks:
        logger.info(
            "Stream exceeds %.1f hours (%d chunks) — capping processing to first %.1f hours (%d chunks)",
            max_duration_hours, len(chunk_files), max_duration_hours, max_chunks,
        )
        chunk_files = chunk_files[:max_chunks]

    # Filter out quiet/AFK chunks concurrently
    activity_results = await asyncio.gather(
        *(_is_chunk_active(path) for path in chunk_files)
    )

    result: list[tuple[Path, float]] = []
    skipped_count = 0
    for i, (path, is_active) in enumerate(zip(chunk_files, activity_results)):
        offset = float(i * chunk_seconds)
        if is_active:
            result.append((path, offset))
        else:
            skipped_count += 1

    # Fallback: if all chunks were flagged silent (e.g. ultra quiet mic), keep all chunks
    if not result:
        logger.warning("All chunks flagged silent — retaining all chunks as fallback")
        result = [(path, float(i * chunk_seconds)) for i, path in enumerate(chunk_files)]
    else:
        logger.info(
            "Created %d active audio chunks (%d silent/AFK chunks filtered out)",
            len(result), skipped_count,
        )

    return result
