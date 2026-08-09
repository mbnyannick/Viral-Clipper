"""
Progressive Streaming Pipeline — Fully Parallel 10-minute chunked processing.

Architecture
------------
ALL windows are processed in parallel simultaneously using asyncio.gather.
Total wall-clock time = the slowest single window (not sum of all windows).

Key fix for live streams:
  - yt-dlp --download-sections does NOT seek on live HLS (it always grabs current live).
  - Instead: get the raw HLS URL via yt-dlp --get-url, then use ffmpeg -ss/-t to
    seek to exact time positions within the DVR buffer.
  - For live streams, auto-detect total stream duration and always clip the LAST 60 minutes.

For each window (all running concurrently):
  1. Get HLS URL once, seek with ffmpeg -ss for each window
  2. Transcribe via Groq Whisper
  3. Score top moments via DeepSeek AI
  4. Selective HD clip download (ffmpeg -ss seek) for viral timestamps only
  5. Caption rendering + compositing
  6. Deliver to Telegram immediately as each clip finishes
"""

import asyncio
import html
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from pipeline.caption import render_captions
from pipeline.composite import composite_clips
from pipeline.download import (
    extract_metadata,
    _get_cookie_opts,
    download_video_clip_range,
    YT_CLIENT_CHAINS,
    _kick_vod_get_hls_url,
)
from pipeline.errors import PipelineError
from pipeline.score import Moment, score_moments, _generate_fallback_moments, verify_and_clean_visual_moments
from pipeline.text_utils import mask_profanity
from pipeline import get_public_base_url

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to 'Xh Ym' or 'X mins'."""
    sec = int(seconds)
    if sec <= 0:
        return "Live Stream 🔴"
    hrs = sec // 3600
    mins = (sec % 3600) // 60
    if hrs > 0:
        return f"{hrs}h {mins}m" if mins > 0 else f"{hrs}h"
    return f"{mins} mins" if mins > 0 else f"{sec}s"

_ASSETS_DIR = Path("assets")
_HD_DOWNLOAD_SEM = None
_AUDIO_DOWNLOAD_SEM = None


def _get_hd_sem():
    global _HD_DOWNLOAD_SEM
    if _HD_DOWNLOAD_SEM is None:
        _HD_DOWNLOAD_SEM = asyncio.Semaphore(4)
    return _HD_DOWNLOAD_SEM


def _get_audio_sem():
    global _AUDIO_DOWNLOAD_SEM
    if _AUDIO_DOWNLOAD_SEM is None:
        _AUDIO_DOWNLOAD_SEM = asyncio.Semaphore(3)
    return _AUDIO_DOWNLOAD_SEM


def _is_admin_chat(chat_id: int | str) -> bool:
    op_id = os.environ.get("TELEGRAM_OPERATOR_CHAT_ID", "").strip()
    if not op_id or op_id == "0":
        return True
    return str(chat_id).strip() == op_id


async def _send_safe(bot, chat_id, text: str, parse_mode: str = "") -> None:
    try:
        kwargs = {"chat_id": chat_id, "text": text}
        if parse_mode:
            kwargs["parse_mode"] = parse_mode
        await bot.send_message(**kwargs)
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)


async def _check_audio_volume(audio_path: Path) -> float:
    """Run quick FFmpeg volumedetect filter to check mean audio volume in dB."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-i", str(audio_path),
            "-af", "volumedetect", "-f", "null", "/dev/null",
            stderr=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
        )
        _, stderr = await proc.communicate()
        err_text = stderr.decode(errors="replace")
        for line in err_text.splitlines():
            if "mean_volume:" in line:
                parts = line.split("mean_volume:")
                if len(parts) > 1:
                    val_str = parts[1].replace("dB", "").strip()
                    return float(val_str)
    except Exception as exc:
        logger.warning("Volume detect check failed: %s", exc)
    return -20.0  # Default assume audible if detection fails


async def _send_clip(bot, chat_id, clip_path: Path, caption: str, reply_markup=None) -> None:
    target = clip_path
    if clip_path.exists() and clip_path.stat().st_size > 49 * 1024 * 1024:
        logger.warning("Clip %s (%.1f MB) exceeds 50MB Telegram limit. Auto-compressing...",
                       clip_path.name, clip_path.stat().st_size / (1024 * 1024))
        compressed = clip_path.parent / f"{clip_path.stem}_comp.mp4"
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", str(clip_path),
                "-c:v", "libx264", "-crf", "28", "-preset", "fast",
                "-c:a", "aac", "-b:a", "128k", str(compressed),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=120.0)
            if compressed.exists() and compressed.stat().st_size > 0 and compressed.stat().st_size < clip_path.stat().st_size:
                target = compressed
        except Exception as comp_exc:
            logger.warning("Auto-compression failed: %s", comp_exc)

    try:
        with open(target, "rb") as fh:
            kwargs = {
                "chat_id": chat_id, "video": fh, "caption": caption,
                "parse_mode": "HTML", "supports_streaming": True,
            }
            if reply_markup:
                kwargs["reply_markup"] = reply_markup
            await bot.send_video(**kwargs)
        logger.info("  Delivered: %s", target.name)
    except Exception as exc:
        logger.warning("Failed to deliver %s: %s", target.name, exc)


class _ProgressTracker:
    """
    Maintains a single Telegram message that is edited every 3 seconds with a
    clean, user-friendly progress card showing real-time per-clip granular status.

    Stages tracked:
      1. 📡 Scanning   — audio download (one per window)
      2. 🧠 Analyzing  — transcription + AI scoring
      3. ⬇️ Downloading — HD clip segments for found moments
      4. 🏙️ Compositing — FFmpeg rendering + captions
      5. 🚀 Delivering  — sending to Telegram
    """

    # Per-clip state emojis
    _CLIP_STATES = {
        "pending":      ("⏳", "Pending"),
        "downloading":  ("⬇️", "Downloading HD..."),
        "compositing":  ("🎨", "Compositing..."),
        "delivering":   ("🚀", "Sending to Telegram..."),
        "done":         ("✅", "Done!"),
        "failed":       ("❌", "Failed"),
    }

    def __init__(
        self,
        bot,
        chat_id,
        total_windows: int,
        streamer: str,
        title: str,
        platform: str = "",
        live_status: str = "",
        dur_display: str = "",
        target_clips: int = 10,
    ) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.total = total_windows
        self.streamer = streamer
        self.title = title
        self.platform = platform
        self.live_status = live_status
        self.dur_display = dur_display
        self.target_clips = target_clips

        # Counters (all mutated from async coroutines — single-threaded asyncio, no lock needed)
        self.scanned = 0
        self.analyzed = 0
        self.moments_found = 0
        self.composited = 0
        self.delivered = 0

        # Per-clip state tracking: {clip_index: state_key}
        self._clip_states: dict[int, str] = {}
        self._start_time: float = 0.0
        self._spinner_frames = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]
        self._spinner_idx = 0

        self._msg_id: int | None = None
        self._task: asyncio.Task | None = None
        self._last_edit: float = 0.0
        self._stopped = False

    def set_clip_state(self, clip_idx: int, state: str) -> None:
        """Update per-clip pipeline state. Called from _render_and_deliver_one."""
        self._clip_states[clip_idx] = state

    # ───────────────────────────────────────────────────────────────────────────
    def _bar(self, count: int, total: int, width: int = 10) -> str:
        """Render a simple block progress bar: ▓▓▓▓▓░░░░░"""
        filled = round(count / max(total, 1) * width)
        return "▓" * filled + "░" * (width - filled)

    def _elapsed_str(self) -> str:
        import time
        elapsed = int(time.time() - self._start_time) if self._start_time else 0
        mins, secs = divmod(elapsed, 60)
        return f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"

    def _render(self) -> str:
        import html as _html
        import time
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        spin = self._spinner_frames[self._spinner_idx]

        s_name = _html.escape(self.streamer)
        t_name = (_html.escape(self.title[:45] + ("…" if len(self.title) > 45 else "")))\
            if self.title and self.title not in ("N/A", "") else ""

        plat_str = f"• <b>Platform:</b> {_html.escape(self.platform)}\n" if self.platform else ""
        stat_str = f"• <b>Live Status:</b> {_html.escape(self.live_status)}\n" if self.live_status else ""
        title_str = f"• <b>Title:</b> {t_name}\n" if t_name else ""
        dur_str = f"• <b>Duration:</b> {_html.escape(self.dur_display)}\n" if self.dur_display else ""

        elapsed = self._elapsed_str()

        # ── Phase 1: Scanning & Analysis overview bar ───────────────────────────────────
        scan_bar = self._bar(self.scanned, self.total)
        ai_bar = self._bar(self.analyzed, self.total)

        # ── Phase 2: Per-clip granular status ──────────────────────────────────────────
        clip_lines = ""
        if self._clip_states:
            sorted_clips = sorted(self._clip_states.items())
            for clip_idx, state in sorted_clips:
                emoji, label = self._CLIP_STATES.get(state, ("⏳", state))
                clip_lines += f"  {emoji} <b>Clip #{clip_idx + 1:02d}</b> — {label}\n"

        # ── Current active phase label ────────────────────────────────────────────────
        if self.delivered > 0 and self.delivered >= max(1, self.moments_found):
            phase = f"🎉 <b>Done!</b> {self.delivered} clip{'s' if self.delivered != 1 else ''} delivered!"
        elif self.delivered > 0:
            phase = f"🚀 <b>Delivering</b> — {self.delivered}/{self.moments_found} clips sent {spin}"
        elif self.composited > 0:
            phase = f"🎨 <b>Compositing</b> — Rendering video clips... {spin}"
        elif self.moments_found > 0:
            disp = min(self.moments_found, self.target_clips) if self.target_clips > 0 else self.moments_found
            phase = f"✂️ <b>Extracting</b> — {disp} top moments selected {spin}"
        elif self.analyzed > 0:
            phase = f"🧠 <b>AI Analyzing</b> — Finding viral moments... {spin}"
        else:
            phase = f"📡 <b>Audio Scanning</b> — {self.scanned}/{self.total} windows {spin}"

        card = (
            f"🎦 <b>Live Clipping in Progress</b>\n"
            f"───────────────────\n"
            f"{plat_str}"
            f"{stat_str}"
            f"• <b>Streamer:</b> {s_name}\n"
            f"{title_str}"
            f"{dur_str}"
            f"• <b>Elapsed:</b> ⏱️ {elapsed}\n"
            f"───────────────────\n"
            f"📡 <b>Scanning:</b> <code>{scan_bar}</code> {self.scanned}/{self.total}\n"
            f"🧠 <b>AI Scoring:</b> <code>{ai_bar}</code> {self.analyzed}/{self.total}\n"
        )
        if clip_lines:
            card += (
                f"───────────────────\n"
                f"<b>🎥 Clip Status:</b>\n"
                f"{clip_lines}"
            )
        card += (
            f"───────────────────\n"
            f"⚡️ {phase}"
        )
        return card

    # ───────────────────────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Send the initial progress message and start the edit loop."""
        import time
        self._start_time = time.time()
        try:
            msg = await self.bot.send_message(
                chat_id=self.chat_id,
                text=self._render(),
                parse_mode="HTML",
            )
            self._msg_id = msg.message_id
        except Exception as exc:
            logger.warning("ProgressTracker: failed to send initial message: %s", exc)
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        """Edit progress card every 3 seconds for smooth real-time updates."""
        while not self._stopped:
            await asyncio.sleep(3)
            if self._stopped:
                break
            await self._edit()

    async def _edit(self) -> None:
        if self._msg_id is None:
            return
        try:
            await self.bot.edit_message_text(
                chat_id=self.chat_id,
                message_id=self._msg_id,
                text=self._render(),
                parse_mode="HTML",
            )
        except Exception as exc:
            # "Message is not modified" is expected when nothing changed — silently ignore
            if "not modified" not in str(exc).lower():
                logger.warning("ProgressTracker edit failed: %s", exc)

    async def stop(self, final_text: str = "") -> None:
        """Stop the edit loop and write the final state."""
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()
        if final_text and self._msg_id is not None:
            try:
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=self._msg_id,
                    text=final_text,
                    parse_mode="HTML",
                )
            except Exception as exc:
                logger.warning("ProgressTracker final edit failed: %s", exc)
        elif self._msg_id is not None:
            await self._edit()


async def _get_hls_url(url: str) -> str:
    """
    Resolve the raw HLS .m3u8 URL from a stream page URL.
    - Kick VODs: use Kick API directly (yt-dlp always fails on kick VOD URLs).
    - Everything else: use yt-dlp --get-url (~2s).
    """
    u_lower = url.lower()

    # For YouTube channel handle URLs, append /live so yt-dlp targets the live stream directly
    if ("youtube.com/@" in u_lower or "youtube.com/c/" in u_lower) and "/live" not in u_lower and "/watch" not in u_lower and "/videos" not in u_lower:
        url = url.rstrip("/") + "/live"
        u_lower = url.lower()

    # ── Kick VOD fast path — bypass yt-dlp entirely ──────────────────────────
    if "kick.com" in u_lower and "/videos/" in u_lower:
        hls_url, _, _, _ = await _kick_vod_get_hls_url(url)
        logger.info("Kick VOD HLS resolved via API: %s...", hls_url[:80])
        return hls_url

    is_youtube = "youtu" in u_lower
    impersonate_opts = []
    cookie_opts = _get_cookie_opts()
    yt_client_chains = YT_CLIENT_CHAINS if is_youtube else [[]]

    last_exc = None
    for yt_opts in yt_client_chains:
        cmd = [
            "yt-dlp",
            "--no-playlist",
            *cookie_opts,
            *impersonate_opts,
            *yt_opts,
            "-f", "b/best",
            "-g",
            url,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            lines = [l.strip() for l in stdout.decode().strip().splitlines() if l.strip()]
            if lines:
                hls_url = lines[-1]
                logger.info("Resolved HLS URL: %s...", hls_url[:80])
                return hls_url
        last_exc = PipelineError("get_hls_url", stderr.decode(errors="replace")[-400:])

    if last_exc:
        raise last_exc
    raise PipelineError("get_hls_url", "yt-dlp returned no URL")


async def _get_stream_duration(hls_url: str, stream_url: str = "") -> float:
    """
    Get total stream DVR duration in seconds.

    For VODs: ffprobe reads duration directly from the file/playlist.
    For live streams: ffprobe returns 0 (live HLS has no duration header).
    Fallback: use the stream's Unix start timestamp from yt-dlp metadata to
    calculate how many seconds the stream has been running.
    """
    # Try ffprobe first (works for VODs)
    cmd = [
        "ffprobe", "-v", "quiet",
        "-user_agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-print_format", "json",
        "-show_format",
        hls_url,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    try:
        import json as _json
        data = _json.loads(stdout.decode())
        dur = float(data.get("format", {}).get("duration", 0))
        if dur > 0:
            logger.info("Stream duration from ffprobe: %.0fs (%.1fh)", dur, dur / 3600)
            return dur
    except Exception:
        pass

    # Fallback for live streams: calculate elapsed time from stream start timestamp
    if stream_url:
        try:
            u_lower = stream_url.lower()
            is_youtube = "youtu" in u_lower
            is_kick = "kick.com" in u_lower
            impersonate_opts = ["--impersonate", "chrome"] if is_kick else []
            cookie_opts = _get_cookie_opts()
            yt_client_chains = YT_CLIENT_CHAINS if is_youtube else [[]]

            for yt_opts in yt_client_chains:
                proc2 = await asyncio.create_subprocess_exec(
                    "yt-dlp", "--no-playlist", *cookie_opts, *impersonate_opts, *yt_opts,
                    "--print", "%(timestamp)s",
                    stream_url,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout2, _ = await proc2.communicate()
                ts_str = stdout2.decode().strip()
                if ts_str and ts_str.isdigit():
                    import time as _time
                    stream_start_ts = int(ts_str)
                    elapsed = _time.time() - stream_start_ts
                    if elapsed > 0:
                        logger.info(
                            "Stream duration from timestamp: %.0fs (%.1fh) — started %s ago",
                            elapsed, elapsed / 3600,
                            f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m",
                        )
                        return elapsed
        except Exception as exc:
            logger.warning("Timestamp fallback failed: %s", exc)

    logger.warning("Could not determine stream duration — defaulting to 3600s")
    return 3600.0


async def _download_full_audio(
    hls_url: str,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> Path:
    """
    Download the full scan-range audio ONCE so windows can be sliced locally.
    (The old approach re-ran ffmpeg output-seek per window, forcing HLS to
    re-download every segment from the start of the DVR playlist each time.)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_sec = max(1.0, end_sec - start_sec)
    cmd = [
        "ffmpeg", "-y",
        "-user_agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-ss", str(start_sec),        # input seek — skips DVR segments instead of re-downloading from start
        "-i", hls_url,                # HLS playlist URL
        "-t", str(duration_sec),      # capture duration
        "-vn",                        # audio only — no video track
        "-acodec", "aac",
        "-b:a", "64k",                # low bitrate — speech optimization
        "-f", "mp4",
        str(output_path),
    ]
    logger.info(
        "Downloading full scan-range audio once (%.0fs–%.0fs, %.0fs)...",
        start_sec, end_sec, duration_sec,
    )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise PipelineError(
            "full_audio",
            f"{output_path.name}: {stderr.decode(errors='replace')[-400:]}",
        )
    return output_path


async def _slice_audio_window(
    full_audio: Path,
    start_sec: float,
    duration_sec: float,
    output_path: Path,
) -> Path:
    """Slice a window out of an already-downloaded local audio file (fast seek)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),        # input seek — instant on local file
        "-i", str(full_audio),
        "-t", str(duration_sec),
        "-vn",
        "-acodec", "aac",
        "-b:a", "64k",
        "-f", "mp4",
        str(output_path),
    ]
    logger.info(
        "Local slice: %.0fs–%.0fs → %s",
        start_sec, start_sec + duration_sec, output_path.name,
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise PipelineError(
            "audio_window",
            f"{output_path.name}: {stderr.decode(errors='replace')[-400:]}",
        )
    return output_path


async def _download_audio_window(
    hls_url: str,
    start_sec: float,
    duration_sec: float,
    output_path: Path,
) -> Path:
    """
    Download a specific time window from an HLS stream using fast output seek.
    (Fallback path used when full-audio pre-download is unavailable.)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-user_agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-i", hls_url,                # HLS playlist URL
        "-ss", str(start_sec),        # seek position
        "-t", str(duration_sec),      # capture duration
        "-vn",                        # audio only — no video track
        "-acodec", "aac",
        "-b:a", "64k",                # low bitrate — speech optimization
        "-f", "mp4",
        str(output_path),
    ]

    logger.info(
        "ffmpeg seek: %.0fs–%.0fs → %s",
        start_sec, start_sec + duration_sec, output_path.name,
    )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise PipelineError(
            "audio_window",
            f"{output_path.name}: {stderr.decode(errors='replace')[-400:]}",
        )
    return output_path


async def _download_hd_clip_from_hls(
    hls_url: str,
    start_sec: float,
    end_sec: float,
    output_path: Path,
    stream_url: str = "",
) -> Path:
    """
    Download a specific clip range at high quality directly from the HLS stream.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_sec - start_sec

    cmd = [
        "ffmpeg", "-y",
        "-user_agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-ss", str(start_sec),        # input seek — skips DVR segments instead of re-downloading from start
        "-i", hls_url,
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "veryfast",
        "-crf", "23",
        "-maxrate", "8000k",
        "-bufsize", "16000k",
        str(output_path),
    ]

    logger.info(
        "HD clip seek: %.1fs–%.1fs → %s",
        start_sec, end_sec, output_path.name,
    )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise PipelineError(
            "hd_clip",
            f"{output_path.name}: {stderr.decode(errors='replace')[-400:]}",
        )
    return output_path


async def _scan_and_score_window(
    *,
    hls_url: str,
    window_index: int,
    total_windows: int,
    window_start: float,
    window_duration: float,
    window_dir: Path,
    deepseek_api_key: str,
    deepseek_model: str,
    streamer: str,
    video_title: str,
    tracker: "_ProgressTracker",
    full_audio: Path | None = None,
) -> dict:
    """
    Phase 1: Audio-scan window, transcribe with Deepgram, and score candidate moments.
    Fast parallel execution across the full VOD timeline.
    """
    label = (
        f"Window {window_index + 1}/{total_windows} "
        f"({int(window_start // 60)}m–{int((window_start + window_duration) // 60)}m)"
    )
    logger.info("=== [PARALLEL SCAN] Starting %s ===", label)
    window_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Audio Download ──────────────────────────────────────────────────────────
    audio_path = window_dir / "audio.m4a"
    audio_sem = _get_audio_sem()
    async with audio_sem:
        try:
            if full_audio is not None and full_audio.exists() and full_audio.stat().st_size >= 4096:
                await _slice_audio_window(
                    full_audio=full_audio,
                    start_sec=window_start,
                    duration_sec=window_duration,
                    output_path=audio_path,
                )
            else:
                await _download_audio_window(
                    hls_url=hls_url,
                    start_sec=window_start,
                    duration_sec=window_duration,
                    output_path=audio_path,
                )
        except Exception as exc:
            logger.warning("%s — audio download failed (%s). Skipping.", label, exc)
            tracker.scanned += 1
            return {"window_dir": window_dir, "segments": [], "moments": []}

    if not audio_path.exists() or audio_path.stat().st_size < 4096:
        logger.warning("%s — audio missing/empty. Skipping.", label)
        tracker.scanned += 1
        return {"window_dir": window_dir, "segments": [], "moments": []}

    # Audio volume pre-filter: skip silent/dead-static windows in 0.05s without API costs
    mean_vol = await _check_audio_volume(audio_path)
    if mean_vol < -50.0:
        logger.info("%s — Audio window is silent (mean volume %.1f dB < -50 dB). Skipping APIs.", label, mean_vol)
        tracker.scanned += 1
        tracker.analyzed += 1
        return {"window_dir": window_dir, "segments": [], "moments": []}

    tracker.scanned += 1  # 📡 Scanned counter

    # ── 2. Dual-Engine Transcription (Deepgram Nova-2 + Groq Whisper Fallback) ──────
    try:
        from pipeline.transcribe import transcribe_chunks
        segments = await transcribe_chunks([(audio_path, window_start)])
    except Exception as exc:
        logger.warning("%s — Transcription failed: %s", label, exc)
        tracker.analyzed += 1
        return {"window_dir": window_dir, "segments": [], "moments": []}

    if not segments:
        logger.info("%s — quiet window.", label)
        tracker.analyzed += 1
        return {"window_dir": window_dir, "segments": [], "moments": []}

    # ── 3. Moment Scoring with Fallback Guarantee ──────────────────────────────────
    try:
        moments = await score_moments(
            segments=segments,
            api_key=deepseek_api_key,
            top_n=1,
            model=deepseek_model,
            streamer=streamer,
            video_title=video_title,
        )
    except Exception as exc:
        logger.warning("%s — scoring failed (%s). Using speech density fallback.", label, exc)
        moments = _generate_fallback_moments(segments, top_n=1, streamer=streamer)

    if not moments:
        moments = _generate_fallback_moments(segments, top_n=1, streamer=streamer)

    # ── 3.5 Visual Overlay & Subscribe-Button Avoidance Scan ────────────────────
    moments = verify_and_clean_visual_moments(moments, video_path=audio_path)

    tracker.analyzed += 1  # 🧠 Analyzed counter

    # Duration clamping (20s - 60s)
    MIN_DUR, MAX_DUR = 20.0, 60.0
    clamped = []
    for m in moments:
        start = m.start
        end = m.end
        dur = end - start
        if dur < MIN_DUR:
            end = start + MIN_DUR
        elif dur > MAX_DUR:
            end = start + MAX_DUR
        if end > window_start + window_duration:
            end = window_start + window_duration
            start = max(window_start, end - MIN_DUR)
        clamped.append(Moment(
            index=m.index, start=start, end=end,
            caption_lines=m.caption_lines, emoji=m.emoji,
            score=getattr(m, "score", 85),
            reasoning=getattr(m, "reasoning", ""),
            title=getattr(m, "title", ""),
            bgm_track=getattr(m, "bgm_track", "none"),
            sfx_events=getattr(m, "sfx_events", []),
        ))

    tracker.moments_found += len(clamped)
    return {"window_dir": window_dir, "segments": segments, "moments": clamped}


async def run_streaming_pipeline(
    url: str,
    bot,
    chat_id,
    run_dir: Path,
    layout_mode: str = "pillarbox",
    stream_start_sec: float | None = None,
    stream_end_sec: float | None = None,
    chunk_minutes: int = 10,
    target_total_clips: int = 3,
    campaign_brief: str = "",
    target_duration: str = "auto",
    enable_subtitles: bool = True,
) -> None:
    """
    Global Top-N Selection Pipeline:
    1. Quick parallel audio-scan across full VOD timeline.
    2. Rank all candidate moments globally and pick ONLY the top target_total_clips (e.g., 3 clips total).
    3. Render HD video, MediaPipe face tracking, and deliver clips to Telegram immediately as each finishes.
    """
    deepseek_api_key = os.environ["DEEPSEEK_API_KEY"]
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    clip_window_minutes = int(os.getenv("CLIP_WINDOW_MINUTES", "60"))

    chunk_sec = chunk_minutes * 60.0
    watermark_path = _ASSETS_DIR / "watermark.png"

    # ── Step 1: Resolve HLS URL ──────────────────────────────────────────────
    try:
        hls_url = await _get_hls_url(url)
    except PipelineError as exc:
        await _send_safe(bot, chat_id, f"❌ Could not resolve stream URL: {exc.reason}")
        return

    # ── Step 2: Probe DVR duration ────────────────────────────────────────────
    dvr_duration = await _get_stream_duration(hls_url, stream_url=url)

    if stream_end_sec is None:
        stream_end_sec = dvr_duration if dvr_duration > 0 else chunk_sec * 6
    if stream_start_sec is None:
        stream_start_sec = max(0.0, stream_end_sec - clip_window_minutes * 60.0)

    # ── Step 3: Extract Metadata ──────────────────────────────────────────────
    streamer_info = await extract_metadata(url)
    streamer = streamer_info.get("streamer", "Streamer")
    video_title = streamer_info.get("title", "")

    # Build windows
    windows = []
    t = stream_start_sec
    while t < stream_end_sec:
        windows.append(t)
        t += chunk_sec

    total = len(windows)
    total_mins = int((stream_end_sec - stream_start_sec) / 60)

    logger.info(
        "PARALLEL pipeline: %d windows × %d min | Target Clips: %d total",
        total, chunk_minutes, target_total_clips,
    )

    u_lower = url.lower()
    is_live_now = (dvr_duration <= 0 or ("kick.com" in u_lower and not ("/videos/" in u_lower or "clips" in u_lower)))
    platform_name = "Kick 🟩" if "kick.com" in u_lower else ("Twitch 🟣" if "twitch.tv" in u_lower else ("YouTube ▶️" if "youtu" in u_lower else "Video 🌐"))
    live_stat = "🔴 LIVE NOW" if is_live_now else "📁 Recorded VOD"
    dur_str = "Live Stream 🔴" if is_live_now else (format_duration(stream_end_sec - stream_start_sec) if stream_end_sec else "")

    tracker = _ProgressTracker(
        bot=bot,
        chat_id=chat_id,
        total_windows=total,
        streamer=streamer,
        title=video_title,
        platform=platform_name,
        live_status=live_stat,
        dur_display=dur_str,
        target_clips=target_total_clips,
    )
    await tracker.start()

    # ── Phase 0: Download full scan-range audio ONCE (huge speedup vs per-window HLS seek) ──
    full_audio = run_dir / "full_audio.m4a"
    try:
        await _download_full_audio(
            hls_url=hls_url,
            start_sec=stream_start_sec,
            end_sec=stream_end_sec,
            output_path=full_audio,
        )
        logger.info("Full scan-range audio ready: %s (%.1f MB)", full_audio.name, full_audio.stat().st_size / 1e6)
    except Exception as full_exc:
        logger.warning("Full-audio pre-download failed (%s) — falling back to per-window HLS seeks.", full_exc)
        full_audio = None

    # ── Phase 1: Parallel Audio Scan & AI Moment Extraction ────────────────────
    scan_tasks = [
        _scan_and_score_window(
            hls_url=hls_url,
            window_index=i,
            total_windows=total,
            window_start=w_start,
            window_duration=min(chunk_sec, stream_end_sec - w_start),
            window_dir=run_dir / f"window_{i:03d}",
            deepseek_api_key=deepseek_api_key,
            deepseek_model=deepseek_model,
            streamer=streamer,
            video_title=video_title,
            tracker=tracker,
            full_audio=full_audio,
        )
        for i, w_start in enumerate(windows)
    ]

    scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)

    # ── Phase 2: Global Ranking & Slicing Top N Clips ─────────────────────────
    all_candidate_moments = []
    window_segments_map = {}

    for r in scan_results:
        if isinstance(r, dict) and r.get("moments"):
            for m in r["moments"]:
                all_candidate_moments.append((m, r["window_dir"], r["segments"]))
                window_segments_map[id(m)] = r["segments"]
        elif isinstance(r, Exception):
            logger.warning("Scan window exception: %s", r)

    if not all_candidate_moments:
        await tracker.stop("⚠️ No viral moments found in this stream/VOD.")
        return

    # Sort all candidate moments globally by virality score descending
    all_candidate_moments.sort(key=lambda item: getattr(item[0], "score", 85), reverse=True)

    # Slice ONLY the top target_total_clips (e.g. exactly 3 clips total for full 5h stream)
    top_selected = all_candidate_moments[:target_total_clips]

    # Re-assign clean 0-indexed indices for final delivery
    final_moments_to_render = []
    for idx, (m, wdir, segs) in enumerate(top_selected):
        new_m = Moment(
            index=idx, start=m.start, end=m.end,
            caption_lines=m.caption_lines, emoji=m.emoji,
            score=getattr(m, "score", 90),
            reasoning=getattr(m, "reasoning", ""),
            title=getattr(m, "title", ""),
            bgm_track=getattr(m, "bgm_track", "none"),
            sfx_events=getattr(m, "sfx_events", []),
        )
        final_moments_to_render.append((new_m, wdir, segs))

    logger.info("Global Top-N selection: Picked top %d moments across %d windows", len(final_moments_to_render), total)

    # ── Phase 3: Targeted HD Download, Compositing & Immediate Delivery ─────────
    hd_sem = _get_hd_sem()
    delivered_count = [0]

    async def _render_and_deliver_one(moment: Moment, wdir: Path, segments: list[dict]):
        clip_out = wdir / f"clip_{moment.index:03d}.mp4"
        dl_start = max(0.0, moment.start - 1.0)
        dl_end = moment.end + 1.0

        async with hd_sem:
            try:
                tracker.set_clip_state(moment.index, "downloading")
                await _download_hd_clip_from_hls(
                    hls_url=hls_url, start_sec=dl_start, end_sec=dl_end, output_path=clip_out, stream_url=url,
                )
            except Exception as exc:
                logger.warning("HD clip download failed idx %d: %s", moment.index, exc)
                tracker.set_clip_state(moment.index, "failed")
                return

        captions_dir = wdir / "captions"
        finals_dir = wdir / "finals"

        tracker.set_clip_state(moment.index, "compositing")
        captions = await asyncio.get_event_loop().run_in_executor(
            None, render_captions, [moment], _ASSETS_DIR, captions_dir, layout_mode,
        )

        async def _on_clip_done(final_path: Path, m: Moment) -> None:
            tracker.set_clip_state(m.index, "delivering")
            clip_num = m.index + 1
            mins = int(m.start // 60)
            secs = int(m.start % 60)
            caption_title = mask_profanity(" / ".join(m.caption_lines)) if m.caption_lines else f"Clip {clip_num}"
            emoji = getattr(m, "emoji", "🔥")
            clean_streamer = streamer if streamer else "Streamer"
            tag = "#" + re.sub(r"[^\w]", "", clean_streamer)

            video_caption = (
                f"🎬 <b>Clip {clip_num:02d}</b> • {clean_streamer} [{mins}m{secs:02d}s]\n"
                f"<i>{html.escape(caption_title)} {emoji}</i>"
            )
            # Generate HD Cover Thumbnail capturing Aura Word (*FAIL*, *COOKED*) @ t=1.8s
            try:
                from pipeline.thumbnail import generate_cover_thumbnail
                thumbs_dir = wdir / "thumbnails"
                thumb_out = thumbs_dir / f"thumbnail_{m.index:02d}.jpg"
                cap_png = captions[0] if captions else None
                await generate_cover_thumbnail(final_path, m, thumb_out, cap_png, frame_ts=1.8)
                
                clips_public_dir = Path("tmp/clips")
                clips_public_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(thumb_out, clips_public_dir / f"thumbnail_{m.index:02d}.jpg")
                logger.info("  Generated Aura Cover Thumbnail for Stream Clip #%02d (t=1.8s)", clip_num)
            except Exception as thumb_exc:
                logger.warning("Thumbnail generation warning for stream clip #%02d: %s", clip_num, thumb_exc)

            if _is_admin_chat(chat_id):
                action_keyboard = InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("⚡ Post to ALL Now", callback_data=f"post:all:{clip_num}"),
                        InlineKeyboardButton("📅 Schedule Peak Time", callback_data=f"sched:all:{clip_num}"),
                    ],
                    [
                        InlineKeyboardButton("📱 TikTok", callback_data=f"post:tiktok:{clip_num}"),
                        InlineKeyboardButton("🔴 YouTube", callback_data=f"post:youtube:{clip_num}"),
                        InlineKeyboardButton("📸 IG Reels", callback_data=f"post:instagram:{clip_num}"),
                        InlineKeyboardButton("📘 Facebook", callback_data=f"post:facebook:{clip_num}"),
                    ],
                    [
                        InlineKeyboardButton("🗑️ Discard Clip", callback_data=f"discard:{clip_num}"),
                    ],
                ])
            else:
                action_keyboard = None
            await _send_clip(bot, chat_id, final_path, video_caption, reply_markup=action_keyboard)

            raw_yt = getattr(m, "title", caption_title)
            yt_title = html.escape(mask_profanity(raw_yt)) + f" {emoji} {tag} #Shorts #Viral"
            tt_title = html.escape(caption_title) + f" {emoji} {tag} #viral #fyp #streamer #highlights"
            ig_title = html.escape(caption_title) + f" {emoji} {tag} #reels #viral #explorepage #trending"
            fb_title = html.escape(caption_title) + f" {emoji} {tag} #facebookreels #viral #facebook"

            if not _is_admin_chat(chat_id):
                card_text = (
                    f"📌 <b>Clip {clip_num:02d} Tap-To-Copy Metadata</b>\n\n"
                    f"🔴 <b>YouTube Shorts Title:</b>\n<code>{yt_title}</code>\n\n"
                    f"🎵 <b>TikTok Caption:</b>\n<code>{tt_title}</code>\n\n"
                    f"📸 <b>Instagram Reels Caption:</b>\n<code>{ig_title}</code>\n\n"
                    f"📘 <b>Facebook Reels Caption:</b>\n<code>{fb_title}</code>"
                )
                await _send_safe(bot, chat_id, card_text, parse_mode="HTML")

            delivered_count[0] += 1
            tracker.delivered += 1
            tracker.composited += 1
            tracker.set_clip_state(m.index, "done")

        await composite_clips(
            clips=[clip_out], captions=captions, watermark_path=watermark_path,
            moments=[moment], output_dir=finals_dir, layout_mode=layout_mode,
            segments=segments, enable_subtitles=enable_subtitles, on_clip_ready=_on_clip_done,
        )

    render_tasks = [
        _render_and_deliver_one(m, wdir, segs) for m, wdir, segs in final_moments_to_render
    ]
    await asyncio.gather(*render_tasks, return_exceptions=True)

    total_delivered = delivered_count[0]
    scope = "full VOD" if stream_start_sec == 0.0 else f"last {total_mins} minutes"

    if total_delivered == 0:
        await tracker.stop("⚠️ Could not render target clips from this stream.")
        return

    # ── Build pending schedule session for "Schedule All" confirmation ──────
    try:
        from bot.handlers import _pending_schedule_sessions
        clips_public_dir = Path("tmp/clips")
        clips_public_dir.mkdir(parents=True, exist_ok=True)
        session_clips = []
        for idx, (m, wdir, segs) in enumerate(final_moments_to_render, start=1):
            clip_path = wdir / f"clip_{m.index:03d}.mp4"
            if not clip_path.exists():
                clip_path = wdir / "finals" / f"final_00.mp4"
            if not clip_path.exists():
                continue

            safe_fid = "".join(c for c in clip_path.name if c.isalnum())[:20]
            public_filename = f"clip_{safe_fid}.mp4"
            public_path = clips_public_dir / public_filename
            try:
                if not public_path.exists():
                    shutil.copy2(clip_path, public_path)
                video_url = f"{get_public_base_url()}/clips/{public_filename}"
            except Exception:
                video_url = ""

            raw_title = getattr(m, "title", "") or getattr(m, "caption_lines", [""])[0]
            hashtags = getattr(m, "hashtags", "") or ""
            payload_base = {
                "clip_id": f"clip_{idx:03d}",
                "title": raw_title,
                "caption": raw_title,
                "description": raw_title,
                "video_url": video_url,
                "thumbnail_url": f"{get_public_base_url()}/clips/thumbnail_{idx-1:02d}.jpg",
                "cover_timestamp_ms": 1800,
                "video_filename": f"clip_{idx:03d}.mp4",
                "mime_type": "video/mp4",
                "hashtags": hashtags,
                "chat_id": chat_id,
            }
            session_clips.append({"clip_num": idx, "payload": payload_base})

        if session_clips:
            _pending_schedule_sessions[chat_id] = session_clips
            sched_all_kb = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("📅 Schedule All Clips", callback_data=f"sched_all:confirm:{len(session_clips)}"),
                    InlineKeyboardButton("⏭️ Skip Auto-Schedule", callback_data=f"sched_all:skip:{len(session_clips)}"),
                ]
            ])
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"📅 <b>Peak-Hour Auto-Scheduler</b>\n\n"
                    f"Would you like to auto-schedule all <b>{len(session_clips)} clips</b> across peak viral hours on TikTok, YouTube Shorts, IG Reels, and Facebook Reels?"
                ),
                parse_mode="HTML",
                reply_markup=sched_all_kb,
            )
    except Exception as sched_session_exc:
        logger.warning("Could not build schedule session card: %s", sched_session_exc)

    final_card = (
        f"✅ <b>Processing Complete!</b>\n"
        f"• Delivered {total_delivered} clip(s) from {streamer}'s {scope}.\n"
        f"• Enjoy your clips below!"
    )
    await tracker.stop(final_card)

