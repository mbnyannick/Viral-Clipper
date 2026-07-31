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
import logging
import os
import subprocess
from pathlib import Path

from pipeline.caption import render_captions
from pipeline.composite import composite_clips
from pipeline.download import extract_metadata, _get_cookie_opts, download_video_clip_range, YT_CLIENT_CHAINS, _kick_vod_get_hls_url
from pipeline.errors import PipelineError
from pipeline.score import Moment, score_moments

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path("assets")
_HD_DOWNLOAD_SEM = None


def _get_hd_sem():
    global _HD_DOWNLOAD_SEM
    if _HD_DOWNLOAD_SEM is None:
        _HD_DOWNLOAD_SEM = asyncio.Semaphore(4)
    return _HD_DOWNLOAD_SEM


async def _send_safe(bot, chat_id, text: str, parse_mode: str = "") -> None:
    try:
        kwargs = {"chat_id": chat_id, "text": text}
        if parse_mode:
            kwargs["parse_mode"] = parse_mode
        await bot.send_message(**kwargs)
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)


async def _send_clip(bot, chat_id, clip_path: Path, caption: str) -> None:
    try:
        with open(clip_path, "rb") as fh:
            await bot.send_video(
                chat_id=chat_id, video=fh, caption=caption, supports_streaming=True,
            )
        logger.info("  Delivered: %s", clip_path.name)
    except Exception as exc:
        logger.warning("Failed to deliver %s: %s", clip_path.name, exc)


class _ProgressTracker:
    """
    Maintains a single Telegram message that is edited every 3 seconds with a
    clean, user-friendly progress card. No technical details exposed.

    Stages tracked:
      1. 📡 Scanning   — audio download (one per window)
      2. 🧠 Analyzing  — transcription + AI scoring
      3. ⬇️ Downloading — HD clip segments for found moments
      4. 🏞️ Finishing  — compositing + rendering
    """

    def __init__(self, bot, chat_id, total_windows: int, streamer: str, title: str) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.total = total_windows
        self.streamer = streamer
        self.title = title

        # Counters (all mutated from async coroutines — single-threaded asyncio, no lock needed)
        self.scanned = 0
        self.analyzed = 0
        self.moments_found = 0
        self.composited = 0
        self.delivered = 0

        self._msg_id: int | None = None
        self._task: asyncio.Task | None = None
        self._last_edit: float = 0.0
        self._stopped = False

    # ───────────────────────────────────────────────────────────────────────────
    def _bar(self, count: int, total: int, width: int = 10) -> str:
        """Render a simple block progress bar: ▓▓▓▓▓░░░░░"""
        filled = round(count / max(total, 1) * width)
        return "▓" * filled + "░" * (width - filled)

    def _render(self) -> str:
        n = self.total
        title_part = f" • {self.title[:28]}…" if self.title and len(self.title) > 3 else ""

        def _stage(icon: str, label: str, done: int, total: int, override: str = "") -> str:
            if override:
                return f"{icon} {label:<12} {override}"
            pct = done / max(total, 1)
            bar = self._bar(done, total)
            if done >= total:
                return f"{icon} {label:<12} {bar}  {done}/{total} ✅"
            return f"{icon} {label:<12} {bar}  {done}/{total}"

        # Stage 3: downloading/compositing
        if self.moments_found > 0:
            dl_override = f"🎯 {self.moments_found} moment{'s' if self.moments_found != 1 else ''} found"
        else:
            dl_override = "Waiting…" if self.analyzed < n else "None found"

        finish_override = (
            f"{self.composited} clip{'s' if self.composited != 1 else ''} ready"
            if self.composited > 0 else "Waiting…"
        )

        lines = [
            f"🎬 *Processing {self.streamer}*{title_part}",
            "",
            _stage("📡", "Scanning", self.scanned, n),
            _stage("🧠", "Analyzing", self.analyzed, n),
            f"⬇️ {'Downloading':<12} {dl_override}",
            f"🏞️ {'Finishing':<12} {finish_override}",
        ]
        return "\n".join(lines)

    # ───────────────────────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Send the initial progress message and start the edit loop."""
        try:
            msg = await self.bot.send_message(
                chat_id=self.chat_id,
                text=self._render(),
                parse_mode="Markdown",
            )
            self._msg_id = msg.message_id
        except Exception as exc:
            logger.warning("ProgressTracker: failed to send initial message: %s", exc)
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        """Edit the progress message every 3 seconds until stopped."""
        import time
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
                parse_mode="Markdown",
            )
        except Exception as exc:
            # "Message is not modified" is expected when nothing changed — silently ignore
            if "not modified" not in str(exc).lower():
                logger.debug("ProgressTracker edit failed: %s", exc)

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
                    parse_mode="Markdown",
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

    # ── Kick VOD fast path — bypass yt-dlp entirely ──────────────────────────
    if "kick.com" in u_lower and "/videos/" in u_lower:
        hls_url, _, _, _ = await _kick_vod_get_hls_url(url)
        logger.info("Kick VOD HLS resolved via API: %s...", hls_url[:80])
        return hls_url

    is_youtube = "youtu" in u_lower
    is_kick = "kick.com" in u_lower
    impersonate_opts = ["--impersonate", "chrome"] if is_kick else []
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
            "-f", "worstvideo+bestaudio/worst/best",
            "--get-url",
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


async def _download_audio_window(
    hls_url: str,
    start_sec: float,
    duration_sec: float,
    output_path: Path,
) -> Path:
    """
    Download a specific time window from an HLS stream using ffmpeg seek.

    Uses ffmpeg -ss (input seek) to jump directly to start_sec in the DVR buffer,
    then captures duration_sec seconds of audio. This correctly handles both
    live DVR streams and VODs — no real-time pacing issue.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-user_agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-ss", str(start_sec),        # seek to start position in DVR
        "-i", hls_url,                # HLS playlist URL
        "-t", str(duration_sec),      # capture this many seconds
        "-vn",                        # audio only — no video track
        "-acodec", "aac",
        "-b:a", "64k",                # low bitrate — only need intelligible speech
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
    Download a specific clip range at full quality from the HLS DVR using ffmpeg seek,
    or via yt-dlp download_video_clip_range for YouTube / direct HTTP streams.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_sec - start_sec

    is_youtube = stream_url and ("youtu" in stream_url.lower())
    is_m3u8 = ".m3u8" in hls_url.lower()

    if is_youtube or not is_m3u8:
        if stream_url:
            return await download_video_clip_range(stream_url, start_sec, end_sec, output_path)

    cmd = [
        "ffmpeg", "-y",
        "-user_agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "-ss", str(start_sec),
        "-i", hls_url,
        "-t", str(duration),
        "-c", "copy",                 # stream copy — fast, lossless
        "-avoid_negative_ts", "make_zero",
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


async def _process_window_parallel(
    *,
    hls_url: str,
    window_index: int,
    total_windows: int,
    window_start: float,
    window_duration: float,
    window_dir: Path,
    layout_mode: str,
    deepseek_api_key: str,
    deepseek_model: str,
    streamer: str,
    video_title: str,
    clips_per_window: int,
    watermark_path: Path,
    bot,
    chat_id,
    global_clip_counter: list,
    progress_counter: list,
    tracker: "_ProgressTracker",
    stream_url: str = "",
) -> list:
    """
    Fully self-contained window processor. Runs concurrently with all other windows.
    Each window seeks to a different position in the HLS DVR — guaranteed unique content.
    """
    label = (
        f"Window {window_index + 1}/{total_windows} "
        f"({int(window_start // 60)}m–{int((window_start + window_duration) // 60)}m)"
    )
    logger.info("=== [PARALLEL] Starting %s ===", label)
    window_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Download audio window via ffmpeg seek ────────────────────────────────────────
    audio_path = window_dir / "audio.m4a"
    try:
        await _download_audio_window(
            hls_url=hls_url,
            start_sec=window_start,
            duration_sec=window_duration,
            output_path=audio_path,
        )
    except PipelineError as exc:
        logger.warning("%s — audio failed: %s. Skipping.", label, exc.reason)
        tracker.scanned += 1
        return []

    if not audio_path.exists() or audio_path.stat().st_size < 4096:
        logger.warning("%s — audio too small/missing. Skipping.", label)
        tracker.scanned += 1
        return []

    logger.info("%s — audio ready: %.1f MB", label, audio_path.stat().st_size / 1e6)
    tracker.scanned += 1  # 📡 Scanning counter

    # ── 2. Transcribe ───────────────────────────────────────────────────────────────
    last_exc = None
    data = None
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        logger.warning("%s — DEEPGRAM_API_KEY not set. Skipping.", label)
        tracker.analyzed += 1
        return []

    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&utterances=true"
    headers = {"Authorization": f"Token {deepgram_api_key}"}

    try:
        import httpx
        async with httpx.AsyncClient() as client:
            for attempt in range(1, 4):
                try:
                    with open(audio_path, "rb") as fh:
                        audio_data = fh.read()
                    response = await client.post(url, headers=headers, content=audio_data, timeout=120.0)
                    response.raise_for_status()
                    data = response.json()
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.warning("%s — transcription attempt %d/3 failed: %s", label, attempt, exc)
                    if attempt < 3:
                        await asyncio.sleep(attempt * 2.0)
            if not data:
                raise Exception(last_exc)
    except Exception as exc:
        logger.warning("%s — transcription fully failed: %s. Skipping.", label, exc)
        tracker.analyzed += 1
        return []

    # Map Deepgram utterances to segments
    raw_utterances = data.get("results", {}).get("utterances", [])
    segments = []

    for utt in raw_utterances:
        utt_start = utt.get("start", 0.0)
        utt_end = utt.get("end", 0.0)
        utt_text = utt.get("transcript", "")

        utt_words = []
        for w in utt.get("words", []):
            w_text = w.get("word", "").strip()
            w_start = w.get("start", 0.0)
            w_end = w.get("end", 0.0)
            if w_text:
                utt_words.append({
                    "word": w_text,
                    "start": round(w_start + window_start, 3),
                    "end": round(w_end + window_start, 3),
                })

        segments.append({
            "text": utt_text.strip(),
            "start": round(utt_start + window_start, 3),
            "end": round(utt_end + window_start, 3),
            "words": utt_words,
        })

    if not segments:
        logger.info("%s — silent/AFK. Skipping.", label)
        tracker.analyzed += 1
        return []

    # ── 3. Score moments ────────────────────────────────────────────────────────────
    try:
        moments = await score_moments(
            segments=segments,
            api_key=deepseek_api_key,
            top_n=clips_per_window,
            model=deepseek_model,
            streamer=streamer,
            video_title=video_title,
        )
    except PipelineError as exc:
        logger.warning("%s — scoring failed: %s. Skipping.", label, exc.reason)
        tracker.analyzed += 1
        return []

    tracker.analyzed += 1  # 🧠 Analyzing counter

    if not moments:
        return []

    t_end = window_start + window_duration
    valid = [m for m in moments if window_start <= m.start < t_end and m.end > m.start]
    if not valid:
        valid = moments

    # Hard clamp: enforce 20s minimum, 60s maximum duration regardless of AI output
    MIN_DUR, MAX_DUR = 20.0, 60.0
    clamped = []
    for m in valid:
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
        ))
    valid = clamped

    # Assign globally unique indices
    indexed = []
    for m in valid:
        idx = global_clip_counter[0]
        global_clip_counter[0] += 1
        indexed.append(Moment(
            index=idx, start=m.start, end=m.end,
            caption_lines=m.caption_lines, emoji=m.emoji,
        ))

    tracker.moments_found += len(indexed)  # ⬇️ Downloading counter

    # ── 4. Selective HD clip downloads via ffmpeg seek ───────────────────────────────
    clips_dir = window_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    clip_paths = []
    valid_for_render = []

    sem = _get_hd_sem()

    async def _dl_one(moment):
        clip_out = clips_dir / f"clip_{moment.index:03d}.mp4"
        dl_start = max(0.0, moment.start - 1.0)
        dl_end = moment.end + 1.0
        async with sem:
            try:
                await _download_hd_clip_from_hls(
                    hls_url=hls_url,
                    start_sec=dl_start,
                    end_sec=dl_end,
                    output_path=clip_out,
                    stream_url=stream_url,
                )
                return clip_out, moment
            except PipelineError as exc:
                logger.warning("%s — HD clip failed idx %d: %s", label, moment.index, exc.reason)
                return None, None

    results = await asyncio.gather(*(_dl_one(m) for m in indexed))
    for clip_out, moment in results:
        if clip_out and moment:
            clip_paths.append(clip_out)
            valid_for_render.append(moment)

    if not clip_paths:
        logger.warning("%s — all HD downloads failed.", label)
        return indexed

    # ── 5. Captions + compositing ──────────────────────────────────────────────────
    captions_dir = window_dir / "captions"
    finals_dir = window_dir / "finals"

    captions = await asyncio.get_event_loop().run_in_executor(
        None, render_captions, valid_for_render, _ASSETS_DIR, captions_dir, layout_mode,
    )
    final_clips = await composite_clips(
        clips=clip_paths, captions=captions, watermark_path=watermark_path,
        moments=valid_for_render, output_dir=finals_dir, layout_mode=layout_mode,
        segments=segments,
    )

    tracker.composited += len(final_clips)  # 🏞️ Finishing counter

    # ── 6. Deliver ───────────────────────────────────────────────────────────────────
    for i, final_path in enumerate(final_clips):
        m = valid_for_render[i]
        clip_num = m.index + 1
        mins = int(m.start // 60)
        secs = int(m.start % 60)
        await _send_clip(
            bot, chat_id, final_path,
            f"🎬 Clip {clip_num} • {streamer}  [{mins}m{secs:02d}s]"
        )
        tracker.delivered += 1

    logger.info("=== [PARALLEL] %s complete — %d clips delivered ===", label, len(final_clips))
    return valid_for_render

async def run_streaming_pipeline(
    url: str,
    bot,
    chat_id,
    run_dir: Path,
    layout_mode: str = "pillarbox",
    stream_start_sec: float | None = None,
    stream_end_sec: float | None = None,
    chunk_minutes: int = 10,
    clips_per_window: int = 1,
) -> None:
    """
    Process a stream URL by running ALL time windows in parallel.

    For live streams with stream_start_sec=None / stream_end_sec=None,
    automatically clips the LAST 60 minutes of the DVR buffer.

    For VODs, pass explicit start/end seconds to target a specific range.

    Total wall-clock time ≈ time for the slowest single window.
    """
    deepseek_api_key = os.environ["DEEPSEEK_API_KEY"]
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    clip_window_minutes = int(os.getenv("CLIP_WINDOW_MINUTES", "60"))

    chunk_sec = chunk_minutes * 60.0
    watermark_path = _ASSETS_DIR / "watermark.png"

    # ── Step 1: Resolve HLS URL (fast, ~2s) ──────────────────────────────────
    try:
        hls_url = await _get_hls_url(url)
    except PipelineError as exc:
        await _send_safe(bot, chat_id, f"❌ Could not resolve stream URL: {exc.reason}")
        return

    # ── Step 2: Probe DVR duration ────────────────────────────────────────────
    dvr_duration = await _get_stream_duration(hls_url, stream_url=url)

    # ── Step 3: Auto-calculate last 60 minutes if not specified ───────────────
    if stream_end_sec is None:
        stream_end_sec = dvr_duration if dvr_duration > 0 else chunk_sec * 6
    if stream_start_sec is None:
        stream_start_sec = max(0.0, stream_end_sec - clip_window_minutes * 60.0)

    # ── Step 4: Metadata ──────────────────────────────────────────────────────
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
    dvr_hrs = dvr_duration / 3600 if dvr_duration > 0 else 0

    logger.info(
        "PARALLEL pipeline: %d windows × %d min | stream range %.0fs–%.0fs | DVR: %.0fs",
        total, chunk_minutes, stream_start_sec, stream_end_sec, dvr_duration,
    )

    # Initialize live progress tracker card
    tracker = _ProgressTracker(
        bot=bot,
        chat_id=chat_id,
        total_windows=total,
        streamer=streamer,
        title=video_title,
    )
    await tracker.start()

    global_clip_counter = [0]
    progress_counter = [0]

    window_tasks = [
        _process_window_parallel(
            hls_url=hls_url,
            window_index=i,
            total_windows=total,
            window_start=w_start,
            window_duration=min(chunk_sec, stream_end_sec - w_start),
            window_dir=run_dir / f"window_{i:03d}",
            layout_mode=layout_mode,
            deepseek_api_key=deepseek_api_key,
            deepseek_model=deepseek_model,
            streamer=streamer,
            video_title=video_title,
            clips_per_window=clips_per_window,
            watermark_path=watermark_path,
            bot=bot,
            chat_id=chat_id,
            global_clip_counter=global_clip_counter,
            progress_counter=progress_counter,
            tracker=tracker,
            stream_url=url,
        )
        for i, w_start in enumerate(windows)
    ]

    all_results = await asyncio.gather(*window_tasks, return_exceptions=True)

    all_moments = []
    for r in all_results:
        if isinstance(r, list):
            all_moments.extend(r)
        elif isinstance(r, Exception):
            logger.warning("Window raised exception: %s", r)

    total_delivered = global_clip_counter[0]
    scope = "full VOD" if stream_start_sec == 0.0 else f"last {total_mins} minutes"

    if not all_moments or total_delivered == 0:
        await tracker.stop("⚠️ No viral moments found in this stream/VOD.")
        return

    final_card = (
        f"✅ *Processing Complete!*\n"
        f"• Delivered {total_delivered} clip(s) from {streamer}'s {scope}.\n"
        f"• Enjoy your clips below!"
    )
    await tracker.stop(final_card)

