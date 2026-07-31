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
from pipeline.score import Moment, score_moments, _generate_fallback_moments

logger = logging.getLogger(__name__)

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
        import html
        n = self.total
        s_name = html.escape(self.streamer)
        t_name = html.escape(self.title[:28] + ("…" if len(self.title) > 28 else ""))
        title_part = f" • {t_name}" if t_name else ""

        def _stage(icon: str, label: str, done: int, total: int, override: str = "") -> str:
            if override:
                return f"{icon} {label:<10} {override}"
            bar = self._bar(done, total)
            if done >= total:
                return f"{icon} {label:<10} {bar}  {done}/{total} ✅"
            return f"{icon} {label:<10} {bar}  {done}/{total}"

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
            f"🎬 <b>Processing {s_name}</b>{title_part}",
            "<pre>",
            _stage("📡", "Scanning", self.scanned, n),
            _stage("🧠", "Analyzing", self.analyzed, n),
            f"⬇️ {'Download':<10} {dl_override}",
            f"🏞️ {'Finishing':<10} {finish_override}",
            "</pre>",
        ]
        return "\n".join(lines)

    # ───────────────────────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Send the initial progress message and start the edit loop."""
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
        """Edit progress card and send 1-minute heartbeat status messages."""
        heartbeat_counter = 0
        while not self._stopped:
            await asyncio.sleep(5)
            if self._stopped:
                break
            await self._edit()
            heartbeat_counter += 5
            if heartbeat_counter >= 60:
                heartbeat_counter = 0
                if not self._stopped:
                    msg_text = (
                        f"⏱️ <b>1-Minute Progress Update:</b>\n"
                        f"• 📡 Audio Scanned: {self.scanned}/{self.total} windows\n"
                        f"• 🧠 AI Analyzed: {self.analyzed}/{self.total} windows\n"
                        f"• 🎯 Moments Found: {self.moments_found}\n"
                        f"• 🚀 Clips Delivered: {self.delivered}"
                    )
                    await _send_safe(self.bot, self.chat_id, msg_text, parse_mode="HTML")

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

    tracker.scanned += 1  # 📡 Scanned counter

    # ── 2. Deepgram Transcription ──────────────────────────────────────────────────
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        logger.warning("%s — DEEPGRAM_API_KEY missing. Skipping.", label)
        tracker.analyzed += 1
        return {"window_dir": window_dir, "segments": [], "moments": []}

    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&utterances=true"
    headers = {"Authorization": f"Token {deepgram_api_key}"}
    data = None

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
                    logger.warning("%s — Deepgram attempt %d/3 failed: %s", label, attempt, exc)
                    if attempt < 3:
                        await asyncio.sleep(attempt * 1.5)
    except Exception as exc:
        logger.warning("%s — Deepgram transcription failed: %s", label, exc)
        tracker.analyzed += 1
        return {"window_dir": window_dir, "segments": [], "moments": []}

    raw_utterances = (data or {}).get("results", {}).get("utterances", [])
    segments = []
    for utt in raw_utterances:
        utt_start = utt.get("start", 0.0)
        utt_end = utt.get("end", 0.0)
        utt_text = utt.get("transcript", "")
        utt_words = [
            {
                "word": w.get("word", "").strip(),
                "start": round(w.get("start", 0.0) + window_start, 3),
                "end": round(w.get("end", 0.0) + window_start, 3),
            }
            for w in utt.get("words", [])
            if w.get("word", "").strip()
        ]
        segments.append({
            "text": utt_text.strip(),
            "start": round(utt_start + window_start, 3),
            "end": round(utt_end + window_start, 3),
            "words": utt_words,
        })

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

    tracker = _ProgressTracker(
        bot=bot, chat_id=chat_id, total_windows=total, streamer=streamer, title=video_title,
    )
    await tracker.start()

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
                await _download_hd_clip_from_hls(
                    hls_url=hls_url, start_sec=dl_start, end_sec=dl_end, output_path=clip_out, stream_url=url,
                )
            except Exception as exc:
                logger.warning("HD clip download failed idx %d: %s", moment.index, exc)
                return

        captions_dir = wdir / "captions"
        finals_dir = wdir / "finals"

        captions = await asyncio.get_event_loop().run_in_executor(
            None, render_captions, [moment], _ASSETS_DIR, captions_dir, layout_mode,
        )

        async def _on_clip_done(final_path: Path, m: Moment) -> None:
            clip_num = m.index + 1
            mins = int(m.start // 60)
            secs = int(m.start % 60)
            await _send_clip(
                bot, chat_id, final_path,
                f"🎬 Clip {clip_num} • {streamer}  [{mins}m{secs:02d}s]",
            )
            delivered_count[0] += 1
            tracker.delivered += 1
            tracker.composited += 1

        await composite_clips(
            clips=[clip_out], captions=captions, watermark_path=watermark_path,
            moments=[moment], output_dir=finals_dir, layout_mode=layout_mode,
            segments=segments, on_clip_ready=_on_clip_done,
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

    final_card = (
        f"✅ *Processing Complete!*\n"
        f"• Delivered {total_delivered} clip(s) from {streamer}'s {scope}.\n"
        f"• Enjoy your clips below!"
    )
    await tracker.stop(final_card)

