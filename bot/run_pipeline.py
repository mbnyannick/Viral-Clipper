"""
Pipeline orchestrator with rich metadata preview cards (sent instantly upon metadata extraction),
live status tracking for /update command, exponential notification schedule, and clean user messaging.
"""

import asyncio
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

import html
from telegram import Update
from telegram.ext import ContextTypes

from pipeline.download import download, extract_metadata, _kick_vod_get_hls_url
from pipeline.chunk import chunk_audio
from pipeline.transcribe import transcribe_chunks
from pipeline.score import score_moments, generate_clip_captions, normalize_streamer_name
from pipeline.clip import cut_clips
from pipeline.caption import render_captions
from pipeline.composite import composite_clips
from pipeline.deliver import deliver_clips
from pipeline.errors import PipelineError
from pipeline.streaming_pipeline import run_streaming_pipeline

logger = logging.getLogger(__name__)

_BASE_TMP = Path("tmp")
_ASSETS_DIR = Path("assets")

# Global dict exposed for /update command status tracking
active_run_status: dict = {}


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


def estimate_processing_time(duration_seconds: float, layout_mode: str = "pillarbox") -> str:
    """Calculate estimated processing time with safety buffer."""
    if duration_seconds <= 0:
        return "~2 to 4 minutes"

    duration_hours = duration_seconds / 3600.0
    est_mins = 1.5 + (duration_hours * 0.5)
    if layout_mode == "face_crop":
        est_mins += 0.5

    min_est = max(1, int(est_mins))
    max_est = min_est + 2

    return f"~{min_est} to {max_est} minutes"


class _SmartNotifier:
    """
    Manages progress notifications according to the specified schedule:
    - Phase 1: Update every 60s (up to 4 mins)
    - 5 mins: "taking longer than expected..."
    - >5 mins: update every 5-10 minutes
    """

    def __init__(self, bot, chat_id: int | str) -> None:
        self.bot = bot
        self.chat_id = chat_id
        self.start_time = time.time()
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        for _ in range(4):
            await asyncio.sleep(60)
            await self._send("⚡ Processing your video... Still cooking up the clips!")

        await self._send("⏳ This video is taking longer than expected, but I'm still processing it...")

        while True:
            await asyncio.sleep(300)
            await self._send("⏳ Still processing your video stream... Thank you for your patience!")

    async def _send(self, text: str) -> None:
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as exc:
            logger.warning("Notification send failed: %s", exc)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self.start_time = time.time()
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()


async def run_pipeline(
    url: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    layout_mode: str = "pillarbox",
    enable_watermark: bool = True,
    enable_subtitles: bool = True,
    enable_silence_cut: bool = True,
    top_n_clips: int = 10,
    campaign_brief: str = "",
    target_duration: str = "auto",
) -> None:
    """
    Run the full VIRAL pipeline for *url* and deliver clips back to the
    operator's chat. All errors are caught and reported cleanly.
    """
    global active_run_status
    chat_id = update.effective_chat.id
    bot = context.bot

    run_id = uuid.uuid4().hex[:10]
    run_dir = _BASE_TMP / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    active_run_status[chat_id] = {
        "chat_id": chat_id,
        "url": url,
        "layout_mode": layout_mode,
        "start_time": time.time(),
        "step": "Analyzing Link & Extracting Metadata...",
        "streamer_info": {},
    }

    logger.info("=== Pipeline run %s started for %s (mode=%s) ===", run_id, url, layout_mode)

    notifier = _SmartNotifier(bot, chat_id)
    notifier.start()

    async def send_msg(text: str, parse_mode: str = "Markdown") -> None:
        try:
            await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
        except Exception as exc:
            logger.warning("Failed to send message: %s", exc)

    try:
        # ── 1. Metadata Extraction FIRST (Fast 1-2s query) ──────────────────────
        streamer_info = await extract_metadata(url)
        if chat_id in active_run_status:
            active_run_status[chat_id]["streamer_info"] = streamer_info

        streamer_name = streamer_info.get("streamer", "Streamer")
        video_title = streamer_info.get("title", "")

        if streamer_info.get("is_offline"):
            notifier.stop()
            offline_msg = (
                f"⚪ <b>{streamer_name} is currently offline.</b>\n\n"
                f"Please send a <b>VOD link</b> instead:\n"
                f"• Kick VOD: <code>https://kick.com/{streamer_name.lower()}/videos/XXXXXXXX</code>\n"
                f"• Twitch VOD: <code>https://twitch.tv/videos/XXXXXXXX</code>"
            )
            await send_msg(offline_msg, parse_mode="HTML")
            return

        raw_dur = str(streamer_info.get("duration", "0") or "0").strip()
        try:
            duration_sec = float(raw_dur)
        except ValueError:
            duration_sec = 0.0
        platform = streamer_info.get("platform", "Web Video 🌐")
        live_status = streamer_info.get("live_status", "🔴 LIVE NOW")

        dur_display = format_duration(duration_sec)
        est_time_display = estimate_processing_time(duration_sec, layout_mode=layout_mode)

        # Check for streaming (progressive chunked) mode
        stream_start_min = os.getenv("STREAM_START_MIN")
        stream_end_min = os.getenv("STREAM_END_MIN")
        streaming_chunk_min = int(os.getenv("STREAMING_CHUNK_MINUTES", "10"))
        clips_per_window = int(os.getenv("CLIPS_PER_WINDOW", "1"))

        if stream_start_min is not None and stream_end_min is not None:
            start_sec = float(stream_start_min) * 60.0
            end_sec = float(stream_end_min) * 60.0
            target_mins = int(float(stream_end_min) - float(stream_start_min))
            total_windows = max(1, target_mins // streaming_chunk_min)

            # Send streaming-mode metadata card
            card_msg = (
                f"🎬 **Video Details Identified:**\n"
                f"• **Platform:** {platform}\n"
                f"• **Live Status:** {live_status}\n"
                f"• **Streamer:** {streamer_name}\n"
                f"• **Title:** {video_title if video_title else 'N/A'}\n"
                f"• **Duration:** {dur_display}\n\n"
                f"⚡ **Progressive Mode Active!**\n"
                f"• Clipping {int(stream_start_min)}m–{int(stream_end_min)}m "
                f"in {total_windows} × {streaming_chunk_min}-min windows\n"
                f"• First clip arriving in ~30 seconds!"
            )
            await send_msg(card_msg)

            notifier.stop()
            if chat_id in active_run_status:
                active_run_status[chat_id]["step"] = f"Streaming pipeline: {total_windows} windows..."

            await run_streaming_pipeline(
                url=url,
                bot=bot,
                chat_id=chat_id,
                run_dir=run_dir,
                layout_mode=layout_mode,
                stream_start_sec=start_sec,
                stream_end_sec=end_sec,
                chunk_minutes=streaming_chunk_min,
                clips_per_window=clips_per_window,
            )
            return

        # ── Kick VOD Smart Scan Mode ─────────────────────────────────────────────
        # Instead of downloading the full VOD (hours of data), we audio-scan the
        # entire VOD in parallel 10-min chunks, find viral timestamps via AI, then
        # download ONLY the video segments that scored highest. Much faster.
        is_kick_vod = "kick.com" in url.lower() and "/videos/" in url.lower()
        if is_kick_vod:
            # Fetch the real VOD duration from Kick API (yt-dlp always returns 0 for Kick)
            try:
                _, kick_channel, kick_title, kick_dur_sec = await _kick_vod_get_hls_url(url)
                if kick_dur_sec > 0:
                    duration_sec = float(kick_dur_sec)
                    dur_display = format_duration(duration_sec)
                if kick_title and not video_title:
                    video_title = kick_title
                if kick_channel and streamer_name in ("Streamer", ""):
                    streamer_name = kick_channel.capitalize()
            except Exception as kick_exc:
                logger.warning("Could not prefetch Kick VOD metadata: %s", kick_exc)

            vod_dur_sec = duration_sec if duration_sec > 0 else 18000.0
            total_windows = max(1, int(vod_dur_sec / (streaming_chunk_min * 60)))

            card_msg = (
                f"🎬 **Video Details Identified:**\n"
                f"• **Platform:** {platform}\n"
                f"• **Live Status:** 📁 Recorded VOD\n"
                f"• **Streamer:** {streamer_name}\n"
                f"• **Title:** {video_title if video_title else 'N/A'}\n"
                f"• **Duration:** {dur_display}\n\n"
                f"⚡ **Smart Scan Mode Active!**\n"
                f"• Audio-scanning full VOD in {total_windows} parallel chunks\n"
                f"• Only downloading the top viral segments\n"
                f"• Estimated time: ~5–8 minutes ⚡"
            )
            await send_msg(card_msg)

            notifier.stop()
            if chat_id in active_run_status:
                active_run_status[chat_id]["step"] = f"Smart scan: {total_windows} parallel audio windows..."

            target_total = top_n_clips if top_n_clips > 0 else int(os.getenv("TOP_N_CLIPS", "10"))
            await run_streaming_pipeline(
                url=url,
                bot=bot,
                chat_id=chat_id,
                run_dir=run_dir,
                layout_mode=layout_mode,
                stream_start_sec=0.0,
                stream_end_sec=vod_dur_sec,
                chunk_minutes=streaming_chunk_min,
                target_total_clips=target_total,
                campaign_brief=campaign_brief,
                target_duration=target_duration,
            )
            return


        # Send rich metadata preview card IMMEDIATELY to Telegram and track message object
        card_msg = (
            f"🎬 **Video Details Identified:**\n"
            f"• **Platform:** {platform}\n"
            f"• **Live Status:** {live_status}\n"
            f"• **Streamer:** {streamer_name}\n"
            f"• **Title:** {video_title if video_title else 'N/A'}\n"
            f"• **Duration:** {dur_display}\n\n"
            f"⚙️ **Live Progress:** 📥 Downloading Video Stream...\n\n"
            f"☕ Feel free to step away while I process your clips!"
        )
        status_msg = None
        try:
            status_msg = await bot.send_message(chat_id=chat_id, text=card_msg, parse_mode="Markdown")
        except Exception as exc:
            logger.warning("Failed to send initial status message: %s", exc)

        async def update_status(step_name: str) -> None:
            if chat_id in active_run_status:
                active_run_status[chat_id]["step"] = step_name
            if not status_msg:
                return
            try:
                msg_body = (
                    f"🎬 **Video Details Identified:**\n"
                    f"• **Platform:** {platform}\n"
                    f"• **Live Status:** {live_status}\n"
                    f"• **Streamer:** {streamer_name}\n"
                    f"• **Title:** {video_title if video_title else 'N/A'}\n"
                    f"• **Duration:** {dur_display}\n\n"
                    f"⚙️ **Live Progress:** {step_name}\n\n"
                    f"☕ Feel free to step away while I process your clips!"
                )
                await status_msg.edit_text(msg_body, parse_mode="Markdown")
            except Exception as exc:
                logger.warning("Status edit failed: %s", exc)

        # ── 2. Download Video Stream ──────────────────────────────────────────────
        await update_status("📥 1/5 — Downloading Video Stream...")
        video_path, audio_path, _ = await download(url, run_dir, streamer_info=streamer_info)

        # ── 3. Chunk & Transcribe ───────────────────────────────────────────────
        await update_status("🎙️ 2/5 — Transcribing Speech...")
        chunk_minutes = int(os.getenv("CHUNK_DURATION_MINUTES", "3"))
        chunks = await chunk_audio(audio_path, chunk_duration_minutes=chunk_minutes)
        segments = await transcribe_chunks(chunks)

        # ── 4. Score & Select Moments ───────────────────────────────────────────
        await update_status("🧠 3/5 — Analyzing Virality & Story Arc...")
        top_n = top_n_clips if top_n_clips > 0 else int(os.getenv("TOP_N_CLIPS", "10"))
        deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        moments = await score_moments(
            segments,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            top_n=top_n,
            model=deepseek_model,
            streamer=streamer_name,
            video_title=video_title,
            campaign_brief=campaign_brief,
            target_duration=target_duration,
        )



        # ── 5. Cut, Render & Composite ──────────────────────────────────────────
        await update_status(f"✂️ 5/5 — Compositing {len(moments)} Clips...")
        clips_dir = run_dir / "clips"
        clips = await cut_clips(video_path, moments, clips_dir, url=url)

        captions_dir = run_dir / "captions"
        captions = await asyncio.get_event_loop().run_in_executor(
            None,
            render_captions,
            moments,
            _ASSETS_DIR,
            captions_dir,
            layout_mode,
        )

        finals_dir = run_dir / "finals"
        watermark_custom = Path("tmp/custom_watermark.png")
        if enable_watermark and watermark_custom.exists():
            watermark_path = watermark_custom
        elif enable_watermark:
            watermark_path = _ASSETS_DIR / "watermark.png"
        else:
            watermark_path = _ASSETS_DIR / "no_watermark.png"

        final_clips, clip_captions = await asyncio.gather(
            composite_clips(
                clips,
                captions,
                watermark_path,
                moments,
                finals_dir,
                layout_mode=layout_mode,
                enable_silence_cut=enable_silence_cut,
                enable_subtitles=enable_subtitles,
                segments=segments,
            ),
            generate_clip_captions(
                moments,
                api_key=os.environ["DEEPSEEK_API_KEY"],
                streamer=streamer_name,
                video_title=video_title,
                campaign_brief=campaign_brief,
            ),
        )

        # ── 5.5 HD Cover Thumbnail Generation Pass ─────────────────────────────
        try:
            from pipeline.thumbnail import generate_cover_thumbnail
            thumbs_dir = run_dir / "thumbnails"
            thumbs_dir.mkdir(parents=True, exist_ok=True)

            thumb_tasks = [
                generate_cover_thumbnail(fc, m, thumbs_dir / f"thumbnail_{m.index:02d}.jpg", cap_info[0])
                for fc, cap_info, m in zip(final_clips, captions, moments)
            ]
            await asyncio.gather(*thumb_tasks)
            logger.info("Generated %d HD Cover Thumbnails", len(final_clips))
        except Exception as thumb_exc:
            logger.warning("Thumbnail generation warning (continuing delivery): %s", thumb_exc)

        await update_status("🚀 6/6 — Delivering HD Clips to Telegram!")

        notifier.stop()

        # ── "Almost Ready" Announcement ─────────────────────────────────────────
        await send_msg("⚡ Your clips are ready!", parse_mode="")

        # ── 6. Delivery ─────────────────────────────────────────────────────────
        if chat_id in active_run_status:
            active_run_status[chat_id]["step"] = "Delivering Clips to Telegram..."
        await deliver_clips(final_clips, bot, chat_id, moments=moments, clip_captions=clip_captions)

        if clip_captions:
            title_blocks = []
            for idx, cap in enumerate(clip_captions, start=1):
                clean_cap = html.escape(cap.strip())
                title_blocks.append(f"📌 <b>Clip {idx:02d} YouTube Title:</b>\n<code>{clean_cap}</code>")
            
            header = (
                f"🔴 <b>YouTube Titles (Clips 01–{len(clip_captions):02d})</b>\n"
                f"<i>Tap any box below to copy its YouTube title & hashtags:</i>\n\n"
            )
            summary_box = header + "\n\n".join(title_blocks)
            await send_msg(summary_box, parse_mode="HTML")

        logger.info("=== Pipeline run %s complete ===", run_id)

    except PipelineError as exc:
        notifier.stop()
        logger.error("Pipeline %s failed at '%s': %s", run_id, exc.step, exc.reason)
        error_msg = (
            f"❌ <b>Video Processing Failed</b>\n\n"
            f"<b>Step:</b> <code>{html.escape(exc.step)}</code>\n"
            f"<b>Reason:</b> <code>{html.escape(exc.reason)}</code>\n\n"
            f"<i>The bot has safely skipped this video. You can send another link!</i>"
        )
        await send_msg(error_msg, parse_mode="HTML")
    except Exception as exc:
        notifier.stop()
        logger.exception("Unexpected error in pipeline run %s", run_id)
        error_msg = (
            f"⚠️ <b>Unexpected System Error</b>\n\n"
            f"<code>{html.escape(str(exc))}</code>\n\n"
            f"<i>Please check the server logs for a full traceback. The bot is ready for the next link.</i>"
        )
        await send_msg(error_msg, parse_mode="HTML")

    finally:
        notifier.stop()
        active_run_status.pop(chat_id, None)
        try:
            shutil.rmtree(run_dir, ignore_errors=True)
            logger.info("Cleaned up run directory: %s", run_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cleanup failed for %s: %s", run_dir, exc)
