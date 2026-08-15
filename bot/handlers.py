"""
Telegram message handler & queue manager with interactive commands (/help, /stop, /queue, /clear, /update)
and Inline Keyboard Layout Selector (Pillarbox vs. Face-Crop).
"""

import asyncio
import html
import json
import logging
import os
import re
import time
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from bot.run_pipeline import run_pipeline
from bot.scheduler import next_peak_slot, scheduler
from pipeline import get_public_base_url
from pipeline.text_utils import format_seo_title, generate_rich_hashtags

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+")

# Queue tuple: (url, update, context, layout_mode, enable_watermark, enable_silence_cut, top_n_clips, campaign_brief, target_duration)
_job_queue: asyncio.Queue[tuple[str, Update, ContextTypes.DEFAULT_TYPE, str, bool, bool, int, str, str]] = asyncio.Queue()
_WORKER_TASKS: list[asyncio.Task] = []
MAX_PARALLEL_WORKERS = 2

# Store active task per chat_id: chat_id -> (task, url)
_active_tasks_by_chat: dict[int, tuple[asyncio.Task, str]] = {}

# Store pending link submissions awaiting button clicks: chat_id -> dict
_pending_links: dict[int, dict] = {}
# Store active campaign brief per chat_id: chat_id -> str
_campaign_briefs: dict[int, str] = {}
# Store clip sessions awaiting "Schedule All" confirmation: chat_id -> list of clip dicts
_pending_schedule_sessions: dict[int, list[dict]] = {}


# Persistent approved users database path
APPROVED_USERS_FILE = Path("approved_users.json")


def _make_layout_keyboard(target_duration: str = "auto") -> InlineKeyboardMarkup:
    buttons = [
        [
            InlineKeyboardButton("👤 Face Tracking (9:16 Fullscreen)", callback_data="fmt:face_crop"),
        ],
        [
            InlineKeyboardButton("🎬 9:16 Vertical (Blurred BG)", callback_data="fmt:blurred_frame"),
            InlineKeyboardButton("🖤 9:16 Vertical (Black Canvas)", callback_data="fmt:black_canvas"),
        ],
        [
            InlineKeyboardButton("🔲 9:16 Vertical (Square Blur)", callback_data="fmt:square_blur"),
            InlineKeyboardButton("🩷 9:16 Vertical (Pink Canvas)", callback_data="fmt:pink_canvas"),
        ],
        [
            InlineKeyboardButton("❌ Cancel", callback_data="wiz:cancel"),
        ],
    ]
    return InlineKeyboardMarkup(buttons)



def _make_watermark_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🟢 Yes, Upload Watermark", callback_data="wm:yes"),
            InlineKeyboardButton("🔴 No Watermark", callback_data="wm:no"),
        ],
        [
            InlineKeyboardButton("↩️ Back to Layout", callback_data="wiz:back_layout"),
            InlineKeyboardButton("❌ Cancel", callback_data="wiz:cancel"),
        ],
    ])


def _make_subtitles_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("💬 Yes, Include Word Subtitles", callback_data="sub:yes"),
        ],
        [
            InlineKeyboardButton("🚫 No Subtitles (Clean Video)", callback_data="sub:no"),
        ],
        [
            InlineKeyboardButton("↩️ Back", callback_data="wiz:back_wm"),
            InlineKeyboardButton("❌ Cancel", callback_data="wiz:cancel"),
        ],
    ])


def _make_clips_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("3 Clips", callback_data="clips:3"),
            InlineKeyboardButton("5 Clips", callback_data="clips:5"),
            InlineKeyboardButton("10 Clips (Default)", callback_data="clips:10"),
        ],
        [
            InlineKeyboardButton("15 Clips", callback_data="clips:15"),
            InlineKeyboardButton("20 Clips", callback_data="clips:20"),
            InlineKeyboardButton("50 Clips", callback_data="clips:50"),
        ],
        [
            InlineKeyboardButton("↩️ Back", callback_data="wiz:back_sub"),
            InlineKeyboardButton("❌ Cancel", callback_data="wiz:cancel"),
        ],
    ])


def _make_duration_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🤖 Smart Auto (Recommended)", callback_data="dur:auto"),
        ],
        [
            InlineKeyboardButton("⚡ 0 to 30 Seconds", callback_data="dur:0_30"),
            InlineKeyboardButton("🎬 30 to 60 Seconds", callback_data="dur:30_60"),
        ],
        [
            InlineKeyboardButton("💰 61 to 90 Seconds (TikTok Monetized)", callback_data="dur:61_90"),
        ],
        [
            InlineKeyboardButton("↩️ Back to Clip Count", callback_data="wiz:back_clips"),
            InlineKeyboardButton("❌ Cancel", callback_data="wiz:cancel"),
        ],
    ])


async def _safe_edit_message_text(query, **kwargs) -> None:
    if not query:
        return
    try:
        await query.edit_message_text(**kwargs)
    except BadRequest as exc:
        msg = str(exc)
        if any(err in msg for err in ("Message is not modified", "Query is too old", "message to edit not found", "MESSAGE_ID_INVALID")):
            logger.debug("Ignored Telegram edit error: %s", exc)
            return
        raise


async def _safe_edit_message_reply_markup(query, reply_markup) -> None:
    if not query:
        return
    try:
        await query.edit_message_reply_markup(reply_markup=reply_markup)
    except BadRequest as exc:
        msg = str(exc)
        if any(err in msg for err in ("Message is not modified", "Query is too old", "message to edit not found", "MESSAGE_ID_INVALID")):
            logger.debug("Ignored Telegram edit error: %s", exc)
            return
        raise


async def _safe_edit_message_caption(query, **kwargs) -> None:
    if not query:
        return
    try:
        await query.edit_message_caption(**kwargs)
    except BadRequest as exc:
        msg = str(exc)
        if any(err in msg for err in ("Message is not modified", "Query is too old", "message to edit not found", "MESSAGE_ID_INVALID")):
            logger.debug("Ignored Telegram caption edit error: %s", exc)
            return
        raise


def _load_approved_users() -> dict[int, dict]:
    """Load approved users from disk."""
    if APPROVED_USERS_FILE.exists():
        try:
            with open(APPROVED_USERS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
                return {int(k): v for k, v in raw.items()}
        except Exception as exc:
            logger.warning("Failed to load approved_users.json: %s", exc)
    return {}


def _save_approved_users(users_map: dict[int, dict]) -> None:
    """Save approved users to disk."""
    try:
        APPROVED_USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(APPROVED_USERS_FILE, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in users_map.items()}, f, indent=2)
    except Exception as exc:
        logger.error("Failed to save approved_users.json: %s", exc)


_approved_users_db: dict[int, dict] = _load_approved_users()


def _is_master_admin(user_id: int) -> bool:
    operator_id = os.environ.get("TELEGRAM_OPERATOR_CHAT_ID", "0").strip()
    return bool(operator_id and operator_id.isdigit() and user_id == int(operator_id))


def _is_operator(user_id: int) -> bool:
    if _is_master_admin(user_id):
        return True
    if user_id in _approved_users_db:
        return True
    allowed_users_raw = os.environ.get("ALLOWED_TELEGRAM_USERS", "").strip()
    if allowed_users_raw:
        for uid in allowed_users_raw.split(","):
            uid_str = uid.strip()
            if uid_str.isdigit() and user_id == int(uid_str):
                return True
    return False


async def _handle_unapproved_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        return

    user_id = user.id
    username = f"@{user.username}" if user.username else "No username"
    first_name = user.first_name or ""
    last_name = user.last_name or ""
    full_name = f"{first_name} {last_name}".strip() or "User"

    if update.message:
        await update.message.reply_text(
            "🔒 **Access Restricted**\n\n"
            "Your account is not approved yet. A request has been sent to the Master Admin for approval! "
            "You will receive a notification here once approved.",
            parse_mode="Markdown",
        )

    operator_id_str = os.environ.get("TELEGRAM_OPERATOR_CHAT_ID", "0").strip()
    if operator_id_str.isdigit() and int(operator_id_str) > 0:
        op_id = int(operator_id_str)
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(f"✅ Approve {full_name}", callback_data=f"app:yes:{user_id}"),
                InlineKeyboardButton("❌ Reject", callback_data=f"app:no:{user_id}"),
            ]
        ])
        admin_card = (
            f"🔔 **New Access Request!**\n\n"
            f"• **Name:** {full_name}\n"
            f"• **Username:** {username}\n"
            f"• **User ID:** `{user_id}`\n\n"
            f"Tap below to grant or deny access:"
        )
        try:
            await context.bot.send_message(
                chat_id=op_id,
                text=admin_card,
                reply_markup=keyboard,
                parse_mode="Markdown",
            )
        except Exception as exc:
            logger.error("Failed to send approval alert to Master Admin %d: %s", op_id, exc)


async def handle_users(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /users — Displays all approved users with live processing indicators."""
    if not update.effective_user or not _is_master_admin(update.effective_user.id):
        return

    from bot.run_pipeline import active_run_status
    active_chat = active_run_status.get("chat_id")

    operator_id_str = os.environ.get("TELEGRAM_OPERATOR_CHAT_ID", "0").strip()

    lines = [f"👥 **Approved Users List ({len(_approved_users_db) + 1}):**\n"]

    # Master Admin line
    admin_cid = int(operator_id_str) if operator_id_str.isdigit() else 0
    admin_active = (admin_cid in _active_tasks_by_chat) and not _active_tasks_by_chat[admin_cid][0].done()
    admin_status = "🟢 *Active Processing*" if admin_active else "⚪ *Idle*"
    lines.append(f"1. 👑 **Master Admin** — `{operator_id_str}` — Status: {admin_status}")

    # Inspect queue per chat_id
    queued_chats = [item[1].effective_chat.id for item in list(_job_queue._queue) if item[1] and item[1].effective_chat]

    idx = 2
    for uid, meta in _approved_users_db.items():
        uname = meta.get("username", f"User {uid}")
        
        user_active = (uid in _active_tasks_by_chat) and not _active_tasks_by_chat[uid][0].done()
        if user_active:
            status_str = "🟢 *Active Processing*"
        elif uid in queued_chats:
            q_cnt = queued_chats.count(uid)
            status_str = f"📥 *In Queue ({q_cnt} video{'s' if q_cnt > 1 else ''})*"
        else:
            status_str = "⚪ *Idle*"

        lines.append(f"{idx}. {uname} (`{uid}`) — Status: {status_str}")
        idx += 1

    lines.append("\n💡 *To revoke access for a user:* Type `/revoke <user_id>`")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def handle_revoke(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /revoke <user_id> — Revokes user access."""
    if not update.effective_user or not _is_master_admin(update.effective_user.id):
        return

    if not context.args:
        await update.message.reply_text("👉 Usage: `/revoke <user_id>`", parse_mode="Markdown")
        return

    target_str = context.args[0].strip()
    if target_str.isdigit():
        target_uid = int(target_str)
        if target_uid in _approved_users_db:
            _approved_users_db.pop(target_uid, None)
            _save_approved_users(_approved_users_db)
            await update.message.reply_text(f"🗑️ Access revoked for User `{target_uid}`.", parse_mode="Markdown")
            return

    await update.message.reply_text(f"⚠️ User `{target_str}` not found in approved list.", parse_mode="Markdown")


async def handle_brief(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /brief [text] — Set, view, or clear active Campaign Brief & Rules."""
    if not update.effective_user or not _is_operator(update.effective_user.id):
        return

    chat_id = update.effective_chat.id
    raw_args = " ".join(context.args).strip() if context.args else ""

    if not raw_args:
        current = _campaign_briefs.get(chat_id, "")
        if current:
            msg = (
                f"📝 **Current Active Campaign Brief:**\n\n"
                f"```\n{current}\n```\n\n"
                f"💡 *To update:* Type `/brief <your new campaign rules>`\n"
                f"💡 *To clear:* Type `/brief clear`"
            )
        else:
            msg = (
                "📝 **No Campaign Brief Active.**\n\n"
                "You can set a campaign brief so clips automatically follow creator rules, mandatory hashtags, and caption requirements!\n\n"
                "👉 **Example:** `/brief Creator: Lacy. Caption must mention Lacy. Hashtags: #lacy #lacyclips`\n\n"
                "To clear at any time: `/brief clear`"
            )
        await update.message.reply_text(msg, parse_mode="Markdown")
        return

    if raw_args.lower() in ("clear", "reset", "none", "off"):
        _campaign_briefs.pop(chat_id, None)
        await update.message.reply_text("🗑️ **Campaign Brief cleared!** Future videos will use standard viral scoring.", parse_mode="Markdown")
        return

    _campaign_briefs[chat_id] = raw_args
    msg = (
        f"✅ **Campaign Brief Saved!**\n\n"
        f"```\n{raw_args}\n```\n\n"
        f"🎯 All future video runs will strictly enforce these rules & mandatory hashtags!"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")




async def _launch_job(chat_id: int, num_clips: int, target_duration: str = "auto", edit_message=None) -> None:
    """Helper to finalize wizard choices and enqueue the pipeline job."""
    session = _pending_links.pop(chat_id, None)
    if not session:
        return

    url = session["url"]
    orig_update = session["update"]
    orig_context = session["context"]
    layout_mode = session.get("layout_mode", "black_canvas")
    enable_wm = session.get("enable_watermark", False)
    enable_sub = session.get("enable_subtitles", True)
    wm_name = session.get("wm_name", "None")

    dur_map = {
        "auto": "⚡ Automatic (25–60s)",
        "0_30": "⏱️ 0 – 30 Seconds",
        "15_30": "⏱️ 15 – 30 Seconds",
        "30_60": "⏱️ 30s – 1 Minute",
        "60_120": "⏱️ 1 – 2 Minutes",
    }
    dur_label = dur_map.get(target_duration, "⚡ Automatic (25–60s)")

    mode_map = {
        "black_canvas": "🖤 1:1 Square (Black Canvas)",
        "blurred_frame": "🎬 1:1 Square (Blurred BG)",
        "pink_canvas": "🩷 1:1 Square (Pink Canvas)",
        "red_canvas": "🔴 1:1 Square (Red Canvas)",
        "blue_canvas": "🔵 1:1 Square (Blue Canvas)",
        "purple_canvas": "🟣 1:1 Square (Purple Canvas)",
        "square_blur": "🔲 1:1 Square (Blurred BG)",
        "square_pink": "🩷 1:1 Square (Pink Canvas)",
        "square_black": "🖤 1:1 Square (Black Canvas)",
        "square_red": "🔴 1:1 Square (Red Canvas)",
        "square_blue": "🔵 1:1 Square (Blue Canvas)",
        "square_purple": "🟣 1:1 Square (Purple Canvas)",
        "face_crop": "👤 Face Tracking",
    }
    mode_label = mode_map.get(layout_mode, "🖤 Black Canvas")
    wm_label = f"`{wm_name}`" if enable_wm else "*None*"
    sub_label = "💬 Enabled" if enable_sub else "🚫 Disabled"

    campaign_brief = _campaign_briefs.get(chat_id, "")
    brief_status = "🟢 Active" if campaign_brief else "🔴 None"

    status_text = (
        f"✅ **Options Selected!**\n\n"
        f"• **Layout:** {mode_label}\n"
        f"• **Watermark:** {wm_label}\n"
        f"• **Subtitles:** {sub_label}\n"
        f"• **Campaign Brief:** {brief_status}\n"
        f"• **Target Clips:** *{num_clips} clips*\n"
        f"• **Clip Duration:** {dur_label}\n\n"
        f"🚀 **Video processing has started!** Please wait while clips are generated and composited... ⏳"
    )

    # Safely send/edit the status message — works for both inline buttons (callback)
    # and direct message flows (orig_update.message may be None for callback queries)
    async def _send_status(text: str) -> None:
        """Send status text, preferring edit_message then orig_update.message then bot.send_message."""
        if edit_message:
            try:
                await edit_message.edit_text(text, parse_mode="Markdown")
                return
            except Exception:
                pass
        orig_msg = getattr(orig_update, "message", None)
        if orig_msg:
            try:
                await orig_msg.reply_text(text, parse_mode="Markdown")
                return
            except Exception:
                pass
        # Last resort: send via bot directly
        try:
            from telegram import Bot as _Bot
            bot_obj = getattr(orig_context, "bot", None) if orig_context else None
            if bot_obj:
                await bot_obj.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
        except Exception as _e:
            logger.warning("_launch_job: could not send status msg: %s", _e)

    await _send_status(status_text)

    _ensure_worker_running()
    is_user_active = (chat_id in _active_tasks_by_chat) and not _active_tasks_by_chat[chat_id][0].done()
    await _job_queue.put((url, orig_update, orig_context, layout_mode, enable_wm, enable_sub, True, num_clips, campaign_brief, target_duration))

    if is_user_active:
        user_pos = sum(
            1 for item in list(_job_queue._queue)
            if item[1] and item[1].effective_chat and item[1].effective_chat.id == chat_id
        )
        queue_msg = f"📥 **Added to your queue!** (Position #{user_pos})\n\nProcessing will start automatically once your active video completes."
        await _send_status(queue_msg)


async def _queue_worker() -> None:
    """Background worker that continuously pulls and processes jobs from the queue."""
    while True:
        job = await _job_queue.get()
        url, update, context, layout_mode, enable_wm, enable_sub, enable_silence, top_n_clips, campaign_brief, target_duration = job
        chat_id = update.effective_chat.id if update and update.effective_chat else 0

        try:
            logger.info(
                "Worker picked up job for user %d: %s (mode=%s, wm=%s, sub=%s, clips=%d, dur=%s, Queue remaining: %d)",
                chat_id, url, layout_mode, enable_wm, enable_sub, top_n_clips, target_duration, _job_queue.qsize()
            )
            task = asyncio.create_task(
                run_pipeline(
                    url, update, context,
                    layout_mode=layout_mode,
                    enable_watermark=enable_wm,
                    enable_subtitles=enable_sub,
                    enable_silence_cut=enable_silence,
                    top_n_clips=top_n_clips,
                    campaign_brief=campaign_brief,
                    target_duration=target_duration,
                )
            )
            if chat_id:
                _active_tasks_by_chat[chat_id] = (task, url)

            await task
        except asyncio.CancelledError:
            logger.info("Job for user %d (%s) was cancelled via /stop", chat_id, url)
        except Exception as exc:
            logger.exception("Worker failure for user %d (%s): %s", chat_id, url, exc)
            try:
                await update.message.reply_text("❌ Something went wrong processing this link. Moving to next item...")
            except Exception:
                pass
        finally:
            if chat_id:
                _active_tasks_by_chat.pop(chat_id, None)
            _job_queue.task_done()


def _ensure_worker_running() -> None:
    """Ensure parallel queue workers are active."""
    global _WORKER_TASKS
    _WORKER_TASKS = [t for t in _WORKER_TASKS if not t.done()]
    while len(_WORKER_TASKS) < MAX_PARALLEL_WORKERS:
        t = asyncio.create_task(_queue_worker())
        _WORKER_TASKS.append(t)


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /start — short welcome message."""
    if not update.effective_user or not _is_operator(update.effective_user.id):
        await _handle_unapproved_user(update, context)
        return

    msg = (
        "⚡ <b>Welcome to VIRAL Clip Bot!</b>\n\n"
        "Send any video link (YouTube, Twitch, Kick) or upload a video file to generate viral vertical clips!\n\n"
        "📖 <i>Need help or command options? Type /help to view the full guide.</i>"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /help — shows full admin guide to Master Admin and user guide to approved users."""
    if not update.effective_user or not _is_operator(update.effective_user.id):
        await _handle_unapproved_user(update, context)
        return


    if _is_master_admin(update.effective_user.id):
        # ── Master Admin Guide ─────────────────────────────────────────────────
        msg = r"""
👑 *VIRAL Clip Bot — Admin Control Panel*

Welcome back, Admin! You have full control over this bot and all approved users.

───

📐 *4 Simple Steps:*
1. Send any video link or upload a video file.
2. Choose a canvas style (Black Canvas, Blurred Background, Face Tracking, etc.) or tap `⚡ Quick Run`.
3. Choose how many clips to generate (3, 5, 10, 20, 50 or custom).
4. Choose clip duration (Automatic, 0-30s, 15-30s, 30s-1m, 1-2m).
💡 *Tip:* Use `↩️ Back` or `❌ Cancel` at any step to change your selections or start over.

───

🌐 *Supported Platforms:*
• ▶️ YouTube & YouTube Shorts
• 🟣 Twitch (VODs & Clips)
• 🟩 Kick (VODs only)
• 📁 Direct video file upload

───

🎬 *What Each Clip Includes:*
• Viral moment cut to 9:16 vertical
• Bold punch-word title caption at top
• Word-by-word subtitles with sentiment colors (🔴red/🟢green/🟡yellow)
• Your watermark logo
• YouTube title & hashtags ready to copy

───

🛠️ *User Commands:*
• `/help` — Show this guide
• `/update` — Check your real-time video status
• `/queue` — View waiting queue
• `/stop` — Cancel active processing
• `/cancel` — Reset wizard options
• `/clear` — Clear pending queue

───

🔐 *Admin-Only Commands:*
• `/users` — View all approved users & live status (🟢Active/⚪Idle)
• `/revoke <user_id>` — Remove a user's access
• `/brief <rules>` — Set campaign rules for all clips

───

🔔 *Access Requests:*
When a new user opens the bot, you will receive a private alert card with `✅ Approve` and `❌ Reject` buttons. Only you can approve or deny access.
"""
    else:
        # ── Regular Approved User Guide ────────────────────────────────────────
        msg = r"""
🎬 *VIRAL Clip Bot — Your Quick Guide*

Welcome! Send any video link and this bot will automatically extract the best viral moments as ready-to-post vertical clips.

───

📐 *4 Simple Steps:*
1. Send a video link (YouTube, Twitch, Kick, or upload a file).
2. Choose your clip style (Black Canvas, Blurred Background, Face Tracking, etc.).
3. Choose how many clips you want (3, 5, 10, 20, or type any number).
4. Choose clip duration (Automatic, 0-30s, 15-30s, 30s-1m, 1-2m).
💡 *Tip:* Use `↩️ Back` or `❌ Cancel` at any step to change your options.

───

🌐 *Supported Platforms:*
• ▶️ YouTube & YouTube Shorts
• 🟣 Twitch (VODs & Clips)
• 🟩 Kick (VODs only)
• 📁 Direct video file upload

───

📦 *What You'll Receive:*
• Top viral moments cut into 9:16 vertical clips.
• Bold punch-word title caption on each clip.
• 🔴🟢🟡 Word-by-word subtitles synced to the speaker's voice.
• A ready-to-copy YouTube title & hashtags under each video.
• A full titles summary card at the end for easy copy-pasting.
• A ZIP file with all clips and titles in one download.

───

🛠️ *Commands:*
• `/help` — Show this guide
• `/update` — Check your processing status
• `/queue` — See how many videos are waiting
• `/stop` — Cancel your active video
• `/cancel` — Reset wizard options
• `/clear` — Clear your waiting queue

───

💡 *Tips:*
• Processing takes a few minutes — you'll be notified when clips are ready.
• You can queue multiple links and they'll process one by one automatically.
• Tap any YouTube title box to copy it instantly on mobile.
"""

    await update.message.reply_text(msg, parse_mode="MarkdownV2")




# Store last submitted video URL per chat_id: chat_id -> str
_last_submitted_url: dict[int, str] = {}


def _get_or_recover_session(chat_id: int, update: Update, query=None, context=None) -> dict | None:
    """Get active link session or auto-recover from session cache or message text so button clicks never expire."""
    session = _pending_links.get(chat_id)
    if session:
        if context and not session.get("context"):
            session["context"] = context
        return session

    # Reconstruct session using stored URL cache or message text
    url = _last_submitted_url.get(chat_id)
    if not url and query and query.message and query.message.text:
        text = query.message.text
        match = _URL_RE.search(text)
        if match:
            url = match.group(0)

    if url:
        session = {
            "url": url,
            "update": update,
            "context": context,
            "layout_mode": "face_crop",
            "enable_watermark": False,
            "enable_subtitles": True,
            "num_clips": 10,
            "target_duration": "auto",
        }
        _pending_links[chat_id] = session
        return session


    return None



def _extract_title_and_caption(caption_text: str, clip_num: str) -> tuple[str, str]:
    """
    Extract the clean SEO title and social caption/hashtags from a video message.
    Guarantees:
    - Never includes meta headers ('🔴 YouTube Title:', '📱 Caption & Hashtags:', '🎬 Clip ...', '💡 ...').
    - Pure high-CTR story title for YouTube/TikTok/Reels.
    - Pure caption body + hashtags for video description.
    """
    if not caption_text:
        return f"Viral Clip #{clip_num} 🔥😂💀", "#Shorts #Viral"

    raw = caption_text

    # 1. Preferred: Extract from <code>...</code> blocks if HTML formatted
    code_blocks = re.findall(r"<code>(.*?)</code>", raw, flags=re.DOTALL)
    if code_blocks:
        code_cleaned = [html.unescape(b).strip() for b in code_blocks if b.strip()]
        # Strip any accidental label inside the code block
        code_cleaned = [
            re.sub(r"(?i)^(?:🔴|📱|📋)?\s*(?:YouTube\s*Title|Caption\s*&?\s*Hashtags?):?\s*", "", b).strip()
            for b in code_cleaned
        ]
        code_cleaned = [b for b in code_cleaned if b]

        if len(code_cleaned) >= 2:
            title = format_seo_title(code_cleaned[0], default_emoji="🔥😂💀")
            description = code_cleaned[1]
            return title, description
        elif len(code_cleaned) == 1:
            block = code_cleaned[0]
            parts = [p.strip() for p in block.split("\n\n") if p.strip()]
            if len(parts) >= 2:
                title = format_seo_title(parts[0], default_emoji="🔥😂💀")
                description = "\n\n".join(parts[1:])
                return title, description

    # 2. Plain text / fallback extraction: Strip HTML tags
    clean_text = re.sub(r"<[^>]+>", "", raw)
    clean_text = html.unescape(clean_text)

    # 3. Filter out meta headers and literal label rows
    lines = []
    for line in clean_text.splitlines():
        line_s = line.strip()
        if not line_s:
            continue
        # Filter out clip headers, reasoning lines, and literal label markers
        if re.match(r"^(?:🎬|📹|💡|🔴|📱|📋)?\s*(?:Clip\s*\d+|Hook\s*Score|Reasoning|YouTube\s*Title|Caption\s*&?\s*Hashtags?):?", line_s, flags=re.IGNORECASE):
            continue
        lines.append(line_s)

    if not lines:
        return f"Viral Clip #{clip_num} 🔥😂💀", "#Shorts #Viral"

    # First non-header line is the clean SEO Title
    title = format_seo_title(lines[0], default_emoji="🔥😂💀")

    # Remaining lines form the description and hashtags
    if len(lines) > 1:
        description = "\n\n".join(lines[1:])
    else:
        description = f"{title}\n\n#Shorts #Viral"

    # Clean any leftover meta prefixes from description
    description = re.sub(r"(?i)^(?:🔴|📱|📋)?\s*(?:YouTube\s*Title|Caption\s*&?\s*Hashtags?):?\s*", "", description).strip()

    return title, description


def _update_keyboard_posting(reply_markup: InlineKeyboardMarkup, platform: str) -> InlineKeyboardMarkup:
    if not reply_markup or not reply_markup.inline_keyboard:
        return reply_markup

    new_rows = []
    for row in reply_markup.inline_keyboard:
        new_row = []
        for btn in row:
            cb_data = btn.callback_data or ""
            if platform == "all" or f"post:{platform}:" in cb_data or f"sched:{platform}:" in cb_data:
                label = "TikTok" if "tiktok" in cb_data else ("YouTube" if "youtube" in cb_data else ("IG Reels" if "instagram" in cb_data else ("Facebook" if "facebook" in cb_data else "ALL")))
                new_row.append(InlineKeyboardButton(text=f"⏳ Posting to {label}...", callback_data=cb_data))
            else:
                new_row.append(btn)
        new_rows.append(new_row)
    return InlineKeyboardMarkup(new_rows)


def _update_keyboard_posted(reply_markup: InlineKeyboardMarkup, platform: str) -> InlineKeyboardMarkup:
    if not reply_markup or not reply_markup.inline_keyboard:
        return reply_markup

    new_rows = []
    for row in reply_markup.inline_keyboard:
        new_row = []
        for btn in row:
            cb_data = btn.callback_data or ""
            btn_text = btn.text
            if platform == "all":
                if "post:all" in cb_data or "sched:all" in cb_data or "post:" in cb_data or "sched:" in cb_data:
                    if "post:all" in cb_data:
                        new_row.append(InlineKeyboardButton(text="✅ Posted to ALL", callback_data=cb_data))
                    elif "sched:all" in cb_data:
                        new_row.append(InlineKeyboardButton(text="✅ Scheduled (Peak)", callback_data=cb_data))
                    elif "tiktok" in cb_data:
                        new_row.append(InlineKeyboardButton(text="✅ TikTok", callback_data=cb_data))
                    elif "youtube" in cb_data:
                        new_row.append(InlineKeyboardButton(text="✅ YouTube", callback_data=cb_data))
                    elif "instagram" in cb_data:
                        new_row.append(InlineKeyboardButton(text="✅ IG Reels", callback_data=cb_data))
                    elif "facebook" in cb_data:
                        new_row.append(InlineKeyboardButton(text="✅ Facebook", callback_data=cb_data))
                    else:
                        new_row.append(btn)
                else:
                    new_row.append(btn)
            elif f"post:{platform}:" in cb_data or f"sched:{platform}:" in cb_data:
                label = "TikTok" if "tiktok" in cb_data else ("YouTube" if "youtube" in cb_data else ("IG Reels" if "instagram" in cb_data else ("Facebook" if "facebook" in cb_data else platform.title())))
                action_word = "Scheduled" if "sched:" in cb_data else "Posted"
                new_row.append(InlineKeyboardButton(text=f"✅ {label}", callback_data=cb_data))
            else:
                new_row.append(btn)
        new_rows.append(new_row)
    return InlineKeyboardMarkup(new_rows)


async def _handle_social_post_button(update: Update, context: ContextTypes.DEFAULT_TYPE, platform: str, clip_num: str) -> None:
    query = update.callback_query
    chat_id = update.effective_chat.id
    msg = query.message if query else None
    logger.info("⚡ [POST BUTTON TAPPED] platform=%s, clip_num=%s, chat_id=%s", platform, clip_num, chat_id)

    webhook_url = os.environ.get("N8N_WEBHOOK_URL", os.environ.get("MAKE_WEBHOOK_URL", f"{get_public_base_url()}/webhook/viral-post")).strip()
    if not webhook_url:
        if query:
            await query.answer("⚠️ No Webhook URL set.", show_alert=True)
        return

    if msg and msg.reply_markup:
        posting_kb = _update_keyboard_posting(msg.reply_markup, platform)
        await _safe_edit_message_reply_markup(query, reply_markup=posting_kb)

    video_url = ""
    raw_caption = msg.caption if msg and msg.caption else ""
    title_text, clean_caption = _extract_title_and_caption(raw_caption, clip_num)
    extracted_hashtags = " ".join([w for w in clean_caption.split() if w.startswith("#")]) or "#Viral #Shorts"
    clips_dir = Path("tmp/clips")
    clips_dir.mkdir(parents=True, exist_ok=True)

    try:
        c_num = int(clip_num)
    except ValueError:
        c_num = 1

    # 1. Instant local disk lookup (0.001s sub-millisecond resolution)
    matching_clips = [
        f for f in clips_dir.glob("*.mp4")
        if f"clip_{c_num:03d}" in f.name or f"clip_{c_num:02d}" in f.name or f"clip_{c_num}." in f.name or f"_{c_num:02d}." in f.name or f"_{c_num:03d}." in f.name
    ]
    if matching_clips:
        video_url = f"{get_public_base_url()}/clips/{matching_clips[0].name}"
        logger.info("  Instant local disk clip match: %s", video_url)

    # 2. Telegram File download fallback (only if file is missing on disk)
    if not video_url and msg and msg.video:
        try:
            safe_fid = "".join(c for c in msg.video.file_id if c.isalnum())[:20]
            public_filename = f"clip_msg_{safe_fid}.mp4"
            public_path = clips_dir / public_filename
            if not public_path.exists():
                tg_file = await asyncio.wait_for(context.bot.get_file(msg.video.file_id), timeout=10.0)
                await asyncio.wait_for(tg_file.download_to_drive(public_path), timeout=25.0)
            if public_path.exists() and public_path.stat().st_size > 1000:
                video_url = f"{get_public_base_url()}/clips/{public_filename}"
                logger.info("  Direct Telegram file resolved: %s", video_url)
        except Exception as f_exc:
            logger.warning("Could not download Telegram video file: %s", f_exc)

    if not video_url:
        video_url = f"{get_public_base_url()}/clips/clip_{c_num:03d}.mp4"

    def _post_json_sync(url: str, post_data: dict) -> tuple[int, str]:
        import json
        import urllib.request
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        data_bytes = json.dumps(post_data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(data_bytes)),
                "User-Agent": "ViralBot/1.0",
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=1.5, context=ctx) as response:
            return response.status, response.read().decode("utf-8", errors="replace")

    target_platforms = ["tiktok", "youtube", "instagram", "facebook"] if platform == "all" else [platform]
    
    # Instantly update keyboard to green posted state
    if msg and msg.reply_markup:
        try:
            posted_kb = _update_keyboard_posted(msg.reply_markup, platform)
            await _safe_edit_message_reply_markup(query, reply_markup=posted_kb)
        except Exception as k_exc:
            logger.warning("Could not update keyboard to posted state: %s", k_exc)

    from .social_publisher import direct_publish_clip

    # 1. Dispatch directly to Social APIs (Zernio for TikTok/YouTube, WoopSocial for Instagram/Facebook)
    pub_results = await asyncio.to_thread(
        direct_publish_clip,
        platform,
        title_text,
        clean_caption,
        extracted_hashtags,
        video_url,
    )
    logger.info("Direct social publish results for clip #%s: %s", clip_num, pub_results)

    success_count = sum(1 for (ok, _) in pub_results.values() if ok)
    display_plat = "ALL PLATFORMS" if platform == "all" else platform.upper()

    # Save to local publication log
    try:
        pub_log = Path("tmp/published_clips.json")
        pub_data = []
        if pub_log.exists():
            pub_data = json.loads(pub_log.read_text(encoding="utf-8"))
        pub_data.append({
            "clip_num": clip_num,
            "platform": platform,
            "video_url": video_url,
            "title": title_text,
            "caption": clean_caption,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        })
        pub_log.write_text(json.dumps(pub_data, indent=2), encoding="utf-8")
    except Exception as log_exc:
        logger.warning("Could not append to published_clips.json: %s", log_exc)

    # Instantly update keyboard on the video to show permanent blue tick / checkmark without popup text spam
    if msg and msg.reply_markup:
        try:
            posted_kb = _update_keyboard_posted(msg.reply_markup, platform)
            await _safe_edit_message_reply_markup(query, reply_markup=posted_kb)
        except Exception as k_exc:
            logger.warning("Could not update keyboard to posted state: %s", k_exc)

    if query:
        plat_label = "ALL Platforms" if platform == "all" else platform.upper()
        await query.answer(f"✅ Dispatched to {plat_label}!", show_alert=False)

    return


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles button taps from Telegram Inline Keyboards."""
    query = update.callback_query
    if not query or not query.data or not update.effective_user:
        return

    data = query.data
    chat_id = update.effective_chat.id

    # Always answer callback query immediately to stop Telegram loading spinner
    try:
        await query.answer()
    except Exception:
        pass

    if data.startswith("app:"):
        if not _is_master_admin(update.effective_user.id):
            await query.answer("🚫 Only the Master Admin can approve users.", show_alert=True)
            return

        parts = data.split(":")
        action = parts[1]
        target_uid = int(parts[2])

        if action == "yes":
            username = parts[3] if len(parts) > 3 else f"User {target_uid}"
            _approved_users_db[target_uid] = {
                "username": username,
                "approved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _save_approved_users(_approved_users_db)
            await _safe_edit_message_text(
                query,
                text=f"✅ **User Approved!**\n\n`{username}` (`{target_uid}`) has been granted access.",
                parse_mode="Markdown",
            )
            try:
                await context.bot.send_message(
                    chat_id=target_uid,
                    text="🎉 **Access Approved!**\n\nYour account has been approved by the Admin. You can now send video links to clip!",
                    parse_mode="Markdown",
                )
            except Exception as exc:
                logger.warning("Could not notify user %d of approval: %s", target_uid, exc)
        else:
            await _safe_edit_message_text(
                query,
                text=f"❌ Access request rejected for User `{target_uid}`.",
                parse_mode="Markdown",
            )
        return

    if data.startswith("done:"):
        done_target = data.split(":")[1]
        await query.answer(f"✅ Already submitted for {done_target.upper()}!", show_alert=True)
        return

    if data.startswith("post:"):
        if not update.effective_user or not _is_operator(update.effective_user.id):
            await query.answer("🔒 Auto-Posting is restricted to approved users.", show_alert=True)
            return
        parts = data.split(":")
        platform = parts[1]
        clip_num = parts[2] if len(parts) > 2 else "1"
        await _handle_social_post_button(update, context, platform, clip_num)
        return


    if data.startswith("sched_all:"):
        parts = data.split(":")
        action = parts[1]   # "confirm" or "skip"
        total_clips = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0

        if action == "skip":
            await query.answer("👍 Got it — manage clips individually.", show_alert=False)
            await _safe_edit_message_text(
                query,
                text="⏭️ <b>Skipped auto-schedule.</b>\n\nUse the buttons on each individual clip to post or schedule them manually.",
                parse_mode="HTML",
            )
            _pending_schedule_sessions.pop(chat_id, None)
            return

        if action == "confirm":
            if not update.effective_user or not _is_operator(update.effective_user.id):
                await query.answer("🔒 Only approved users can schedule all clips.", show_alert=True)
                return

            session_clips = _pending_schedule_sessions.get(chat_id, [])
            if not session_clips:
                await query.answer("⚠️ No clip session found. Please re-process the video.", show_alert=True)
                return

            await query.answer("📅 Scheduling all clips...", show_alert=False)
            await _safe_edit_message_text(
                query,
                text=f"⏳ <b>Scheduling {len(session_clips)} clips to peak slots...</b>",
                parse_mode="HTML",
            )

            platforms = ["tiktok", "instagram", "facebook", "youtube"]
            scheduled_lines = []

            for clip in session_clips:
                clip_num = clip.get("clip_num", "?")
                payload_base = clip.get("payload", {})
                staggered_map = scheduler.schedule_clip_staggered(platforms, payload_base)
                from zoneinfo import ZoneInfo
                ET = ZoneInfo("America/New_York")
                for plat, fire_at in staggered_map.items():
                    slot_str = fire_at.astimezone(ET).strftime("%b %d %I:%M%p ET")
                    emoji = {"tiktok": "📱", "instagram": "📸", "facebook": "📘", "youtube": "🔴"}.get(plat, "🌐")
                    scheduled_lines.append(f"{emoji} Clip #{clip_num} → {slot_str}")

            # Send schedule confirmation with cancel button for each clip
            preview_text = "\n".join(scheduled_lines[:40])  # cap display at 40 lines
            if len(scheduled_lines) > 40:
                preview_text += f"\n<i>...and {len(scheduled_lines) - 40} more slots</i>"

            cancel_kb = InlineKeyboardMarkup([
                [InlineKeyboardButton("❌ Cancel All Scheduled Posts", callback_data="sched_all:cancel_all:0")],
            ])
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"✅ <b>All {len(session_clips)} clips scheduled!</b>\n\n"
                    f"{preview_text}\n\n"
                    f"💡 <i>Use /schedule to review. Tap ❌ Cancel All to wipe the queue if needed.</i>"
                ),
                parse_mode="HTML",
                reply_markup=cancel_kb,
            )
            _pending_schedule_sessions.pop(chat_id, None)
            return

        if action == "cancel_all":
            if not update.effective_user or not _is_operator(update.effective_user.id):
                await query.answer("🔒 Only approved operators can cancel all.", show_alert=True)
                return
            # Clear entire queue for this chat
            scheduler._queue = [item for item in scheduler._queue if item.get("chat_id") != chat_id]
            scheduler._save()
            await query.answer("🗑️ All scheduled posts cancelled.", show_alert=True)
            await _safe_edit_message_text(
                query,
                text="🗑️ <b>All your scheduled posts have been cancelled.</b>\n\nThe queue is now empty. Use the individual clip buttons to re-schedule.",
                parse_mode="HTML",
            )
    if data.startswith("discard:"):
        if not update.effective_user or not _is_operator(update.effective_user.id):
            await query.answer("🔒 Only approved operators can discard clips.", show_alert=True)
            return
        clip_num = data.split(":")[1] if ":" in data else "?"
        await query.answer(f"🗑️ Clip #{clip_num} Discarded!", show_alert=True)
        await _safe_edit_message_caption(
            query,
            caption=f"🗑️ <b>Clip #{clip_num} DISCARDED</b>\n\n<i>This clip has been removed from QA review.</i>",
            parse_mode="HTML",
            reply_markup=None,
        )
        return

    if data.startswith("sched:"):


        if not update.effective_user or not _is_operator(update.effective_user.id):
            await query.answer("🔒 Scheduling is restricted to approved users.", show_alert=True)
            return
        parts = data.split(":")
        sched_platform = parts[1]  # "all" or specific platform
        clip_num = parts[2] if len(parts) > 2 else "1"
        msg = query.message if query else None
        raw_caption = msg.caption if msg and msg.caption else ""
        title_text, clean_caption = _extract_title_and_caption(raw_caption, clip_num)
        extracted_hashtags = " ".join([w for w in clean_caption.split() if w.startswith("#")]) or "#Viral #Shorts"

        video_url = ""
        clips_dir = Path("tmp/clips")
        clips_dir.mkdir(parents=True, exist_ok=True)
        try:
            c_num = int(clip_num)
        except ValueError:
            c_num = 1

        # 1. Instant local disk lookup
        matching_clips = [
            f for f in clips_dir.glob("*.mp4")
            if f"clip_{c_num:03d}" in f.name or f"clip_{c_num:02d}" in f.name or f"clip_{c_num}." in f.name or f"_{c_num:02d}." in f.name or f"_{c_num:03d}." in f.name
        ]
        if matching_clips:
            video_url = f"{get_public_base_url()}/clips/{matching_clips[0].name}"

        # 2. Telegram File download fallback (only if file missing from local disk)
        if not video_url and msg and msg.video:
            try:
                safe_fid = "".join(c for c in msg.video.file_id if c.isalnum())[:20]
                public_filename = f"clip_msg_{safe_fid}.mp4"
                public_path = clips_dir / public_filename
                if not public_path.exists():
                    tg_file = await asyncio.wait_for(context.bot.get_file(msg.video.file_id), timeout=30.0)
                    await asyncio.wait_for(tg_file.download_to_drive(public_path), timeout=45.0)
                if public_path.exists() and public_path.stat().st_size > 1000:
                    video_url = f"{get_public_base_url()}/clips/{public_filename}"
            except Exception as exc:
                logger.warning("Scheduler: could not resolve video URL from Telegram msg: %s", exc)

        if not video_url:
            video_url = f"{get_public_base_url()}/clips/clip_{c_num:03d}.mp4"

        platforms = ["tiktok", "instagram", "facebook", "youtube"] if sched_platform == "all" else [sched_platform]

        base_payload = {
            "clip_id": f"clip_{int(clip_num):03d}",
            "title": title_text,
            "caption": clean_caption,
            "description": clean_caption,
            "video_url": video_url,
            "video_filename": f"clip_{int(clip_num):03d}.mp4",
            "mime_type": "video/mp4",
            "hashtags": extracted_hashtags,
            "chat_id": chat_id,
        }

        scheduler.schedule_clip_staggered(platforms, base_payload)

        # Update button to scheduled checkmark state without popup spam text
        if msg and msg.reply_markup:
            try:
                posted_kb = _update_keyboard_posted(msg.reply_markup, sched_platform)
                await _safe_edit_message_reply_markup(query, reply_markup=posted_kb)
            except Exception as k_exc:
                logger.warning("Could not update keyboard to scheduled state: %s", k_exc)

        await query.answer("📅 Scheduled for peak time!", show_alert=False)
        return

    if data.startswith("pub:mobile:"):
        clip_num = data.split("pub:mobile:")[1]
        msg = query.message if query else None
        raw_caption = msg.caption if msg and msg.caption else ""
        title_text, clean_caption = _extract_title_and_caption(raw_caption, clip_num)

        # Split caption body from hashtag block
        caption_parts = clean_caption.split("\n\n")
        caption_body = caption_parts[0].strip() if caption_parts else clean_caption
        inline_tags = " ".join([w for w in clean_caption.split() if w.startswith("#")])
        rich_tags = generate_rich_hashtags(topic=title_text, existing_hashtags=inline_tags)

        # YouTube Shorts: Clean SEO title with 2-3 emojis + full description
        yt_copy = format_seo_title(title_text, default_emoji="🔥😂💀")
        yt_desc = f"{caption_body}\n\n{rich_tags}"

        # Unified TikTok / Instagram / Facebook Reels block
        unified_copy = f"{caption_body} 🔥😂💀\n\n{rich_tags}"

        await query.answer(f"📋 Copy captions for Clip #{clip_num}!", show_alert=False)

        copy_card = (
            f"📋 <b>Clip #{clip_num} — Tap to Copy &amp; Post Manually</b>\n\n"
            f"🔴 <b>YouTube Shorts Title</b> <i>(2–3 emojis, high CTR):</i>\n"
            f"<code>{html.escape(yt_copy)}</code>\n\n"
            f"📝 <b>YouTube Shorts Description &amp; Hashtags:</b>\n"
            f"<code>{html.escape(yt_desc)}</code>\n\n"
            f"📱 <b>TikTok / Instagram / Facebook Reels Caption &amp; Hashtags:</b>\n"
            f"<code>{html.escape(unified_copy)}</code>\n\n"
            f"<i>💡 Tap any code box once to copy, then paste directly into the app!</i>"
        )

        quick_links_kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🎵 Open TikTok", url="https://www.tiktok.com/"),
                InlineKeyboardButton("📸 Open Instagram", url="https://www.instagram.com/"),
            ],
            [
                InlineKeyboardButton("📘 Open Facebook", url="https://www.facebook.com/"),
                InlineKeyboardButton("🔴 Open YouTube", url="https://www.youtube.com/"),
            ],
        ])

        await query.message.reply_text(copy_card, reply_markup=quick_links_kb, parse_mode="HTML")
        return



    # Handle Wizard Undo / Back / Cancel Navigation
    if data == "wiz:cancel":
        _pending_links.pop(chat_id, None)
        await _safe_edit_message_text(query, text="❌ Setup cancelled. Send any video link whenever you're ready!")
        return

    if data == "wiz:back_layout":
        session = _get_or_recover_session(chat_id, update, query, context)
        url = session.get("url", "") if session else ""
        title_label = f"🎬 **Video URL:**\n`{url}`" if url else "🎬 **Video link received!**"
        await _safe_edit_message_text(
            query,
            text=(
                f"{title_label}\n\n"
                f"🎨 **Choose your Video Canvas Background Style:**\n\n"
                f"Select a background color or style below, or tap `⚡ Quick Run` for instant Black Canvas:"
            ),
            reply_markup=_make_layout_keyboard(),
            parse_mode="Markdown",
        )
        return

    if data == "wiz:back_wm":
        session = _get_or_recover_session(chat_id, update, query, context)
        layout_mode = session.get("layout_mode", "black_canvas") if session else "black_canvas"
        mode_label = "👤 Face Tracking" if layout_mode == "face_crop" else "🎬 Blurred Background" if layout_mode == "blurred_frame" else "🖤 Black Canvas"
        await _safe_edit_message_text(
            query,
            text=(
                f"✅ **Layout selected:** *{mode_label}*\n\n"
                f"🏷️ **Does this video have a watermark logo?**"
            ),
            reply_markup=_make_watermark_keyboard(),
            parse_mode="Markdown",
        )
        return

    if data == "wiz:back_sub":
        session = _get_or_recover_session(chat_id, update, query, context)
        layout_mode = session.get("layout_mode", "black_canvas") if session else "black_canvas"
        mode_label = "👤 Face Tracking" if layout_mode == "face_crop" else "🎬 Blurred Background" if layout_mode == "blurred_frame" else "🖤 Black Canvas"
        await _safe_edit_message_text(
            query,
            text=(
                f"✅ **Layout selected:** *{mode_label}*\n\n"
                f"💬 **Include Word-by-Word Subtitles on the clips?**"
            ),
            reply_markup=_make_subtitles_keyboard(),
            parse_mode="Markdown",
        )
        return

    if data == "wiz:back_clips":
        session = _get_or_recover_session(chat_id, update, query, context)
        layout_mode = session.get("layout_mode", "black_canvas") if session else "black_canvas"
        mode_label = "👤 Face Tracking" if layout_mode == "face_crop" else "🎬 Blurred Background" if layout_mode == "blurred_frame" else "🖤 Black Canvas"
        sub_label = "💬 Enabled" if session.get("enable_subtitles", True) else "🚫 Disabled"
        await _safe_edit_message_text(
            query,
            text=(
                f"✅ **Layout:** *{mode_label}* | **Subtitles:** *{sub_label}*\n\n"
                f"✂️ **How many top clips would you like to generate?**\n\n"
                f"Select an option below, or reply with any custom number (e.g., 3, 5, 10, 20):"
            ),
            reply_markup=_make_clips_keyboard(),
            parse_mode="Markdown",
        )
        return

    # Step 1: Layout Selected -> Prompt for Watermark (Step 2)
    if data == "fmt:quick_run":
        session = _get_or_recover_session(chat_id, update, query, context)
        if not session:
            await _safe_edit_message_text(query, text="⚠️ Session expired. Please re-send your video URL.")
            return
        session["layout_mode"] = "black_canvas"
        session["enable_watermark"] = False
        session["enable_subtitles"] = True
        session["wm_name"] = "None"
        await _launch_job(chat_id, 10, edit_message=query.message)
        return



    if data.startswith("fmt:"):
        layout_mode = data.split("fmt:")[1]
        session = _get_or_recover_session(chat_id, update, query, context)
        if not session:
            await _safe_edit_message_text(query, text="⚠️ Session expired. Please re-send your video URL.")
            return

        session["layout_mode"] = layout_mode
        mode_label = "👤 Face Tracking" if layout_mode == "face_crop" else "🎬 Blurred Background" if layout_mode == "blurred_frame" else "🖤 Black Canvas"

        await _safe_edit_message_text(
            query,
            text=(
                f"✅ **Layout selected:** *{mode_label}*\n\n"
                f"🏷️ **Does this video have a watermark logo?**"
            ),
            reply_markup=_make_watermark_keyboard(),
            parse_mode="Markdown",
        )
        return

    # Step 2: Watermark Option Tapped
    if data.startswith("wm:"):
        choice = data.split("wm:")[1]
        session = _get_or_recover_session(chat_id, update, query, context)
        if not session:
            await _safe_edit_message_text(query, text="⚠️ Session expired. Please re-send your video URL.")
            return

        current_layout = session.get("layout_mode", "black_canvas")
        mode_label = "👤 Face Tracking" if current_layout == "face_crop" else "🎬 Blurred Background" if current_layout == "blurred_frame" else "🖤 Black Canvas"

        if choice == "yes":
            session["awaiting_wm"] = True
            await _safe_edit_message_text(
                query,
                text=(
                    f"✅ **Layout selected:** *{mode_label}*\n\n"
                    f"📸 **Send your PNG Watermark Logo Image now!**\n\n"
                    f"Attach or drag & drop your transparent `.png` logo image in Telegram chat."
                ),
                parse_mode="Markdown",
            )
            return
        else:
            session["enable_watermark"] = False
            session["wm_name"] = "None"
            await _safe_edit_message_text(
                query,
                text=(
                    f"✅ **Layout selected:** *{mode_label}*\n\n"
                    f"💬 **Include Word-by-Word Subtitles on the clips?**"
                ),
                reply_markup=_make_subtitles_keyboard(),
                parse_mode="Markdown",
            )
            return

    # Step 3: Subtitles Option Tapped -> Prompt for Clip Count
    if data.startswith("sub:"):
        choice = data.split("sub:")[1]
        session = _get_or_recover_session(chat_id, update, query, context)
        if not session:
            await _safe_edit_message_text(query, text="⚠️ Session expired. Please re-send your video URL.")
            return

        session["enable_subtitles"] = (choice == "yes")
        session["awaiting_clip_count"] = True

        current_layout = session.get("layout_mode", "black_canvas")
        mode_label = "👤 Face Tracking" if current_layout == "face_crop" else "🎬 Blurred Background" if current_layout == "blurred_frame" else "🖤 Black Canvas"
        sub_label = "💬 Enabled" if session["enable_subtitles"] else "🚫 Disabled"

        await _safe_edit_message_text(
            query,
            text=(
                f"✅ **Layout:** *{mode_label}* | **Subtitles:** *{sub_label}*\n\n"
                f"✂️ **How many top clips would you like to generate?**\n\n"
                f"Select an option below, or reply with any custom number (e.g., 3, 5, 10, 20):"
            ),
            reply_markup=_make_clips_keyboard(),
            parse_mode="Markdown",
        )
        return

    # Step 4: Clip Count Option Tapped -> Prompt for Clip Duration (Step 4)
    if data.startswith("clips:"):
        try:
            num_clips = int(data.split("clips:")[1])
        except ValueError:
            num_clips = 10

        session = _get_or_recover_session(chat_id, update, query, context)
        if not session:
            await _safe_edit_message_text(query, text="⚠️ Session expired. Please re-send your video URL.")
            return

        session["num_clips"] = num_clips
        current_layout = session.get("layout_mode", "black_canvas")
        mode_label = "👤 Face Tracking" if current_layout == "face_crop" else "🎬 Blurred Background" if current_layout == "blurred_frame" else "🖤 Black Canvas"
        sub_label = "💬 Enabled" if session.get("enable_subtitles", True) else "🚫 Disabled"

        await _safe_edit_message_text(
            query,
            text=(
                f"✅ **Layout:** *{mode_label}* | **Subtitles:** *{sub_label}* | **Clips:** *{num_clips}*\n\n"
                f"⏱️ **How long should each clip be?**\n\n"
                f"Select a target duration below, or tap `⚡ Automatic` for AI-optimized story length:"
            ),
            reply_markup=_make_duration_keyboard(),
            parse_mode="Markdown",
        )
        return

    # Step 4: Clip Duration Option Tapped -> Launch Job
    if data.startswith("dur:"):
        target_duration = data.split("dur:")[1]
        session = _get_or_recover_session(chat_id, update, query, context)
        if not session:
            await _safe_edit_message_text(query, text="⚠️ Session expired. Please re-send your video URL.")
            return


        num_clips = session.get("num_clips", 10)
        await _launch_job(chat_id, num_clips, target_duration=target_duration, edit_message=query.message)
        return


async def handle_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /update (or /status) — displays real-time status for the calling user ONLY."""
    if not update.effective_user or not _is_operator(update.effective_user.id):
        return

    chat_id = update.effective_chat.id
    from bot.run_pipeline import active_run_status

    task_tuple = _active_tasks_by_chat.get(chat_id)
    is_active = task_tuple is not None and not task_tuple[0].done()

    user_pending_count = sum(
        1 for item in list(_job_queue._queue)
        if item[1] and item[1].effective_chat and item[1].effective_chat.id == chat_id
    )

    if not is_active and user_pending_count == 0:
        await update.message.reply_text("ℹ️ You currently have no active videos processing or in queue. Send any video link to start clipping!")
        return

    if is_active:
        user_status = active_run_status.get(chat_id, {})
        elapsed_sec = int(time.time() - user_status.get("start_time", time.time()))
        elapsed_mins = elapsed_sec // 60
        rem_sec = elapsed_sec % 60
        time_str = f"{elapsed_mins}m {rem_sec}s" if elapsed_mins > 0 else f"{elapsed_sec}s"

        info = user_status.get("streamer_info", {})
        streamer = info.get("streamer", "Streamer")
        title = info.get("title", "N/A")
        platform = info.get("platform", "Video")
        step = user_status.get("step", "Processing...")
        layout_mode = user_status.get("layout_mode", "pillarbox")

        mode_label = "🎬 Blurred Background" if layout_mode in ("blurred_frame", "pillarbox", "default") else "👤 Face Tracking"

        msg = (
            f"📊 *Your Video Status Dashboard*\n\n"
            f"• *Status:* ⚙️ {step}\n"
            f"• *Platform:* {platform}\n"
            f"• *Streamer:* {streamer}\n"
            f"• *Title:* {title if title else 'N/A'}\n"
            f"• *Layout:* {mode_label}\n"
            f"• *Elapsed Time:* ⏱️ {time_str}\n\n"
            f"⚡ _Your video is actively processing. Highlights will be delivered here automatically!_"
        )
        if user_pending_count > 0:
            msg += f"\n\n📥 *Pending in Queue:* You also have {user_pending_count} additional video(s) waiting."
        await update.message.reply_text(msg, parse_mode="Markdown")
        return

    msg = (
        f"📥 *Your Video Queue Status*\n\n"
        f"• *Status:* Waiting in queue\n"
        f"• *Pending Videos:* {user_pending_count} video(s)\n\n"
        f"⚡ Processing will begin automatically!"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def handle_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /stop — cancels the calling user's active processing run ONLY."""
    if not update.effective_user or not _is_operator(update.effective_user.id):
        return
    chat_id = update.effective_chat.id

    task_tuple = _active_tasks_by_chat.get(chat_id)
    if task_tuple and not task_tuple[0].done():
        task_tuple[0].cancel()
        await update.message.reply_text("🛑 Cancelled your active video processing run!")
    else:
        await update.message.reply_text("ℹ️ You have no active video processing run to stop.")


async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /cancel — resets wizard selections."""
    if not update.effective_user or not _is_operator(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    _pending_links.pop(chat_id, None)
    await update.message.reply_text("❌ Setup cancelled. Send any video link whenever you're ready to start!")


async def handle_queue(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /queue — displays waiting queue count for the calling user ONLY."""
    if not update.effective_user or not _is_operator(update.effective_user.id):
        return
    chat_id = update.effective_chat.id

    user_pending = sum(
        1 for item in list(_job_queue._queue)
        if item[1] and item[1].effective_chat and item[1].effective_chat.id == chat_id
    )
    is_active = (chat_id in _active_tasks_by_chat) and not _active_tasks_by_chat[chat_id][0].done()

    if is_active:
        active_url = _active_tasks_by_chat[chat_id][1]
        msg = (
            f"📥 *Your Queue Status*\n\n"
            f"• *Currently Processing:* `{active_url}`\n"
            f"• *Waiting in Queue:* {user_pending} video(s)"
        )
    elif user_pending > 0:
        msg = (
            f"📥 *Your Queue Status*\n\n"
            f"• *Currently Processing:* None\n"
            f"• *Waiting in Queue:* {user_pending} video(s)"
        )
    else:
        msg = "ℹ️ Your queue is completely empty. Send any video link to start clipping!"

    await update.message.reply_text(msg, parse_mode="Markdown")


async def handle_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /clear — empties pending video jobs for the calling user ONLY."""
    if not update.effective_user or not _is_operator(update.effective_user.id):
        return
    chat_id = update.effective_chat.id

    cleared_count = 0
    remaining_items = []

    while not _job_queue.empty():
        try:
            item = _job_queue.get_nowait()
            u = item[1]
            cid = u.effective_chat.id if u and u.effective_chat else 0
            if cid == chat_id:
                cleared_count += 1
                _job_queue.task_done()
            else:
                remaining_items.append(item)
        except asyncio.QueueEmpty:
            break

    for item in remaining_items:
        await _job_queue.put(item)

    if cleared_count > 0:
        await update.message.reply_text(f"🗑️ Cleared {cleared_count} video(s) from your queue.")
    else:
        await update.message.reply_text("ℹ️ Your queue was already empty.")


async def handle_brief(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /brief — Set or clear custom AI campaign brief rules for clip scoring."""
    if not update.effective_user or not _is_operator(update.effective_user.id):
        await _handle_unapproved_user(update, context)
        return

    chat_id = update.effective_chat.id
    args = context.args if (context and context.args) else []
    text = " ".join(args).strip()

    if not text:
        current = _campaign_briefs.get(chat_id, "")
        if current:
            msg = (
                f"🟢 **Active Campaign Brief:**\n\n"
                f"_{current}_\n\n"
                f"💡 *To update:* `/brief <new rules>`\n"
                f"💡 *To clear:* `/brief clear`"
            )
        else:
            msg = (
                f"🔴 **No Campaign Brief Active.**\n\n"
                f"Send custom rules to guide AI clip selection:\n"
                f"• `/brief Focus on funny reactions & Kai rage moments`\n"
                f"• `/brief Only clip moments where they talk about money or drama`\n"
                f"• `/brief Highlight Speed and Kai gaming moments`"
            )
        await update.message.reply_text(msg, parse_mode="Markdown")
        return

    if text.lower() in ("clear", "off", "reset", "none", "disable"):
        _campaign_briefs.pop(chat_id, None)
        await update.message.reply_text("🔴 **Campaign Brief Cleared.** Default viral AI scoring is active.", parse_mode="Markdown")
        return

    _campaign_briefs[chat_id] = text
    await update.message.reply_text(
        f"🟢 **Campaign Brief Saved!**\n\n"
        f"AI will now enforce these rules on all your video runs:\n"
        f"_{text}_\n\n"
        f"💡 *To remove rules anytime, send:* `/brief clear`",
        parse_mode="Markdown",
    )


async def handle_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /schedule — shows pending scheduled posts for this user."""
    if not update.effective_user or not _is_operator(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    summary = scheduler.get_pending_summary(chat_id)
    await update.message.reply_text(summary, parse_mode="HTML")



async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id if update.effective_user else 0
    chat_id = update.effective_chat.id if update.effective_chat else 0

    logger.info(
        "Incoming Telegram message: user_id=%s, chat_id=%s, text=%r, photo=%s, doc=%s",
        user_id, chat_id, update.message.text, bool(update.message.photo), bool(update.message.document),
    )

    if not _is_operator(user_id):
        logger.info("Routing unapproved user %d to approval request flow...", user_id)
        await _handle_unapproved_user(update, context)
        return

    session = _pending_links.get(chat_id)

    # 1. Check if user is replying with a custom number of clips during clip count selection step
    if session and session.get("awaiting_clip_count") and update.message.text:
        text = update.message.text.strip()
        m = re.search(r"\d+", text)
        if m:
            num_clips = int(m.group(0))
            if 1 <= num_clips <= 200:
                session["awaiting_clip_count"] = False
                session["num_clips"] = num_clips
                await update.message.reply_text(
                    f"✅ **Target Clips set to {num_clips} clips!**\n\n"
                    f"⏱️ **How long should each clip be?**",
                    reply_markup=_make_duration_keyboard(),
                    parse_mode="Markdown",
                )
                return

    # 2. Handle watermark PNG image / document / photo / sticker / attachment uploads
    photo = update.message.photo
    doc = update.message.document
    sticker = update.message.sticker
    video = update.message.video

    is_media = bool(photo) or bool(doc) or bool(sticker) or bool(update.message.effective_attachment)
    is_video = bool(video) or (doc and doc.mime_type and "video" in doc.mime_type)

    if is_media and not is_video:
        logger.info("Processing uploaded watermark image from user %d...", user_id)
        if photo:
            file_id = photo[-1].file_id
        elif doc:
            file_id = doc.file_id
        elif sticker:
            file_id = sticker.file_id
        elif update.message.effective_attachment and hasattr(update.message.effective_attachment, "file_id"):
            file_id = update.message.effective_attachment.file_id
        else:
            file_id = None

        if file_id:
            caption_text = (update.message.caption or "").strip()
            wm_dir = Path("tmp/watermarks")
            wm_dir.mkdir(parents=True, exist_ok=True)

            if caption_text:
                clean_name = re.sub(r"[^\w\s-]", "", caption_text).strip().lower().replace(" ", "_")
                wm_target = wm_dir / f"{clean_name}.png"
            else:
                wm_target = wm_dir / f"uploaded_{int(time.time())}.png"

            status_msg = await update.message.reply_text("📥 Saving watermark image on Oracle VM... ⏳")

            tg_file = None
            for attempt in range(3):
                try:
                    tg_file = await context.bot.get_file(file_id, read_timeout=60.0, write_timeout=60.0)
                    await tg_file.download_to_drive(str(wm_target), read_timeout=60.0, write_timeout=60.0)
                    break
                except Exception as exc:
                    logger.warning("Telegram watermark download attempt %d failed: %s", attempt + 1, exc)
                    await asyncio.sleep(1.5)

            if not tg_file or not wm_target.exists():
                await status_msg.edit_text("❌ Failed to download watermark image from Telegram. Please re-send the PNG image.")
                return

            import shutil
            shutil.copy(wm_target, Path("tmp/custom_watermark.png"))

            if session and session.get("awaiting_wm"):
                session["enable_watermark"] = True
                session["wm_name"] = wm_target.name
                session["awaiting_wm"] = False
                session["awaiting_clip_count"] = True

                await status_msg.edit_text(
                    f"✅ **Watermark Received & Attached!** (`{wm_target.name}`)\n\n"
                    f"✂️ **How many top clips would you like to generate?**\n\n"
                    f"Select an option below, or reply with any custom number (e.g., 2, 25, 100):",
                    reply_markup=_make_clips_keyboard(),
                    parse_mode="Markdown",
                )
            else:
                await status_msg.edit_text(
                    f"✅ **Watermark Received & Saved!** (`{wm_target.name}`)\n\n"
                    f"It is active and will be auto-applied directly below the video frame on all generated clips.",
                    parse_mode="Markdown",
                )
            return

        if caption_text:
            clean_name = re.sub(r"[^\w\s-]", "", caption_text).strip().lower().replace(" ", "_")
            wm_target = wm_dir / f"{clean_name}.png"
        else:
            wm_target = wm_dir / f"uploaded_{int(time.time())}.png"

        status_msg = await update.message.reply_text("📥 Saving watermark image on Oracle VM... ⏳")
        tg_file = await context.bot.get_file(file_id)
        await tg_file.download_to_drive(str(wm_target))

        import shutil
        shutil.copy(wm_target, Path("tmp/custom_watermark.png"))

        if session and session.get("awaiting_wm"):
            session["enable_watermark"] = True
            session["wm_name"] = wm_target.name
            session["awaiting_wm"] = False
            session["awaiting_clip_count"] = True

            await status_msg.edit_text(
                f"✅ **Watermark Received & Attached!** (`{wm_target.name}`)\n\n"
                f"✂️ **How many top clips would you like to generate?**\n\n"
                f"Select an option below, or reply with any custom number (e.g., 2, 25, 100):",
                reply_markup=_make_clips_keyboard(),
                parse_mode="Markdown",
            )
        else:
            await status_msg.edit_text(
                f"✅ **Watermark Received & Saved!** (`{wm_target.name}`)\n\n"
                f"It is active and will be auto-applied directly below the video frame on all generated clips.",
                parse_mode="Markdown",
            )
        return

    # 3. Handle video URL links or direct video file uploads
    video_obj = update.message.video or (doc and doc.mime_type and "video" in doc.mime_type)
    url_target = None

    if video_obj:
        file_id = doc.file_id if doc else update.message.video.file_id
        file_name = getattr(doc or update.message.video, "file_name", "desktop_video.mp4") or "desktop_video.mp4"
        upload_dir = Path("tmp/telegram_uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        local_target = upload_dir / f"{time.time_ns()}_{file_name}"

        status_msg = await update.message.reply_text("📥 Downloading your video file on Oracle VM... ⏳")
        tg_file = await context.bot.get_file(file_id)
        await tg_file.download_to_drive(str(local_target))
        url_target = str(local_target.resolve())
        await status_msg.edit_text("✅ Video file downloaded to VM!")
    elif update.message.text:
        text = update.message.text.strip()
        match = _URL_RE.search(text)
        if match:
            url_target = match.group(0)

    if not url_target:
        await handle_help(update, context)
        return

    _last_submitted_url[chat_id] = url_target
    _pending_links[chat_id] = {
        "url": url_target,
        "update": update,
        "context": context,
    }


    keyboard = _make_layout_keyboard()

    title_label = f"📁 **Desktop Video File Received!**\n`{Path(url_target).name}`" if Path(url_target).exists() else f"🎬 **Video URL received!**\n`{url_target}`"

    prompt_text = (
        f"{title_label}\n\n"
        f"🎨 **Choose your Video Canvas Background Style:**\n\n"
        f"Select a background color or style below, or tap `⚡ Quick Run` for instant Black Canvas:"
    )

    await update.message.reply_text(
        prompt_text,
        reply_markup=keyboard,
        parse_mode="Markdown",
    )



