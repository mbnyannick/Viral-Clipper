"""
Telegram message handler & queue manager with interactive commands (/help, /stop, /queue, /clear, /update)
and Inline Keyboard Layout Selector (Pillarbox vs. Face-Crop).
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from bot.run_pipeline import run_pipeline

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

# Persistent approved users database path
APPROVED_USERS_FILE = Path("approved_users.json")


def _make_layout_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("⚡ Quick Run (Black Canvas Default)", callback_data="fmt:quick_run"),
        ],
        [
            InlineKeyboardButton("🖤 Black Canvas", callback_data="fmt:black_canvas"),
            InlineKeyboardButton("🎬 Blurred Background", callback_data="fmt:blurred_frame"),
        ],
        [
            InlineKeyboardButton("🔴 Red Canvas", callback_data="fmt:red_canvas"),
            InlineKeyboardButton("🔵 Blue Canvas", callback_data="fmt:blue_canvas"),
            InlineKeyboardButton("🟣 Purple Canvas", callback_data="fmt:purple_canvas"),
        ],
        [
            InlineKeyboardButton("👤 Face Tracking", callback_data="fmt:face_crop"),
        ],
        [
            InlineKeyboardButton("❌ Cancel", callback_data="wiz:cancel"),
        ],
    ])


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
            InlineKeyboardButton("⚡ Automatic (Recommended 25–60s)", callback_data="dur:auto"),
        ],
        [
            InlineKeyboardButton("⏱️ 0 to 30 Seconds", callback_data="dur:0_30"),
            InlineKeyboardButton("⏱️ 15 to 30 Seconds", callback_data="dur:15_30"),
        ],
        [
            InlineKeyboardButton("⏱️ 30s to 1 Minute", callback_data="dur:30_60"),
            InlineKeyboardButton("⏱️ 1 to 2 Minutes", callback_data="dur:60_120"),
        ],
        [
            InlineKeyboardButton("↩️ Back to Clip Count", callback_data="wiz:back_clips"),
            InlineKeyboardButton("❌ Cancel", callback_data="wiz:cancel"),
        ],
    ])


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


# Queue tuple: (url, update, context, layout_mode, enable_watermark, enable_silence_cut, top_n_clips, campaign_brief, target_duration)
_job_queue: asyncio.Queue[tuple[str, Update, ContextTypes.DEFAULT_TYPE, str, bool, bool, int, str, str]] = asyncio.Queue()
_worker_task: asyncio.Task | None = None
_current_job_url: str | None = None
_active_pipeline_task: asyncio.Task | None = None

# Store pending link submissions awaiting button clicks: chat_id -> dict
_pending_links: dict[int, dict] = {}
# Store active campaign brief per chat_id: chat_id -> str
_campaign_briefs: dict[int, str] = {}


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
        "black_canvas": "🖤 Black Canvas",
        "blurred_frame": "🎬 Blurred Background",
        "red_canvas": "🔴 Red Canvas",
        "blue_canvas": "🔵 Blue Canvas",
        "purple_canvas": "🟣 Purple Canvas",
        "face_crop": "👤 Face Tracking",
    }
    mode_label = mode_map.get(layout_mode, "🖤 Black Canvas")
    wm_label = f"`{wm_name}`" if enable_wm else "*None*"

    campaign_brief = _campaign_briefs.get(chat_id, "")
    brief_status = "🟢 Active" if campaign_brief else "🔴 None"

    status_text = (
        f"✅ **Options Selected!**\n\n"
        f"• **Layout:** {mode_label}\n"
        f"• **Watermark:** {wm_label}\n"
        f"• **Campaign Brief:** {brief_status}\n"
        f"• **Target Clips:** *{num_clips} clips*\n"
        f"• **Clip Duration:** {dur_label}\n\n"
        f"🚀 **Video processing has started!** Please wait while clips are generated and composited... ⏳"
    )

    if edit_message:
        try:
            await edit_message.edit_text(status_text, parse_mode="Markdown")
        except Exception:
            await orig_update.message.reply_text(status_text, parse_mode="Markdown")
    else:
        await orig_update.message.reply_text(status_text, parse_mode="Markdown")

    _ensure_worker_running()
    is_user_active = (chat_id in _active_tasks_by_chat) and not _active_tasks_by_chat[chat_id][0].done()
    await _job_queue.put((url, orig_update, orig_context, layout_mode, enable_wm, True, num_clips, campaign_brief, target_duration))

    if is_user_active:
        user_pos = sum(
            1 for item in list(_job_queue._queue)
            if item[1] and item[1].effective_chat and item[1].effective_chat.id == chat_id
        )
        await orig_update.message.reply_text(
            f"📥 **Added to your queue!** (Position #{user_pos})\n\nProcessing will start automatically once your active video completes."
        )


async def _queue_worker() -> None:
    """Background worker that continuously pulls and processes jobs from the queue."""
    while True:
        job = await _job_queue.get()
        url, update, context, layout_mode, enable_wm, enable_silence, top_n_clips, campaign_brief, target_duration = job
        chat_id = update.effective_chat.id if update and update.effective_chat else 0

        try:
            logger.info(
                "Worker picked up job for user %d: %s (mode=%s, wm=%s, clips=%d, dur=%s, Queue remaining: %d)",
                chat_id, url, layout_mode, enable_wm, top_n_clips, target_duration, _job_queue.qsize()
            )
            task = asyncio.create_task(
                run_pipeline(
                    url, update, context,
                    layout_mode=layout_mode,
                    enable_watermark=enable_wm,
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


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Command /help (or /start) — shows admin guide to Master Admin and user guide to approved users."""
    if not update.effective_user or not _is_operator(update.effective_user.id):
        await _handle_unapproved_user(update, context)
        return

    if _is_master_admin(update.effective_user.id):
        # ── Master Admin Guide ─────────────────────────────────────────────────
        msg = (
            "👑 *VIRAL Clip Bot — Admin Control Panel*\n\n"
            "Welcome back, Admin\! You have full control over this bot and all approved users\.\n\n"
            "───\n\n"
            "📐 *4 Simple Steps:*\n"
            "1\\. Send any video link or upload a video file\.\n"
            "2\\. Choose a canvas style \\(Black Canvas, Blurred Background, Face Tracking, etc\\.\\) or tap `⚡ Quick Run`\.\n"
            "3\\. Choose how many clips to generate \\(3, 5, 10, 20, 50 or custom\\)\.\n"
            "4\\. Choose clip duration \\(Automatic, 0\\-30s, 15\\-30s, 30s\\-1m, 1\\-2m\\)\.\n"
            "💡 *Tip:* Use `↩️ Back` or `❌ Cancel` at any step to change your selections or start over\.\n\n"
            "───\n\n"
            "🌐 *Supported Platforms:*\n"
            "• ▶️ YouTube \\& YouTube Shorts\n"
            "• 🟣 Twitch \\(VODs \\& Clips\\)\n"
            "• 🟩 Kick \\(VODs \\& Clips\\)\n"
            "• 📁 Direct video file upload\n\n"
            "───\n\n"
            "🎬 *What Each Clip Includes:*\n"
            "• Viral moment cut to 9:16 vertical\n"
            "• Bold punch\-word title caption at top\n"
            "• Word\-by\-word subtitles with sentiment colors \\(🔴red/🟢green/🟡yellow\\)\n"
            "• Your watermark logo\n"
            "• YouTube title \\& hashtags ready to copy\n\n"
            "───\n\n"
            "🛠️ *User Commands:*\n"
            "• `/help` — Show this guide\n"
            "• `/update` — Check your real\-time video status\n"
            "• `/queue` — View waiting queue\n"
            "• `/stop` — Cancel active processing\n"
            "• `/cancel` — Reset wizard options\n"
            "• `/clear` — Clear pending queue\n\n"
            "───\n\n"
            "🔐 *Admin\\-Only Commands:*\n"
            "• `/users` — View all approved users \\& live status \\(🟢Active/⚪Idle\\)\n"
            "• `/revoke <user\\_id>` — Remove a user's access\n"
            "• `/brief <rules>` — Set campaign rules for all clips\n\n"
            "───\n\n"
            "🔔 *Access Requests:*\n"
            "When a new user opens the bot, you will receive a private alert card with "
            "`✅ Approve` and `❌ Reject` buttons\\. Only you can approve or deny access\."
        )
    else:
        # ── Regular Approved User Guide ────────────────────────────────────────
        msg = (
            "🎬 *VIRAL Clip Bot — Your Quick Guide*\n\n"
            "Welcome\! Send any video link and this bot will automatically extract the best "
            "viral moments as ready\\-to\\-post vertical clips\.\n\n"
            "───\n\n"
            "📐 *4 Simple Steps:*\n"
            "1\\. Send a video link \\(YouTube, Twitch, Kick, or upload a file\\)\.\n"
            "2\\. Choose your clip style \\(Black Canvas, Blurred Background, Face Tracking, etc\.\\)\.\n"
            "3\\. Choose how many clips you want \\(3, 5, 10, 20, or type any number\\)\.\n"
            "4\\. Choose clip duration \\(Automatic, 0\\-30s, 15\\-30s, 30s\\-1m, 1\\-2m\\)\.\n"
            "💡 *Tip:* Use `↩️ Back` or `❌ Cancel` at any step to change your options\.\n\n"
            "───\n\n"
            "🌐 *Supported Platforms:*\n"
            "• ▶️ YouTube \\& YouTube Shorts\n"
            "• 🟣 Twitch \\(VODs \\& Clips\\)\n"
            "• 🟩 Kick \\(VODs \\& Clips\\)\n"
            "• 📁 Direct video file upload\n\n"
            "───\n\n"
            "📦 *What You'll Receive:*\n"
            "• Top viral moments cut into 9:16 vertical clips\.\n"
            "• Bold punch\-word title caption on each clip\.\n"
            "• 🔴🟢🟡 Word\-by\-word subtitles synced to the speaker's voice\.\n"
            "• A ready\-to\-copy YouTube title \\& hashtags under each video\.\n"
            "• A full titles summary card at the end for easy copy\-pasting\.\n"
            "• A ZIP file with all clips and titles in one download\.\n\n"
            "───\n\n"
            "🛠️ *Commands:*\n"
            "• `/help` — Show this guide\n"
            "• `/update` — Check your processing status\n"
            "• `/queue` — See how many videos are waiting\n"
            "• `/stop` — Cancel your active video\n"
            "• `/cancel` — Reset wizard options\n"
            "• `/clear` — Clear your waiting queue\n\n"
            "───\n\n"
            "💡 *Tips:*\n"
            "• Processing takes a few minutes — you'll be notified when clips are ready\.\n"
            "• You can queue multiple links and they'll process one by one automatically\.\n"
            "• Tap any YouTube title box to copy it instantly on mobile\."
        )

    await update.message.reply_text(msg, parse_mode="MarkdownV2")




def _get_or_recover_session(chat_id: int, update: Update, query=None) -> dict | None:
    """Get active link session or auto-recover from message text so button clicks never expire."""
    session = _pending_links.get(chat_id)
    if session:
        return session

    # Reconstruct session directly from Telegram message text if missing (e.g. after bot restart)
    if query and query.message and query.message.text:
        text = query.message.text
        match = _URL_RE.search(text)
        if match:
            url = match.group(0)
            layout = "black_canvas"
            if "Face Tracking" in text:
                layout = "face_crop"
            elif "Blurred Background" in text or "Blurred Frame" in text:
                layout = "blurred_frame"
            elif "Red Canvas" in text:
                layout = "red_canvas"
            elif "Blue Canvas" in text:
                layout = "blue_canvas"
            elif "Purple Canvas" in text:
                layout = "purple_canvas"

            session = {
                "url": url,
                "update": update,
                "context": None,
                "layout_mode": layout,
                "enable_watermark": False,
            }
            _pending_links[chat_id] = session
            return session
    return None


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles button taps from Telegram Inline Keyboards."""
    query = update.callback_query
    if not query or not query.data or not update.effective_user:
        return

    data = query.data
    chat_id = update.effective_chat.id

    # Handle Master Admin approval button clicks
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
            await query.edit_message_text(
                f"✅ **User Approved!**\n\n`{username}` (`{target_uid}`) has been granted access.",
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
            await query.edit_message_text(f"❌ Access request rejected for User `{target_uid}`.", parse_mode="Markdown")
        return

    if not _is_operator(update.effective_user.id):
        await query.answer("Unauthorized", show_alert=True)
        return

    await query.answer()
    data = query.data
    chat_id = update.effective_chat.id

    # Handle Publishing Action Buttons
    if data.startswith("pub:mobile:"):
        clip_num = data.split("pub:mobile:")[1]
        await query.answer(f"📱 1-Tap Mobile Upload for Clip #{clip_num}!", show_alert=True)
        await query.message.reply_text(
            f"🚀 **1-Tap Mobile Upload Ready for Clip #{clip_num}!**\n\n"
            f"1. Tap the title box above to copy title & hashtags.\n"
            f"2. Long-press the video file above and save to your camera roll.\n"
            f"3. Open YouTube Shorts / TikTok / Reels app and post in 1 tap!",
            parse_mode="Markdown",
        )
        return



    # Handle Wizard Undo / Back / Cancel Navigation
    if data == "wiz:cancel":
        _pending_links.pop(chat_id, None)
        await query.edit_message_text("❌ Setup cancelled. Send any video link whenever you're ready!")
        return

    if data == "wiz:back_layout":
        session = _get_or_recover_session(chat_id, update, query)
        url = session.get("url", "") if session else ""
        title_label = f"🎬 **Video URL:**\n`{url}`" if url else "🎬 **Video link received!**"
        await query.edit_message_text(
            f"{title_label}\n\n"
            f"🎨 **Choose your Video Canvas Background Style:**\n\n"
            f"Select a background color or style below, or tap `⚡ Quick Run` for instant Black Canvas:",
            reply_markup=_make_layout_keyboard(),
            parse_mode="Markdown",
        )
        return

    if data == "wiz:back_wm":
        session = _get_or_recover_session(chat_id, update, query)
        layout_mode = session.get("layout_mode", "black_canvas") if session else "black_canvas"
        mode_label = "👤 Face Tracking" if layout_mode == "face_crop" else "🎬 Blurred Background" if layout_mode == "blurred_frame" else "🖤 Black Canvas"
        await query.edit_message_text(
            f"✅ **Layout selected:** *{mode_label}*\n\n"
            f"🏷️ **Does this video have a watermark logo?**",
            reply_markup=_make_watermark_keyboard(),
            parse_mode="Markdown",
        )
        return

    if data == "wiz:back_sub":
        session = _get_or_recover_session(chat_id, update, query)
        layout_mode = session.get("layout_mode", "black_canvas") if session else "black_canvas"
        mode_label = "👤 Face Tracking" if layout_mode == "face_crop" else "🎬 Blurred Background" if layout_mode == "blurred_frame" else "🖤 Black Canvas"
        await query.edit_message_text(
            f"✅ **Layout selected:** *{mode_label}*\n\n"
            f"💬 **Include Word-by-Word Subtitles on the clips?**",
            reply_markup=_make_subtitles_keyboard(),
            parse_mode="Markdown",
        )
        return

    if data == "wiz:back_clips":
        session = _get_or_recover_session(chat_id, update, query)
        layout_mode = session.get("layout_mode", "black_canvas") if session else "black_canvas"
        mode_label = "👤 Face Tracking" if layout_mode == "face_crop" else "🎬 Blurred Background" if layout_mode == "blurred_frame" else "🖤 Black Canvas"
        sub_label = "💬 Enabled" if session.get("enable_subtitles", True) else "🚫 Disabled"
        await query.edit_message_text(
            f"✅ **Layout:** *{mode_label}* | **Subtitles:** *{sub_label}*\n\n"
            f"✂️ **How many top clips would you like to generate?**\n\n"
            f"Select an option below, or reply with any custom number (e.g., 3, 5, 10, 20):",
            reply_markup=_make_clips_keyboard(),
            parse_mode="Markdown",
        )
        return

    # Step 1: Layout Selected -> Prompt for Watermark (Step 2)
    if data == "fmt:quick_run":
        session = _get_or_recover_session(chat_id, update, query)
        if not session:
            await query.edit_message_text("⚠️ Session expired. Please re-send your video URL.")
            return
        session["layout_mode"] = "black_canvas"
        session["enable_watermark"] = False
        session["enable_subtitles"] = True
        session["wm_name"] = "None"
        await _launch_job(chat_id, 10, edit_message=query.message)
        return

    if data.startswith("fmt:"):
        layout_mode = data.split("fmt:")[1]
        session = _get_or_recover_session(chat_id, update, query)
        if not session:
            await query.edit_message_text("⚠️ Session expired. Please re-send your video URL.")
            return

        session["layout_mode"] = layout_mode
        mode_label = "👤 Face Tracking" if layout_mode == "face_crop" else "🎬 Blurred Background" if layout_mode == "blurred_frame" else "🖤 Black Canvas"

        await query.edit_message_text(
            f"✅ **Layout selected:** *{mode_label}*\n\n"
            f"🏷️ **Does this video have a watermark logo?**",
            reply_markup=_make_watermark_keyboard(),
            parse_mode="Markdown",
        )
        return

    # Step 2: Watermark Option Tapped
    if data.startswith("wm:"):
        choice = data.split("wm:")[1]
        session = _get_or_recover_session(chat_id, update, query)
        if not session:
            await query.edit_message_text("⚠️ Session expired. Please re-send your video URL.")
            return

        current_layout = session.get("layout_mode", "black_canvas")
        mode_label = "👤 Face Tracking" if current_layout == "face_crop" else "🎬 Blurred Background" if current_layout == "blurred_frame" else "🖤 Black Canvas"

        if choice == "yes":
            session["awaiting_wm"] = True
            await query.edit_message_text(
                f"✅ **Layout selected:** *{mode_label}*\n\n"
                f"📸 **Send your PNG Watermark Logo Image now!**\n\n"
                f"Attach or drag & drop your transparent `.png` logo image in Telegram chat.",
                parse_mode="Markdown",
            )
            return
        else:
            session["enable_watermark"] = False
            session["wm_name"] = "None"
            await query.edit_message_text(
                f"✅ **Layout selected:** *{mode_label}*\n\n"
                f"💬 **Include Word-by-Word Subtitles on the clips?**",
                reply_markup=_make_subtitles_keyboard(),
                parse_mode="Markdown",
            )
            return

    # Step 3: Subtitles Option Tapped -> Prompt for Clip Count
    if data.startswith("sub:"):
        choice = data.split("sub:")[1]
        session = _get_or_recover_session(chat_id, update, query)
        if not session:
            await query.edit_message_text("⚠️ Session expired. Please re-send your video URL.")
            return

        session["enable_subtitles"] = (choice == "yes")
        session["awaiting_clip_count"] = True

        current_layout = session.get("layout_mode", "black_canvas")
        mode_label = "👤 Face Tracking" if current_layout == "face_crop" else "🎬 Blurred Background" if current_layout == "blurred_frame" else "🖤 Black Canvas"
        sub_label = "💬 Enabled" if session["enable_subtitles"] else "🚫 Disabled"

        await query.edit_message_text(
            f"✅ **Layout:** *{mode_label}* | **Subtitles:** *{sub_label}*\n\n"
            f"✂️ **How many top clips would you like to generate?**\n\n"
            f"Select an option below, or reply with any custom number (e.g., 3, 5, 10, 20):",
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

        session = _get_or_recover_session(chat_id, update, query)
        if not session:
            await query.edit_message_text("⚠️ Session expired. Please re-send your video URL.")
            return

        session["num_clips"] = num_clips
        current_layout = session.get("layout_mode", "black_canvas")
        mode_label = "👤 Face Tracking" if current_layout == "face_crop" else "🎬 Blurred Background" if current_layout == "blurred_frame" else "🖤 Black Canvas"
        sub_label = "💬 Enabled" if session.get("enable_subtitles", True) else "🚫 Disabled"

        await query.edit_message_text(
            f"✅ **Layout:** *{mode_label}* | **Subtitles:** *{sub_label}* | **Clips:** *{num_clips}*\n\n"
            f"⏱️ **How long should each clip be?**\n\n"
            f"Select a target duration below, or tap `⚡ Automatic` for AI-optimized story length:",
            reply_markup=_make_duration_keyboard(),
            parse_mode="Markdown",
        )
        return

    # Step 4: Clip Duration Option Tapped -> Launch Job
    if data.startswith("dur:"):
        target_duration = data.split("dur:")[1]
        session = _get_or_recover_session(chat_id, update, query)
        if not session:
            await query.edit_message_text("⚠️ Session expired. Please re-send your video URL.")
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
        elapsed_sec = int(time.time() - active_run_status.get("start_time", time.time()))
        elapsed_mins = elapsed_sec // 60
        rem_sec = elapsed_sec % 60
        time_str = f"{elapsed_mins}m {rem_sec}s" if elapsed_mins > 0 else f"{elapsed_sec}s"

        info = active_run_status.get("streamer_info", {})
        streamer = info.get("streamer", "Streamer")
        title = info.get("title", "N/A")
        platform = info.get("platform", "Video")
        step = active_run_status.get("step", "Processing...")
        layout_mode = active_run_status.get("layout_mode", "pillarbox")

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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

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



