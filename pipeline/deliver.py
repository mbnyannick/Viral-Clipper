"""
Step 8 — Delivery.

Sends each finished clip as an individual Telegram video message with a
numbered caption (e.g. "Clip 1/10") so the operator always knows exactly
which clip they're looking at and can tick them off one by one when posting.
"""

import html
import logging
import re
import zipfile
from pathlib import Path

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

from .errors import PipelineError

logger = logging.getLogger(__name__)


async def deliver_clips(
    final_clips: list[Path],
    bot: Bot,
    chat_id: int | str,
    moments: list | None = None,
    clip_captions: list[str] | None = None,
) -> None:
    """
    Send each clip in *final_clips* as a separate numbered video message with
    1-tap HTML copyable unique SEO title, description, hashtags, and Action Buttons.
    """
    if not final_clips:
        await bot.send_message(chat_id=chat_id, text="⚠️ No clips were produced.")
        return

    total = len(final_clips)
    logger.info("Delivering %d clips individually with 1-tap copy captions & action buttons", total)

    for i, path in enumerate(final_clips, start=1):
        m = moments[i - 1] if moments and i - 1 < len(moments) else None
        custom_cap = clip_captions[i - 1] if clip_captions and i - 1 < len(clip_captions) else None
        
        if m and hasattr(m, "score"):
            score = m.score
            if score >= 90:
                tier = "S-Tier"
            elif score >= 80:
                tier = "A-Tier"
            elif score >= 70:
                tier = "B-Tier"
            else:
                tier = "C-Tier"
            reasoning = m.reasoning
        else:
            score = 98
            tier = "S-Tier"
            reasoning = "High viral potential."

        if custom_cap:
            clean_text = html.escape(custom_cap.strip())
            caption = (
                f"📹 <b>Clip {i:02d}/{total:02d}</b> — ⚡ <i>Hook Score: {score}/100 ({tier})</i>\n\n"
                f"💡 <i>{html.escape(reasoning)}</i>\n\n"
                f"<code>{clean_text}</code>"
            )
        else:
            if m:
                raw_title = getattr(m, "title", None)
                if not raw_title and hasattr(m, "caption_lines"):
                    raw_title = " ".join(m.caption_lines)
                elif not raw_title:
                    raw_title = "Viral Clip"

                clean_title = html.escape(raw_title)
                emoji = getattr(m, "emoji", "🤯") or "🤯"
                copy_payload = f"{clean_title} {emoji}\n\n#Shorts #Viral #YouTube #Trending"
                caption = (
                    f"📹 <b>Clip {i:02d}/{total:02d}</b> — ⚡ <i>Hook Score: {score}/100 ({tier})</i>\n\n"
                    f"💡 <i>{html.escape(reasoning)}</i>\n\n"
                    f"<code>{copy_payload}</code>"
                )
            else:
                caption = f"📹 <b>Clip {i:02d}/{total:02d}</b>"

        action_keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📱 1-Tap Mobile Upload", callback_data=f"pub:mobile:{i}"),
            ]
        ])

        try:
            with open(path, "rb") as fh:
                await bot.send_video(
                    chat_id=chat_id,
                    video=fh,
                    caption=caption,
                    parse_mode="HTML",
                    reply_markup=action_keyboard,
                    supports_streaming=True,
                )
            logger.info("  Sent Clip %d/%d with Action Buttons", i, total)
        except Exception as exc:
            try:
                # Fallback video send without markup
                with open(path, "rb") as fh:
                    await bot.send_video(
                        chat_id=chat_id,
                        video=fh,
                        caption=f"📹 Clip {i:02d}/{total:02d}",
                        supports_streaming=True,
                    )
            except Exception:
                raise PipelineError("deliver", f"Clip {i}/{total}: {exc}") from exc

    logger.info("All %d clips delivered", total)

    # Deliver ZIP archive bundle for 1-tap download of all clips + captions
    await deliver_zip_archive(final_clips, bot, chat_id, clip_captions=clip_captions)


async def deliver_zip_archive(
    final_clips: list[Path],
    bot: Bot,
    chat_id: int | str,
    streamer_name: str = "Streamer",
    clip_captions: list[str] | None = None,
) -> None:
    """
    Zip all generated clips + captions into 1-tap download archives.
    Automatically splits into <45MB volumes if total size exceeds Telegram's 50MB limit.
    """
    if not final_clips:
        return

    tmp_dir = final_clips[0].parent
    clean_name = re.sub(r"[^\w]", "_", streamer_name).strip("_")
    base_name = clean_name if clean_name else "Streamer"

    # Always deliver a lightweight Text Captions bundle (.txt document)
    if clip_captions:
        try:
            txt_path = tmp_dir / f"{base_name}_Titles_And_Hashtags.txt"
            captions_text = "\n\n".join(
                f"=== CLIP {i:02d} TITLE & HASHTAGS ===\n{cap}"
                for i, cap in enumerate(clip_captions, start=1)
            )
            txt_path.write_text(captions_text, encoding="utf-8")
            with open(txt_path, "rb") as fh:
                await bot.send_document(
                    chat_id=chat_id,
                    document=fh,
                    caption=f"📝 <b>All Clip Titles & Hashtags (.txt)</b>",
                    parse_mode="HTML",
                )
        except Exception as txt_exc:
            logger.warning("Could not deliver captions txt bundle: %s", txt_exc)

    # Partition clips into <= 45MB zip chunks to stay safely under Telegram's 50MB bot upload limit
    MAX_PART_BYTES = 45 * 1024 * 1024  # 45 MB per zip volume

    current_part: list[Path] = []
    current_part_size = 0
    parts: list[list[Path]] = []

    for clip_p in final_clips:
        c_size = clip_p.stat().st_size if clip_p.exists() else 0
        if current_part and (current_part_size + c_size > MAX_PART_BYTES):
            parts.append(current_part)
            current_part = [clip_p]
            current_part_size = c_size
        else:
            current_part.append(clip_p)
            current_part_size += c_size

    if current_part:
        parts.append(current_part)

    num_parts = len(parts)
    for idx, part_clips in enumerate(parts, start=1):
        suffix = f"_Part{idx}" if num_parts > 1 else ""
        zip_name = f"{base_name}_Clips{suffix}.zip"
        zip_path = tmp_dir / zip_name

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for c_file in part_clips:
                    zipf.write(c_file, arcname=c_file.name)

            with open(zip_path, "rb") as fh:
                part_label = f" (Part {idx}/{num_parts})" if num_parts > 1 else ""
                await bot.send_document(
                    chat_id=chat_id,
                    document=fh,
                    caption=f"📦 <b>Clip Download Archive{part_label}</b> ({len(part_clips)} clips)",
                    parse_mode="HTML",
                )
            logger.info("Sent ZIP volume %s successfully", zip_name)
        except Exception as exc:
            logger.warning("ZIP archive delivery attempt failed for %s: %s", zip_name, exc)
