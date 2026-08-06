"""
Bot entry point.

Sets up rotating file + stdout logging, loads env vars, builds the
python-telegram-bot Application, registers message, command & callback handlers,
and starts polling.
"""

import logging
import logging.handlers
import os
from pathlib import Path

import shutil
import time

from dotenv import load_dotenv
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

load_dotenv()


def _cleanup_old_tmp_files() -> None:
    """Housekeeper job: removes temp working folders older than 2h and output MP4s older than 48h."""
    now = time.time()
    tmp_dir = Path("tmp")
    if tmp_dir.exists():
        cutoff_tmp = 2 * 3600
        for item in tmp_dir.iterdir():
            if item.is_dir() and item.name != "telegram_uploads":
                try:
                    if now - item.stat().st_mtime > cutoff_tmp:
                        shutil.rmtree(item, ignore_errors=True)
                        logging.info("Auto-housekeeper cleaned old temp folder: %s", item.name)
                except Exception as exc:
                    logging.warning("Housekeeper cleanup failed for %s: %s", item, exc)

    output_dir = Path("output")
    if output_dir.exists():
        cutoff_output = 48 * 3600  # Delete rendered MP4 clips older than 48 hours
        for vid in output_dir.glob("*.mp4"):
            try:
                if now - vid.stat().st_mtime > cutoff_output:
                    vid.unlink(missing_ok=True)
                    logging.info("Auto-housekeeper cleaned published clip: %s", vid.name)
            except Exception as exc:
                logging.warning("Output clip cleanup failed for %s: %s", vid.name, exc)



def _setup_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "viral.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
    )
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Quiet noisy httpx / telegram library debug output
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)


def main() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)

    _cleanup_old_tmp_files()

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    logger.info("Starting VIRAL bot")

    from bot.handlers import (
        handle_brief,
        handle_callback_query,
        handle_cancel,
        handle_clear,
        handle_help,
        handle_message,
        handle_queue,
        handle_revoke,
        handle_schedule,
        handle_start,
        handle_stop,
        handle_update,
        handle_users,
    )
    from bot.scheduler import run_scheduler_loop

    from telegram.request import HTTPXRequest

    request = HTTPXRequest(
        read_timeout=60.0,
        write_timeout=60.0,
        connect_timeout=30.0,
        pool_timeout=30.0,
    )
    app = Application.builder().token(token).request(request).build()

    # Register interactive commands
    app.add_handler(CommandHandler("help", handle_help))
    app.add_handler(CommandHandler("start", handle_start))

    app.add_handler(CommandHandler("brief", handle_brief))
    app.add_handler(CommandHandler("update", handle_update))
    app.add_handler(CommandHandler("status", handle_update))
    app.add_handler(CommandHandler("stop", handle_stop))
    app.add_handler(CommandHandler("cancel", handle_cancel))
    app.add_handler(CommandHandler("queue", handle_queue))
    app.add_handler(CommandHandler("clear", handle_clear))
    app.add_handler(CommandHandler("users", handle_users))
    app.add_handler(CommandHandler("revoke", handle_revoke))
    app.add_handler(CommandHandler("schedule", handle_schedule))

    async def _post_init(application) -> None:
        import asyncio
        asyncio.create_task(run_scheduler_loop())
        logger.info("Peak-hour scheduler background task started.")

    app.post_init = _post_init

    # Register button callback handler for layout selection
    app.add_handler(CallbackQueryHandler(handle_callback_query))

    # Register catch-all message handler for all text, photos, documents, stickers, and attachments
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))

    logger.info("Polling started — waiting for messages")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
