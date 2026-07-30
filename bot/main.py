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
    """Housekeeper job: removes old temporary run folders in tmp/ older than 2 hours."""
    tmp_dir = Path("tmp")
    if not tmp_dir.exists():
        return
    now = time.time()
    cutoff = 2 * 3600
    for item in tmp_dir.iterdir():
        if item.is_dir() and item.name != "telegram_uploads":
            try:
                mtime = item.stat().st_mtime
                if now - mtime > cutoff:
                    shutil.rmtree(item, ignore_errors=True)
                    logging.info("Auto-housekeeper cleaned old temp folder: %s", item.name)
            except Exception as exc:
                logging.warning("Housekeeper cleanup failed for %s: %s", item, exc)


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
        handle_stop,
        handle_update,
        handle_users,
    )

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
    app.add_handler(CommandHandler("start", handle_help))
    app.add_handler(CommandHandler("brief", handle_brief))
    app.add_handler(CommandHandler("update", handle_update))
    app.add_handler(CommandHandler("status", handle_update))
    app.add_handler(CommandHandler("stop", handle_stop))
    app.add_handler(CommandHandler("cancel", handle_cancel))
    app.add_handler(CommandHandler("queue", handle_queue))
    app.add_handler(CommandHandler("clear", handle_clear))
    app.add_handler(CommandHandler("users", handle_users))
    app.add_handler(CommandHandler("revoke", handle_revoke))

    # Register button callback handler for layout selection
    app.add_handler(CallbackQueryHandler(handle_callback_query))

    # Register catch-all message handler for all text, photos, documents, stickers, and attachments
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, handle_message))

    logger.info("Polling started — waiting for messages")
    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
