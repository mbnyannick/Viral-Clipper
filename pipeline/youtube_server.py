"""
Lightweight 1-Tap OAuth Web Receiver for YouTube Channel Linking.

Listens on HTTP port (e.g. 8080) for Google OAuth redirects (/oauth/callback).
Exchanges authorization code for refresh_token automatically in the background,
saves it to youtube_tokens.json, and notifies the user in Telegram.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from .youtube_upload import YOUTUBE_SCOPES, save_user_youtube_token

logger = logging.getLogger(__name__)

OAUTH_PORT = int(os.environ.get("OAUTH_PORT", "8080"))
OAUTH_REDIRECT_URI = os.environ.get("OAUTH_REDIRECT_URI", f"http://localhost:{OAUTH_PORT}/oauth/callback")

_TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")


def get_google_auth_url(chat_id: int) -> str:
    """Generate 1-tap Google OAuth Authorization URL for chat_id."""
    client_id = os.environ.get("YOUTUBE_CLIENT_ID", "").strip()
    if not client_id:
        raise ValueError("YOUTUBE_CLIENT_ID environment variable is missing")

    params = {
        "client_id": client_id,
        "redirect_uri": "http://127.0.0.1:8080/oauth/callback",
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/youtube.upload",
        "access_type": "offline",
        "prompt": "consent",
        "state": str(chat_id),
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"


def exchange_code_for_tokens(code: str) -> dict:
    """Exchange authorization code for refresh_token & access_token."""
    client_id = os.environ.get("YOUTUBE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET", "").strip()

    token_url = "https://oauth2.googleapis.com/token"
    payload = urllib.parse.urlencode({
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": "http://127.0.0.1:8080/oauth/callback",
        "grant_type": "authorization_code",
    }).encode("utf-8")

    req = urllib.request.Request(
        token_url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def notify_telegram_user(chat_id: int, text: str) -> None:
    """Send Telegram message notification when channel connection succeeds."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass
    except Exception as exc:
        logger.warning("Failed to send Telegram notification to %s: %s", chat_id, exc)


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for Google OAuth redirect callback."""

    def log_message(self, format, *args):
        logger.info("OAuth Web Server: " + format, *args)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/oauth/callback":
            params = urllib.parse.parse_qs(parsed.query)
            code = params.get("code", [""])[0]
            chat_id_str = params.get("state", ["0"])[0]

            if code and chat_id_str != "0":
                try:
                    chat_id = int(chat_id_str)
                    tokens = exchange_code_for_tokens(code)
                    refresh_token = tokens.get("refresh_token") or tokens.get("access_token", "")

                    if refresh_token:
                        save_user_youtube_token(chat_id, refresh_token)
                        notify_telegram_user(
                            chat_id,
                            "🎉 **YouTube Channel Connected Successfully!**\n\n"
                            "You can now auto-publish any clip directly to YouTube Shorts with 1 click! 🚀"
                        )
                        self._respond_html(
                            "<h1>🎉 YouTube Channel Connected!</h1>"
                            "<p>Your YouTube account has been linked successfully. You may close this tab and return to Telegram.</p>"
                        )
                        return
                except Exception as exc:
                    logger.error("OAuth exchange error for chat %s: %s", chat_id_str, exc)

            self._respond_html(
                "<h1>❌ Connection Failed</h1>"
                "<p>Could not complete YouTube authentication. Please try connecting again from Telegram.</p>",
                status=400,
            )
            return

        self._respond_html("<h1>VIRAL Bot OAuth Receiver</h1>", status=200)

    def _respond_html(self, content: str, status: int = 200) -> None:
        html_doc = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>YouTube Channel Connection</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #0f0f12; color: #ffffff; text-align: center; }}
    .card {{ background: #1a1a24; padding: 40px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); max-width: 400px; }}
    h1 {{ color: #00FF66; margin-bottom: 10px; font-size: 24px; }}
    p {{ color: #a0a0b0; font-size: 15px; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="card">
    {content}
  </div>
</body>
</html>"""
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html_doc.encode("utf-8"))


def start_oauth_web_server() -> None:
    """Start background HTTP OAuth receiver on OAUTH_PORT."""
    try:
        server = HTTPServer(("0.0.0.0", OAUTH_PORT), OAuthCallbackHandler)
        logger.info("Started 1-Tap OAuth Web Receiver on port %d (redirect_uri=%s)", OAUTH_PORT, OAUTH_REDIRECT_URI)
        t = Thread(target=server.serve_forever, daemon=True)
        t.start()
    except Exception as exc:
        logger.warning("Could not start OAuth Web Receiver on port %d: %s", OAUTH_PORT, exc)
