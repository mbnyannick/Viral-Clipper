"""
Direct YouTube Shorts Auto-Publishing via YouTube Data API v3.

Uploads rendered MP4 vertical videos directly to the user's YouTube Channel
with pre-filled title, description, category, and viral hashtags.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import google.oauth2.credentials
import googleapiclient.discovery
import googleapiclient.http

logger = logging.getLogger(__name__)

YOUTUBE_TOKENS_FILE = Path("youtube_tokens.json")
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _load_all_tokens() -> dict[str, dict]:
    if YOUTUBE_TOKENS_FILE.exists():
        try:
            with open(YOUTUBE_TOKENS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_all_tokens(tokens_db: dict[str, dict]) -> None:
    try:
        with open(YOUTUBE_TOKENS_FILE, "w", encoding="utf-8") as f:
            json.dump(tokens_db, f, indent=2)
    except Exception as exc:
        logger.error("Failed to save youtube_tokens.json: %s", exc)


def get_user_youtube_token(chat_id: int) -> str | None:
    """Retrieve saved refresh token for chat_id."""
    tokens_db = _load_all_tokens()
    user_data = tokens_db.get(str(chat_id), {})
    return user_data.get("refresh_token")


def save_user_youtube_token(chat_id: int, refresh_token: str, username: str = "") -> None:
    """Save refresh token for chat_id."""
    tokens_db = _load_all_tokens()
    tokens_db[str(chat_id)] = {
        "refresh_token": refresh_token,
        "username": username,
    }
    _save_all_tokens(tokens_db)


def upload_to_youtube_shorts(
    video_path: Path,
    title: str,
    description: str,
    tags: list[str] | None = None,
    refresh_token: str | None = None,
    privacy_status: str = "public",
) -> str:
    """
    Upload *video_path* directly to YouTube Shorts via YouTube Data API v3.
    Returns the public YouTube Shorts URL (e.g., https://youtube.com/shorts/VIDEO_ID).
    """
    client_id = os.environ.get("YOUTUBE_CLIENT_ID", "")
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        raise RuntimeError("YOUTUBE_CLIENT_ID or YOUTUBE_CLIENT_SECRET environment variable is missing.")

    if not refresh_token:
        raise RuntimeError("No YouTube refresh token provided for this user.")

    if not video_path.exists():
        raise RuntimeError(f"Video file not found at {video_path}")

    # Build OAuth credentials from refresh_token
    credentials = google.oauth2.credentials.Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=YOUTUBE_SCOPES,
    )

    youtube = googleapiclient.discovery.build("youtube", "v3", credentials=credentials)

    clean_tags = tags or ["Shorts", "Viral", "YouTube", "Trending"]

    # Ensure title contains #Shorts hashtag so YouTube categorizes it as a Short
    clean_title = title.strip()
    if "#Shorts" not in clean_title and "#shorts" not in clean_title:
        clean_title = f"{clean_title} #Shorts"
    if len(clean_title) > 100:
        clean_title = clean_title[:97] + "..."

    body = {
        "snippet": {
            "title": clean_title,
            "description": description or f"{clean_title}\n\n#Shorts #Viral #YouTube #Trending",
            "tags": clean_tags,
            "categoryId": "24",  # Entertainment
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }

    logger.info("Uploading Short to YouTube: '%s' (file: %s)", clean_title, video_path.name)

    media = googleapiclient.http.MediaFileUpload(
        str(video_path),
        chunksize=-1,
        resumable=True,
        mimetype="video/mp4",
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            logger.info("  YouTube Upload Progress: %d%%", int(status.progress() * 100))

    video_id = response.get("id")
    if not video_id:
        raise RuntimeError("YouTube API response did not return a video ID.")

    shorts_url = f"https://youtube.com/shorts/{video_id}"
    logger.info("  YouTube Short successfully uploaded! URL: %s", shorts_url)
    return shorts_url
