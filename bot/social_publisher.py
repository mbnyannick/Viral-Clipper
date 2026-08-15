"""
Unified Multi-Platform Direct Social Publisher for VIRAL.

Publishes vertical clips directly to:
  1. TikTok (via Zernio API)
  2. YouTube Shorts (via Zernio API / direct OAuth)
  3. Instagram Reels (via WoopSocial API)
  4. Facebook Reels (via WoopSocial API)
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)

ZERNIO_TIKTOK_KEY = os.environ.get("ZERNIO_TIKTOK_KEY", "sk_02b8f460ab56bace734931e41ab0c7ae01a737dbf20121dd870cc899f42d586d")
ZERNIO_YT_KEY = os.environ.get("ZERNIO_YT_KEY", "sk_0a3c312652a472d2595d46e5ea7a528ac089fc77293a2dbee66324f51deec245")
WOOP_IG_KEY = os.environ.get("WOOP_IG_KEY", "wsk_70ad62bb887c1784.acb8bf93dff7420bf33430b5e0ed5057ceea80d43074526a854cf96b99528804")
WOOP_FB_KEY = os.environ.get("WOOP_FB_KEY", "wsk_6fc4363d65332255.bbbf5403a2c42f58c0e2f5cebdf77e04d9e90a0b88dd19d8639299a90c6f1d39")

ZERNIO_TIKTOK_ACCOUNT = os.environ.get("ZERNIO_TIKTOK_ACCOUNT", "6a7126f0eb10586dadc92b3a")
ZERNIO_YT_ACCOUNT = os.environ.get("ZERNIO_YT_ACCOUNT", "6a7126d9eb10586dadc9277a")
WOOP_IG_PROJECT = os.environ.get("WOOP_IG_PROJECT", "157994660491427840")
WOOP_IG_ACCOUNT = os.environ.get("WOOP_IG_ACCOUNT", "157994846563336192")
WOOP_FB_PROJECT = os.environ.get("WOOP_FB_PROJECT", "157993243961720832")
WOOP_FB_ACCOUNT = os.environ.get("WOOP_FB_ACCOUNT", "158008660860076032")


def _get_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def publish_to_zernio(platform: str, title: str, content: str, video_url: str) -> tuple[bool, str]:
    """Publish to TikTok or YouTube via Zernio API."""
    is_tiktok = (platform.lower() == "tiktok")
    account_id = ZERNIO_TIKTOK_ACCOUNT if is_tiktok else ZERNIO_YT_ACCOUNT
    api_key = ZERNIO_TIKTOK_KEY if is_tiktok else ZERNIO_YT_KEY

    payload = {
        "platforms": [{"platform": "tiktok" if is_tiktok else "youtube", "accountId": account_id}],
        "title": (title[:92] + "...") if len(title) > 95 else title,
        "content": content,
        "mediaItems": [{"type": "video", "url": video_url}],
        "publishNow": True
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.zernio.com/v1/posts",
        data=data_bytes,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ViralBot/1.0"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0, context=_get_ssl_context()) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            logger.info("Zernio %s success (code %d): %s", platform, resp.status, body[:200])
            return True, body
    except urllib.error.HTTPError as he:
        err_body = he.read().decode("utf-8", errors="replace")
        logger.warning("Zernio %s failed (code %d): %s", platform, he.code, err_body[:200])
        return False, err_body
    except Exception as exc:
        logger.warning("Zernio %s exception: %s", platform, exc)
        return False, str(exc)


def _auto_cleanup_woopsocial_media(api_key: str, project_id: str, keep_latest: int = 5) -> None:
    """
    Automatically clean up older media library items in WoopSocial
    to prevent the account from ever reaching its 1 GB storage limit.
    """
    ctx = _get_ssl_context()
    try:
        url = f"https://api.woopsocial.com/v1/media?projectId={project_id}"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {api_key}", "User-Agent": "ViralBot/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15.0, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            items = data.get("media", [])
            if len(items) > keep_latest:
                for item in items[keep_latest:]:
                    mid = item.get("id")
                    if mid:
                        del_req = urllib.request.Request(
                            f"https://api.woopsocial.com/v1/media/{mid}",
                            headers={"Authorization": f"Bearer {api_key}", "User-Agent": "ViralBot/1.0"},
                            method="DELETE",
                        )
                        try:
                            with urllib.request.urlopen(del_req, timeout=10.0, context=ctx):
                                logger.info("Auto-cleaned old WoopSocial media item %s", mid)
                        except Exception as de:
                            logger.debug("Failed deleting old media %s: %s", mid, de)
    except Exception as exc:
        logger.debug("WoopSocial media cleanup skipped: %s", exc)


def publish_to_woopsocial(platform: str, title: str, content: str, video_url: str) -> tuple[bool, str]:
    """Publish to Instagram Reels or Facebook Reels via WoopSocial API."""
    is_ig = ("instagram" in platform.lower() or "ig" in platform.lower())
    api_key = WOOP_IG_KEY if is_ig else WOOP_FB_KEY
    project_id = WOOP_IG_PROJECT if is_ig else WOOP_FB_PROJECT
    social_account_id = WOOP_IG_ACCOUNT if is_ig else WOOP_FB_ACCOUNT
    plat_str = "INSTAGRAM" if is_ig else "FACEBOOK"

    ctx = _get_ssl_context()
    try:
        # 1. Download video binary
        logger.info("Downloading video for WoopSocial %s upload: %s", plat_str, video_url)
        req_v = urllib.request.Request(video_url, headers={"User-Agent": "ViralBot/1.0"})
        with urllib.request.urlopen(req_v, timeout=45.0, context=ctx) as v_resp:
            video_bytes = v_resp.read()

        # 2. Upload media (with auto-retry & auto-purge if storage limit is hit)
        boundary = "----ViralBoundary839218319"
        body_parts = [
            f"--{boundary}\r\n".encode("utf-8"),
            b'Content-Disposition: form-data; name="file"; filename="clip.mp4"\r\n',
            b"Content-Type: video/mp4\r\n\r\n",
            video_bytes,
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ]
        form_data = b"".join(body_parts)

        def _do_upload() -> dict:
            req_upload = urllib.request.Request(
                f"https://api.woopsocial.com/v1/media?projectId={project_id}",
                data=form_data,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "User-Agent": "ViralBot/1.0"
                },
                method="POST"
            )
            with urllib.request.urlopen(req_upload, timeout=90.0, context=ctx) as u_resp:
                return json.loads(u_resp.read().decode("utf-8", errors="replace"))

        try:
            u_data = _do_upload()
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="replace")
            if "storage limit exceeded" in err_body.lower() or he.code == 422:
                logger.warning("WoopSocial storage limit hit. Purging old media library items and retrying...")
                _auto_cleanup_woopsocial_media(api_key, project_id, keep_latest=0)
                u_data = _do_upload()
            else:
                raise he

        media_id = u_data.get("mediaId") or u_data.get("id")
        if not media_id:
            return False, "Failed to retrieve mediaId from WoopSocial"

        logger.info("Uploaded media to WoopSocial %s: mediaId=%s", plat_str, media_id)

        # 3. Create post
        post_payload = {
            "content": [
                {
                    "text": content,
                    "media": [{"type": "MEDIA_LIBRARY", "mediaId": media_id}]
                }
            ],
            "schedule": {"type": "PUBLISH_NOW"},
            "socialAccounts": [
                {
                    "platform": plat_str,
                    "socialAccountId": social_account_id,
                    "postType": "REEL"
                }
            ]
        }
        p_bytes = json.dumps(post_payload).encode("utf-8")
        req_post = urllib.request.Request(
            "https://api.woopsocial.com/v1/posts",
            data=p_bytes,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ViralBot/1.0"
            },
            method="POST"
        )
        with urllib.request.urlopen(req_post, timeout=20.0, context=ctx) as p_resp:
            body = p_resp.read().decode("utf-8", errors="replace")
            logger.info("WoopSocial %s post success (code %d): %s", plat_str, p_resp.status, body[:200])
            # Keep storage clean after successful post
            _auto_cleanup_woopsocial_media(api_key, project_id, keep_latest=5)
            return True, body
    except urllib.error.HTTPError as he:
        err_body = he.read().decode("utf-8", errors="replace")
        logger.warning("WoopSocial %s post failed (code %d): %s", plat_str, he.code, err_body[:200])
        return False, err_body
    except Exception as exc:
        logger.warning("WoopSocial %s post exception: %s", plat_str, exc)
        return False, str(exc)


def _clean_and_deduplicate_content(caption: str, hashtags: str = "") -> str:
    """
    Clean, structure, and strictly deduplicate caption content and hashtags.
    Guarantees:
    - Never duplicates hashtags.
    - Preserves story body lines.
    - Places exactly ONE clean, deduplicated hashtag block at the bottom.
    """
    combined = f"{caption}\n\n{hashtags}".strip() if hashtags else str(caption or "").strip()
    lines = combined.splitlines()
    body_lines: list[str] = []
    seen_tags: set[str] = set()
    unique_tags: list[str] = []

    for line in lines:
        words = line.split()
        if not words:
            if body_lines and body_lines[-1] != "":
                body_lines.append("")
            continue

        # Check if entire line is hashtags
        is_tag_only = all(w.startswith("#") for w in words)
        if is_tag_only:
            for w in words:
                clean_tag = w.strip()
                tag_lower = clean_tag.lower()
                if tag_lower not in seen_tags:
                    seen_tags.add(tag_lower)
                    unique_tags.append(clean_tag)
        else:
            line_body_words = []
            for w in words:
                if w.startswith("#"):
                    clean_tag = w.strip()
                    tag_lower = clean_tag.lower()
                    if tag_lower not in seen_tags:
                        seen_tags.add(tag_lower)
                        unique_tags.append(clean_tag)
                else:
                    line_body_words.append(w)
            if line_body_words:
                body_lines.append(" ".join(line_body_words))

    # Strip empty trailing lines from body
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    clean_body = "\n".join(body_lines).strip()
    tag_str = " ".join(unique_tags).strip()

    if clean_body and tag_str:
        return f"{clean_body}\n\n{tag_str}"
    elif clean_body:
        return clean_body
    return tag_str


def direct_publish_clip(platform: str, title: str, caption: str, hashtags: str, video_url: str) -> dict[str, tuple[bool, str]]:
    """
    Directly publishes to the requested platform or all 4 platforms simultaneously.
    Returns a dict of {platform_name: (success_bool, message)}.
    """
    targets = ["tiktok", "youtube", "instagram", "facebook"] if platform.lower() == "all" else [platform.lower()]
    full_content = _clean_and_deduplicate_content(caption, hashtags)

    results = {}
    for p in targets:
        if p in ("tiktok", "youtube"):
            ok, msg = publish_to_zernio(p, title, full_content, video_url)
            results[p] = (ok, msg)
        elif p in ("instagram", "facebook", "ig", "fb"):
            ok, msg = publish_to_woopsocial(p, title, full_content, video_url)
            results[p] = (ok, msg)
        else:
            results[p] = (False, f"Unknown platform: {p}")

    return results
