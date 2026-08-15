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

        # 2. Upload media
        boundary = "----ViralBoundary839218319"
        body_parts = [
            f"--{boundary}\r\n".encode("utf-8"),
            b'Content-Disposition: form-data; name="file"; filename="clip.mp4"\r\n',
            b"Content-Type: video/mp4\r\n\r\n",
            video_bytes,
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ]
        form_data = b"".join(body_parts)

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
            u_data = json.loads(u_resp.read().decode("utf-8", errors="replace"))
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
            return True, body
    except urllib.error.HTTPError as he:
        err_body = he.read().decode("utf-8", errors="replace")
        logger.warning("WoopSocial %s post failed (code %d): %s", plat_str, he.code, err_body[:200])
        return False, err_body
    except Exception as exc:
        logger.warning("WoopSocial %s post exception: %s", plat_str, exc)
        return False, str(exc)


def direct_publish_clip(platform: str, title: str, caption: str, hashtags: str, video_url: str) -> dict[str, tuple[bool, str]]:
    """
    Directly publishes to the requested platform or all 4 platforms simultaneously.
    Returns a dict of {platform_name: (success_bool, message)}.
    """
    targets = ["tiktok", "youtube", "instagram", "facebook"] if platform.lower() == "all" else [platform.lower()]
    full_content = f"{caption}\n\n{hashtags}".strip() if hashtags else caption

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
