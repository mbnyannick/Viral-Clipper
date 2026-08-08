"""
Step 1 — Download & Metadata Extraction.

Downloads the source video once using yt-dlp, extracts the audio track via
ffmpeg, and retrieves rich platform, streamer, video title, duration, and live status metadata.

Platform-Specific Engines:
- Kick: --impersonate chrome for Cloudflare TLS bypass + /videos/UUID live fallback.
- YouTube: player_client=android,mweb with nodejs JS runtime & optional cookies.txt support.
- Twitch: native HLS extractor with 8 parallel fragment threads.

Returns (video_path, audio_path, streamer_info).
"""

import asyncio
import json
import logging
import re
import urllib.request
from pathlib import Path

from .errors import PipelineError

logger = logging.getLogger(__name__)


async def _run(cmd: list[str], step: str, timeout: float = 180.0) -> str:
    """Run a subprocess with safety timeout, returning stdout or raising PipelineError on failure/timeout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        raise PipelineError(step, f"Subprocess command timed out after {timeout:.0f}s: {' '.join(cmd[:3])}")

    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-600:]
        raise PipelineError(step, tail)
    return stdout.decode(errors="replace").strip()


def clean_streamer_name(raw_name: str) -> str:
    """
    Clean channel/streamer names by stripping suffixes like Live, VODs, Clips, Shorts, Official.
    Examples:
        'Kai Cenat Live' -> 'Kai'
        'N3on Live' -> 'N3on'
        'Adin Live' -> 'Adin'
        'Speed Live' -> 'Speed'
        'Caleb Hammer Official' -> 'Caleb Hammer'
    """
    if not raw_name:
        return "Streamer"

    # Remove channel suffixes case-insensitively
    suffixes = [
        r"\bLive\b", r"\bVODs?\b", r"\bClips?\b", r"\bShorts?\b", r"\bOfficial\b",
        r"\bGaming\b", r"\bReacts?\b", r"\bTV\b", r"\bPodcast\b", r"\bHighlights?\b",
        r"\bExtra\b", r"\bChannel\b", r"\bShow\b", r"\bDaily\b", r"\bReels?\b"
    ]
    cleaned = raw_name
    for s in suffixes:
        cleaned = re.sub(s, "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Known famous single-name streamers
    words = cleaned.split()
    if words:
        first_word = words[0]
        known_famous = {"kai", "n3on", "neon", "adin", "speed", "ishowspeed", "xqc", "bruce", "fanum", "duke", "agent", "ray", "clix", "tarik", "pokimane", "hassan", "hasan", "sammy"}
        if first_word.lower() in known_famous:
            return first_word.capitalize() if first_word.lower() != "n3on" else "N3on"

    return cleaned if cleaned else raw_name


def _parse_url_fallback(url: str) -> str:
    """Extract candidate channel/streamer name from URL if metadata lookup is missing."""
    m = re.search(r"(?:kick\.com|twitch\.tv)/([^/?#]+)", url, re.IGNORECASE)
    if m:
        name = m.group(1).capitalize()
        if name.lower() not in ("video", "videos", "clip", "clips"):
            return clean_streamer_name(name)
    return "Streamer"


def normalize_kick_url(url: str) -> str:
    """Return original URL so Kick VODs and clips keep their exact endpoint."""
    return url


def _parse_kick_vod_id(url: str) -> tuple[str | None, str | None]:
    """
    Extract channel name and VOD ID from a Kick VOD URL.
    Supports:
      - kick.com/username/videos/12345678           (numeric ID)
      - kick.com/username/videos/SLUG-text          (slug ID)
      - kick.com/username/videos/UUID-uuid          (UUID)
    Returns (channel_name, vod_id) or (None, None).
    """
    m = re.match(
        r"https?://(?:www\.)?kick\.com/([^/?#]+)/videos/([^/?#]+)",
        url, re.IGNORECASE,
    )
    if m:
        return m.group(1), m.group(2)
    return None, None


async def _kick_vod_get_hls_url(url: str) -> tuple[str, str, str, int]:
    """
    Fetch the HLS playlist URL for a Kick VOD via Kick API v2.
    Returns (hls_url, channel, title, duration_seconds).
    Fast — no video download, just an API call.
    """
    channel, vod_id = _parse_kick_vod_id(url)
    if not channel or not vod_id:
        raise PipelineError("download", f"Could not parse Kick VOD URL: {url}")

    # Fetch VOD list from Kick API v2 with Cloudflare bypass via curl_cffi
    api_url = f"https://kick.com/api/v2/channels/{channel}/videos?page=1&limit=50"
    data = None
    try:
        from curl_cffi import requests as cffi_requests
        r = cffi_requests.get(api_url, impersonate="chrome", timeout=15)
        if r.status_code == 200:
            data = r.json()
    except Exception as exc:
        logger.warning("curl_cffi request failed for Kick API: %s — trying urllib fallback", exc)

    if data is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        req = urllib.request.Request(api_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            raise PipelineError("download", f"Kick API request failed: {exc}") from exc

    vods = data if isinstance(data, list) else data.get("data", data.get("videos", []))

    # Match the VOD by ID, numeric ID or slug
    matched = None
    vod_id_lower = str(vod_id).lower()
    for v in vods:
        vid_id = str(v.get("id", "")).lower()
        slug = str(v.get("slug", "")).lower()
        uuid_val = str((v.get("video") or {}).get("uuid", "")).lower()
        if vod_id_lower in (vid_id, slug, uuid_val) or slug.startswith(vod_id_lower) or vod_id_lower in slug:
            matched = v
            break

    if not matched:
        if vods:
            matched = vods[0]
            logger.warning("Kick VOD ID %s not found, using most recent VOD instead.", vod_id)
        else:
            raise PipelineError("download", f"No VODs found for Kick channel '{channel}'.")

    hls_url = matched.get("source") or matched.get("playback_url")
    if not hls_url:
        raise PipelineError("download", f"No HLS source URL found for Kick VOD: {matched.get('id')}")

    title = matched.get("session_title") or matched.get("title") or ""
    duration = matched.get("duration") or 0
    # duration from kick API is in ms
    if isinstance(duration, int) and duration > 10000:
        duration = duration // 1000

    logger.info("Kick VOD HLS resolved: %s — %s (%ss)", channel, title, duration)
    return hls_url, channel, title, int(duration)


async def _kick_vod_direct_download(
    url: str,
    video_path: Path,
    audio_path: Path,
) -> tuple[Path, Path, dict]:
    """
    Download a Kick VOD using Kick's API v2 (via curl_cffi TLS impersonate) + ffmpeg HLS.
    Returns (video_path, audio_path, streamer_info).
    """
    hls_url, channel, title, duration = await _kick_vod_get_hls_url(url)

    # Download via ffmpeg HLS directly
    video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-t", "18000",
        "-i", hls_url,
        "-c", "copy",
        "-bsf:a", "aac_adtstoasc",
        str(video_path),
    ]
    try:
        await _run(cmd, step="download", timeout=21600.0)  # 6h — covers capped 5h VODs + overhead
    except PipelineError as exc:
        raise PipelineError("download", f"ffmpeg HLS download failed: {exc.reason}") from exc

    # Extract audio
    await _run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "copy", str(audio_path)],
        step="audio_extraction",
        timeout=3600.0,
    )

    streamer_info = {
        "streamer": channel.capitalize(),
        "title": title,
        "duration": str(duration),
        "platform": "Kick 🟩",
        "content_type": "VOD / Video 🎥",
        "live_status": "📁 Recorded VOD",
        "is_offline": False,
        "effective_url": url,
    }
    return video_path, audio_path, streamer_info



async def _kick_live_direct_download(
    url: str,
    video_path: Path,
    audio_path: Path,
) -> tuple[Path, Path, dict]:
    """Download a Kick Livestream using Kick API v2 + ffmpeg HLS."""
    m = re.search(r"kick\.com/([^/?#]+)", url, re.IGNORECASE)
    if not m:
        raise PipelineError("download", f"Could not parse Kick URL: {url}")
    channel = m.group(1)

    api_url = f"https://kick.com/api/v2/channels/{channel}"
    data = None
    try:
        from curl_cffi import requests as cffi_requests
        r = cffi_requests.get(api_url, impersonate="chrome", timeout=15)
        if r.status_code == 200:
            data = r.json()
    except Exception as exc:
        logger.warning("curl_cffi request failed for Kick API: %s", exc)

    if not data:
        raise PipelineError("download", f"Kick API request failed or channel not found: {channel}")

    playback_url = data.get("playback_url")
    if not playback_url:
        raise PipelineError("download", f"No live HLS playback URL found for Kick channel '{channel}'. Is the streamer offline?")

    livestream = data.get("livestream") or {}
    title = livestream.get("session_title", "")
    
    logger.info("Kick Live direct HLS: %s — %s", channel, title)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-t", "3600",
        "-i", playback_url,
        "-c", "copy",
        "-bsf:a", "aac_adtstoasc",
        str(video_path),
    ]
    try:
        await _run(cmd, step="download", timeout=5400.0)  # 90min — covers capped 1h live buffer + overhead
    except PipelineError as exc:
        raise PipelineError("download", f"ffmpeg HLS download failed: {exc.reason}") from exc

    await _run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "copy", str(audio_path)],
        step="audio_extraction",
        timeout=3600.0,
    )

    platform, content_type, live_status = detect_platform_and_type(url)
    streamer_info = {
        "streamer": channel.capitalize(),
        "title": title,
        "duration": "0",
        "platform": "Kick 🟩",
        "content_type": "Livestream 🔴",
        "live_status": "🔴 LIVE NOW",
        "is_offline": False,
        "effective_url": url,
    }
    return video_path, audio_path, streamer_info


def _get_cookie_opts() -> list[str]:
    """
    Return yt-dlp cookie flags. Checks for Firefox profile in container/host,
    or falls back to cookies.txt if present.
    """
    firefox_paths = [
        Path("/root/.mozilla/firefox"),
        Path("/home/opc/.mozilla/firefox"),
        Path.home() / ".mozilla" / "firefox",
    ]
    for ff_path in firefox_paths:
        if ff_path.exists():
            profiles = list(ff_path.glob("*.default*")) + list(ff_path.glob("*.default"))
            if profiles:
                return ["--cookies-from-browser", f"firefox:{profiles[0]}"]
            return ["--cookies-from-browser", f"firefox:{ff_path}"]

    for cookie_path in [Path("cookies.txt"), Path("assets/cookies.txt"), Path("/app/cookies.txt")]:
        if cookie_path.exists():
            return ["--cookies", str(cookie_path)]
    return []


def detect_platform_and_type(url: str, is_live_flag: str = "") -> tuple[str, str, str]:
    """
    Detect Platform (YouTube/Kick/Twitch), Content Type, and Live Status.
    Returns (platform, content_type, live_status).
    """
    u = url.lower()
    is_live = is_live_flag.lower() in ("true", "1", "live", "was_live")

    if "kick.com" in u:
        platform = "Kick 🟩"
        if "clip" in u:
            content_type = "Clip ✂️"
            live_status = "📁 Recorded Clip"
        elif "videos" in u and not is_live:
            content_type = "VOD / Video 🎥"
            live_status = "📁 Recorded VOD"
        else:
            content_type = "Livestream 🔴"
            live_status = "🔴 LIVE NOW"
    elif "twitch.tv" in u:
        platform = "Twitch 🟣"
        if "clip" in u:
            content_type = "Clip ✂️"
            live_status = "📁 Recorded Clip"
        elif "videos" in u or "/v/" in u:
            content_type = "VOD / Video 🎥"
            live_status = "📁 Recorded VOD"
        else:
            content_type = "Livestream 🔴"
            live_status = "🔴 LIVE NOW"
    elif "youtube.com" in u or "youtu.be" in u:
        platform = "YouTube 🔴"
        if "shorts" in u:
            content_type = "Shorts 📱"
            live_status = "📁 Shorts Video"
        elif is_live:
            content_type = "Livestream 🔴"
            live_status = "🔴 LIVE NOW"
        else:
            content_type = "Video 🎥"
            live_status = "📁 Recorded Video"
    else:
        platform = "Web Stream 🌐"
        content_type = "Video 🎥"
        live_status = "📁 Recorded Video"

    return platform, content_type, live_status


YT_CLIENT_CHAINS = [
    [],  # Default web client (required for cookies.txt authentication)
    ["--extractor-args", "youtube:player_client=web"],
    ["--extractor-args", "youtube:player_client=tvhtml5,web"],
]


async def extract_metadata(url: str) -> dict[str, str]:
    """
    Extract platform, content_type, streamer, video title, duration, and live_status via yt-dlp.
    Supports local desktop video files directly via ffprobe.
    """
    local_p = Path(url)
    if local_p.exists() and local_p.is_file():
        try:
            raw_dur = await _run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprintwrappers=1:nokey=1", str(local_p)],
                step="metadata",
            )
            duration_str = str(int(float(raw_dur.strip()))) if raw_dur.strip() else "0"
        except Exception:
            duration_str = "0"

        return {
            "streamer": "Desktop Upload",
            "title": local_p.stem.replace("_", " ").replace("-", " ").title(),
            "duration": duration_str,
            "platform": "Local File 📁",
            "content_type": "Video File 🎥",
            "live_status": "📁 Local File",
            "is_offline": False,
            "effective_url": str(local_p.resolve()),
        }

    urls_to_try = [url]
    kick_fallback = normalize_kick_url(url)
    if kick_fallback != url:
        urls_to_try.append(kick_fallback)

    u_lower = url.lower()
    is_youtube = "youtu" in u_lower
    is_kick = "kick.com" in u_lower

    yt_client_chains = YT_CLIENT_CHAINS if is_youtube else [[]]

    impersonate_opts = ["--impersonate", "chrome"] if is_kick else []
    cookie_opts = _get_cookie_opts()

    last_exc = None
    for candidate_url in urls_to_try:
        for yt_opts in yt_client_chains:
            try:
                raw = await _run(
                    [
                        "yt-dlp",
                        "--no-check-certificates",
                        "--no-playlist",
                        "--remote-components", "ejs:github",
                        *cookie_opts,
                        *impersonate_opts,
                        *yt_opts,
                        "--print", "%(uploader)s|%(channel)s|%(title)s|%(duration)s|%(is_live)s",
                        candidate_url,
                    ],
                    step="metadata",
                )
                parts = raw.split("|")
                uploader = parts[0].strip() if len(parts) > 0 else ""
                channel = parts[1].strip() if len(parts) > 1 else ""
                title = parts[2].strip() if len(parts) > 2 else ""
                duration_str = parts[3].strip() if len(parts) > 3 else "0"
                is_live_flag = parts[4].strip() if len(parts) > 4 else ""

                streamer_raw = uploader or channel or _parse_url_fallback(candidate_url)
                streamer = clean_streamer_name(streamer_raw)
                if not streamer or streamer.lower() in ("na", "none", "unknown"):
                    streamer = clean_streamer_name(_parse_url_fallback(candidate_url))

                platform, content_type, live_status = detect_platform_and_type(candidate_url, is_live_flag=is_live_flag)

                logger.info(
                    "Extracted metadata — Platform: '%s', Live Status: '%s', Streamer: '%s', Title: '%s', Duration: %ss",
                    platform, live_status, streamer, title, duration_str,
                )
                return {
                    "streamer": streamer,
                    "title": title,
                    "duration": duration_str,
                    "platform": platform,
                    "content_type": content_type,
                    "live_status": live_status,
                    "is_offline": False,
                    "effective_url": candidate_url,
                }
            except Exception as exc:
                last_exc = exc
                logger.warning("Metadata extraction attempt failed for %s: %s", candidate_url, exc)

    platform, content_type, live_status = detect_platform_and_type(url)
    err_msg = str(last_exc).lower() if last_exc else ""
    # Never mark Kick VOD/clip URLs as offline — they use a dedicated downloader
    # that bypasses yt-dlp entirely, so yt-dlp errors are expected and irrelevant.
    u_lower_check = url.lower()
    is_kick_vod_or_clip = "kick.com" in u_lower_check and ("/videos/" in u_lower_check or "clips" in u_lower_check)
    is_offline = (
        not is_kick_vod_or_clip
        and ("offline" in err_msg or "not currently live" in err_msg or "404: not found" in err_msg)
    )
    if is_offline:
        live_status = "⚪ Streamer is OFFLINE right now"

    return {
        "streamer": _parse_url_fallback(url),
        "title": "",
        "duration": "0",
        "platform": platform,
        "content_type": content_type,
        "live_status": live_status,
        "is_offline": is_offline,
        "effective_url": url,
    }


async def download(url: str, output_dir: Path, streamer_info: dict | None = None) -> tuple[Path, Path, dict[str, str]]:
    """
    Download *url* via yt-dlp into *output_dir*.
    Uses 8 parallel fragment threads for ultra-fast downloads and caps active live streams to 2-hour buffer max.

    Returns
    -------
    (video_path, audio_path, streamer_info)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "source.mp4"
    audio_path = output_dir / "source.m4a"

    if streamer_info is None:
        streamer_info = await extract_metadata(url)

    download_url = streamer_info.get("effective_url", url)
    local_p = Path(download_url)
    if local_p.exists() and local_p.is_file():
        logger.info("Using local video file directly: %s", local_p)
        video_path = local_p
        logger.info("Extracting audio track from local file %s", local_p.name)
        try:
            await _run(["ffmpeg", "-y", "-i", str(local_p), "-vn", "-acodec", "copy", str(audio_path)], step="audio_extraction")
        except Exception:
            await _run(["ffmpeg", "-y", "-i", str(local_p), "-vn", "-acodec", "aac", str(audio_path)], step="audio_extraction")
        return video_path, audio_path, streamer_info

    # ── Kick VOD Direct Downloader (bypasses broken yt-dlp kick:vod extractor) ──
    u_check = url.lower()
    is_kick_vod = "kick.com" in u_check and "/videos/" in u_check
    if is_kick_vod:
        logger.info("Kick VOD detected — using direct Kick API + ffmpeg HLS downloader")
        return await _kick_vod_direct_download(url, video_path, audio_path)

    # ── Kick Live Direct Downloader ──
    is_kick_live = "kick.com" in u_check and not ("/videos/" in u_check or "clips" in u_check)
    if is_kick_live:
        raise PipelineError("download", "This stream is currently live! Please wait until the broadcast ends and send the recorded VOD link so the AI can properly analyze the full video for highlights.")

    live_status = streamer_info.get("live_status", "")
    is_live = "LIVE" in live_status or streamer_info.get("duration", "0") in ("0", "NA", "None")

    # Override: URLs with /videos/ or /v/ in them are always VODs, never live
    if "/videos/" in u_check or "/v/" in u_check or "clips" in u_check:
        is_live = False
        
    if is_live:
        raise PipelineError("download", "This stream is currently live! Please wait until the broadcast ends and send the recorded VOD link so the AI can properly analyze the full video for highlights.")

    logger.info("Downloading %s (effective: %s, is_live: %s)", url, download_url, is_live)

    urls_to_download = [download_url]
    # For Kick VOD links, do NOT add the live channel as fallback — that's what caused the bug
    kick_fallback = normalize_kick_url(url)
    if kick_fallback not in urls_to_download and "/videos/" not in url.lower():
        urls_to_download.append(kick_fallback)

    u_lower = download_url.lower()
    is_youtube = "youtu" in u_lower
    is_kick = "kick.com" in u_lower
    is_twitch = "twitch.tv" in u_lower

    yt_client_chains = YT_CLIENT_CHAINS if is_youtube else [[]]

    impersonate_opts = ["--impersonate", "chrome"] if is_kick else []
    cookie_opts = _get_cookie_opts()
    # For active ongoing live streams (Kick and Twitch), cap download duration to 1 hour (3600s)
    live_downloader_opts = ["--downloader", "ffmpeg", "--downloader-args", "ffmpeg:-t 3600"] if (is_live and (is_kick or is_twitch)) else []
    speed_opts = ["-N", "8", "--concurrent-fragments", "8"] if not (is_live and is_twitch) else []

    download_success = False
    last_err = None

    section_opts = []
    try:
        dur = float(streamer_info.get("duration", "0") or "0")
        if dur > 7200.0 and not is_live:
            logger.info("VOD duration is %.1fh (>2h) — capping download section to first 2.5 hours (*00:00:00-02:30:00)", dur / 3600.0)
            section_opts = ["--download-sections", "*00:00:00-02:30:00"]
    except Exception:
        pass

    for target_url in urls_to_download:
        for yt_opts in yt_client_chains:
            for cmd_opts in [
                [*cookie_opts, *impersonate_opts, *yt_opts, *speed_opts, *live_downloader_opts, *section_opts],
                [*cookie_opts, *speed_opts, *live_downloader_opts, *section_opts],
            ]:
                cmd = [
                    "yt-dlp",
                    "--no-check-certificates",
                    "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]/best",
                    "--merge-output-format", "mp4",
                    "--no-playlist",
                    "--remote-components", "ejs:github",
                    *cmd_opts,
                    "--force-overwrites",
                    "-o", str(video_path),
                    target_url,
                ]
                try:
                    await _run(cmd, step="download")
                    download_success = True
                    break
                except PipelineError as exc:
                    last_err = exc

            if download_success:
                break

        if download_success:
            break

    if not download_success and last_err:
        raise last_err

    logger.info("Extracting audio track from %s", video_path.name)
    await _run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "copy",
            str(audio_path),
        ],
        step="audio_extraction",
    )

    logger.info("Download complete — video: %s, audio: %s", video_path, audio_path)
    return video_path, audio_path, streamer_info


async def download_audio_chunk(
    url: str,
    start_sec: float,
    duration_sec: float,
    output_path: Path,
) -> Path:
    """Download audio chunk for discovery pass."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    local_p = Path(url)
    if local_p.exists() and local_p.is_file():
        try:
            await _run([
                "ffmpeg", "-y", "-ss", str(start_sec), "-i", str(local_p), "-t", str(duration_sec),
                "-vn", "-acodec", "aac", str(output_path)
            ], step="audio_chunk_download")
        except Exception as exc:
            raise PipelineError("audio_chunk_download", str(exc)) from exc
        return output_path

    u_lower = url.lower()
    is_youtube = "youtu" in u_lower
    is_kick = "kick.com" in u_lower
    impersonate_opts = ["--impersonate", "chrome"] if is_kick else []
    cookie_opts = _get_cookie_opts()
    yt_client_chains = YT_CLIENT_CHAINS if is_youtube else [[]]

    # Convert start/duration to HH:MM:SS for yt-dlp download-sections
    def _fmt(sec: float) -> str:
        h = int(sec) // 3600
        m = (int(sec) % 3600) // 60
        s = int(sec) % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    section = f"*{_fmt(start_sec)}-{_fmt(start_sec + duration_sec)}"

    download_success = False
    last_err = None

    for yt_opts in yt_client_chains:
        cmd = [
            "yt-dlp",
            "--no-check-certificates",
            "--no-playlist",
            "--remote-components", "ejs:github",
            *cookie_opts,
            *impersonate_opts,
            *yt_opts,
            "-N", "4",
            "--concurrent-fragments", "4",
            "--download-sections", section,
            "-f", "bestaudio[ext=m4a]/bestaudio/best",
            "--force-overwrites",
            "-o", str(output_path),
            url,
        ]

        logger.info(
            "Downloading audio chunk %.0fs–%.0fs → %s",
            start_sec, start_sec + duration_sec, output_path.name,
        )
        try:
            await _run(cmd, step="audio_chunk_download")
            download_success = True
            break
        except PipelineError as exc:
            last_err = exc
            logger.warning("Audio chunk download attempt failed for %s: %s", url, exc)

    if not download_success and last_err:
        raise last_err

    return output_path


async def download_video_clip_range(
    url: str,
    start_sec: float,
    end_sec: float,
    output_path: Path,
) -> Path:
    """Download video clip range for HD pass."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    local_p = Path(url)
    if local_p.exists() and local_p.is_file():
        dur = end_sec - start_sec
        try:
            await _run([
                "ffmpeg", "-y", "-ss", str(start_sec), "-i", str(local_p), "-t", str(dur),
                "-c:v", "libx264", "-c:a", "aac", "-avoid_negative_ts", "make_zero", str(output_path)
            ], step="clip_download")
        except Exception as exc:
            raise PipelineError("clip_download", str(exc)) from exc
        return output_path

    u_lower = url.lower()
    is_youtube = "youtu" in u_lower
    is_kick = "kick.com" in u_lower
    impersonate_opts = ["--impersonate", "chrome"] if is_kick else []
    cookie_opts = _get_cookie_opts()
    yt_client_chains = YT_CLIENT_CHAINS if is_youtube else [[]]

    def _fmt(sec: float) -> str:
        h = int(sec) // 3600
        m = (int(sec) % 3600) // 60
        s = int(sec) % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    section = f"*{_fmt(start_sec)}-{_fmt(end_sec)}"

    download_success = False
    last_err = None

    for yt_opts in yt_client_chains:
        cmd = [
            "yt-dlp",
            "--no-check-certificates",
            "--no-playlist",
            "--remote-components", "ejs:github",
            *cookie_opts,
            *impersonate_opts,
            *yt_opts,
            "-N", "8",
            "--concurrent-fragments", "8",
            "--download-sections", section,
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            "--merge-output-format", "mp4",
            "--force-overwrites",
            "-o", str(output_path),
            url,
        ]

        logger.info(
            "Downloading HD clip %.1fs–%.1fs → %s",
            start_sec, end_sec, output_path.name,
        )
        try:
            await _run(cmd, step="clip_download")
            download_success = True
            break
        except PipelineError as exc:
            last_err = exc
            logger.warning("Clip download attempt failed for %s: %s", url, exc)

    if not download_success and last_err:
        raise last_err

    return output_path
