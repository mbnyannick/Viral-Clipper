"""
bot/scheduler.py — Smart Peak-Hour Post Scheduler

Queues social media posts and fires them at peak engagement windows (US Eastern Time).
Rules:
  - YouTube Shorts: max 3 posts per calendar day (ET), slots at 12pm, 2pm, 7pm ET
  - TikTok:         unlimited, slots at 7am, 12pm, 6pm, 9pm ET
  - Instagram:      unlimited, slots at 7am, 11am, 7pm ET
  - Facebook:       unlimited, slots at 1pm, 6pm, 9pm ET

State is persisted to tmp/schedule_state.json so pending posts survive bot restarts.
"""

import asyncio
import json
import logging
import os
import random
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from pipeline import get_public_base_url

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

PEAK_HOURS: dict[str, list[int]] = {
    "youtube":   [12, 14, 19],                      # 3/day — 12pm, 2pm, 7pm ET
    "tiktok":    [7, 10, 12, 15, 18, 21],           # 6/day — 7am, 10am, 12pm, 3pm, 6pm, 9pm ET
    "instagram": [7, 11, 15, 19],                   # 4/day — 7am, 11am, 3pm, 7pm ET
    "facebook":  [7, 9, 11, 13, 15, 17, 19, 21],   # 8/day — every ~2hrs from 7am to 9pm ET
}

YOUTUBE_DAILY_LIMIT = 3
STATE_FILE = Path("tmp/schedule_state.json")

# Jitter: each post fires at a random offset within ±JITTER_MINUTES of its peak slot.
# This makes posting times look human — 7:00am becomes 6:52am or 7:11am, etc.
JITTER_MINUTES = 18  # max offset in either direction

# Platform stagger: when scheduling one clip to multiple platforms at once,
# add this many minutes between each platform to avoid simultaneous API hits.
PLATFORM_STAGGER_MIN = 5   # minimum minutes between platforms
PLATFORM_STAGGER_MAX = 22  # maximum minutes between platforms



DAILY_LIMITS: dict[str, int] = {
    "youtube":   YOUTUBE_DAILY_LIMIT,  # hard cap: 3/day
    "tiktok":    6,
    "instagram": 4,
    "facebook":  8,
}


def _now_et() -> datetime:
    return datetime.now(ET)


def _et_date_str(dt: datetime | None = None) -> str:
    return (dt or _now_et()).strftime("%Y-%m-%d")


def next_peak_slot(platform: str, after: datetime | None = None) -> datetime:
    """Thin wrapper kept for import compatibility — delegates to scheduler.next_available_slot()."""
    return scheduler.next_available_slot(platform, after=after)


class PlatformScheduler:
    def __init__(self) -> None:
        self._queue: list[dict] = []
        self._yt_daily: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._load()

    def youtube_count_today(self) -> int:
        return self._yt_daily.get(_et_date_str(), 0)

    def youtube_slots_remaining_today(self) -> int:
        return max(0, YOUTUBE_DAILY_LIMIT - self.youtube_count_today())

    def _queued_slots_for(self, platform: str, date_str: str) -> set[int]:
        """Return the set of peak hours already queued for a platform on a given date."""
        occupied: set[int] = set()
        for item in self._queue:
            if item["platform"] == platform and item["fire_at"].startswith(date_str):
                try:
                    h = datetime.fromisoformat(item["fire_at"]).astimezone(ET).hour
                    occupied.add(h)
                except Exception:
                    pass
        return occupied

    def _posted_count_for(self, platform: str, date_str: str) -> int:
        """Return the number of already-fired posts for a platform on a given date."""
        if platform == "youtube":
            return self._yt_daily.get(date_str, 0)
        # For other platforms we trust the queue tracking (fired items are removed from queue)
        return 0

    def next_available_slot(self, platform: str, after: datetime | None = None) -> datetime:
        """
        Return the next available peak slot for the platform that is:
          1. In the future (after `after`, defaults to now ET)
          2. Not already occupied by a queued post on that slot
          3. Within the daily cap for that platform

        If today's cap is full, rolls to the next day's first available peak slot.
        Searches up to 14 days ahead to find a free slot.
        """
        now = (after or _now_et()).astimezone(ET)
        hours = PEAK_HOURS.get(platform, [12, 18])
        daily_limit = DAILY_LIMITS.get(platform, 999)

        for day_offset in range(0, 14):  # search up to 2 weeks ahead
            candidate_day = now + timedelta(days=day_offset)
            date_str = _et_date_str(candidate_day)

            posted = self._posted_count_for(platform, date_str)
            occupied_hours = self._queued_slots_for(platform, date_str)
            slots_used = posted + len(occupied_hours)

            if slots_used >= daily_limit:
                # This day is fully booked — try next day
                continue

            for h in sorted(hours):
                slot = candidate_day.replace(hour=h, minute=0, second=0, microsecond=0)
                if slot <= now:
                    continue  # this slot is in the past
                if h in occupied_hours:
                    continue  # already booked
                return slot  # found a free future peak slot

        # Ultimate fallback: next day noon
        return (now + timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)

    def next_available_youtube_slot(self) -> datetime:
        """Alias for backwards compatibility."""
        return self.next_available_slot("youtube")

    def schedule_clip(self, platform: str, payload: dict, fire_at: datetime) -> None:
        """Schedule a clip with human-like jitter applied to the fire time."""
        # Apply random jitter so posts never fire at exact clock marks
        jitter_secs = random.randint(-JITTER_MINUTES * 60, JITTER_MINUTES * 60)
        jittered_fire_at = fire_at + timedelta(seconds=jitter_secs)
        item = {
            "platform": platform,
            "payload": payload,
            "fire_at": jittered_fire_at.isoformat(),
            "chat_id": payload.get("chat_id"),
        }
        self._queue.append(item)
        self._save()
        logger.info(
            "Scheduled %s clip for %s ET (jitter: %+ds)",
            platform,
            jittered_fire_at.astimezone(ET).strftime("%b %d %I:%M%p"),
            jitter_secs,
        )

    def schedule_clip_staggered(self, platforms: list[str], payload: dict, base_fire_at: datetime | None = None) -> dict[str, datetime]:
        """
        Schedule the same clip to multiple platforms with:
          1. Per-platform jitter (random offset from peak slot)
          2. Inter-platform stagger (each platform fires minutes apart)

        Returns a dict of {platform: actual_fire_datetime} for confirmation display.
        """
        fired_at: dict[str, datetime] = {}
        cumulative_stagger = timedelta(0)

        for plat in platforms:
            slot = base_fire_at or self.next_available_slot(plat)
            # Add cumulative stagger so platforms don't all fire at the same time
            staggered_slot = slot + cumulative_stagger
            self.schedule_clip(plat, {**payload, "platform": plat}, staggered_slot)
            fired_at[plat] = staggered_slot
            # Add a random inter-platform delay before the next platform
            stagger_mins = random.randint(PLATFORM_STAGGER_MIN, PLATFORM_STAGGER_MAX)
            cumulative_stagger += timedelta(minutes=stagger_mins)

        return fired_at

    def get_schedule_preview(self, clip_num: str | int, platforms: list[str]) -> str:
        now_et = _now_et()
        today_str = _et_date_str()
        lines = [f"📅 <b>Clip #{clip_num} Scheduled for Peak Hours!</b>\n"]
        for platform in platforms:
            slot = self.next_available_slot(platform)
            slot_et = slot.astimezone(ET)
            slot_date_str = _et_date_str(slot_et)
            emoji = {"tiktok": "📱", "instagram": "📸", "facebook": "📘", "youtube": "🔴"}.get(platform, "🌐")
            label = {"tiktok": "TikTok", "instagram": "Instagram Reels", "facebook": "Facebook Reels", "youtube": "YouTube Shorts"}.get(platform, platform.title())
            day_note = ""
            if slot_date_str != today_str:
                day_note = " ⟳ <i>(today full — rolled to next day)</i>"
            elif platform == "youtube":
                remaining = self.youtube_slots_remaining_today()
                if remaining <= 1:
                    day_note = " (last YouTube slot today)"
            lines.append(f"{emoji} <b>{label}</b> → {slot_et.strftime('%b %d, %I:%M%p ET')}{day_note}")

        lines.append("\n💡 <i>Posts fire automatically. Use /schedule to view all pending posts.</i>")
        return "\n".join(lines)

    def get_pending_summary(self, chat_id: int) -> str:
        my_items = [i for i in self._queue if i.get("chat_id") == chat_id]
        if not my_items:
            return "📅 <b>Your Schedule</b>\n\nNo posts currently scheduled."
        lines = [f"📅 <b>Your Scheduled Posts ({len(my_items)} pending)</b>\n"]
        for item in sorted(my_items, key=lambda x: x["fire_at"]):
            try:
                ft = datetime.fromisoformat(item["fire_at"]).astimezone(ET)
                ft_str = ft.strftime("%b %d, %I:%M%p ET")
            except Exception:
                ft_str = item["fire_at"]
            clip_id = item["payload"].get("clip_id", "?")
            plat = item["platform"].title()
            emoji = {"tiktok": "📱", "instagram": "📸", "facebook": "📘", "youtube": "🔴"}.get(item["platform"], "🌐")
            lines.append(f"{emoji} {plat} — {clip_id} → {ft_str}")
        yt_used = self.youtube_count_today()
        lines.append(f"\n📊 YouTube today: {yt_used}/{YOUTUBE_DAILY_LIMIT} posts used")
        return "\n".join(lines)

    async def tick(self) -> None:
        async with self._lock:
            now = datetime.now(timezone.utc)
            due = []
            remaining = []
            for item in self._queue:
                try:
                    fire_at = datetime.fromisoformat(item["fire_at"])
                    if fire_at.tzinfo is None:
                        fire_at = fire_at.replace(tzinfo=ET)
                    if fire_at.astimezone(timezone.utc) <= now:
                        due.append(item)
                    else:
                        remaining.append(item)
                except Exception:
                    remaining.append(item)
            if not due:
                return
            self._queue = remaining
            self._save()

        for item in due:
            await self._fire(item)

    async def _fire(self, item: dict) -> None:
        platform = item["platform"]
        payload = item["payload"]
        webhook_url = os.environ.get(
            "MAKE_WEBHOOK_URL",
            f"{get_public_base_url()}/webhook/viral-post",
        ).strip()

        logger.info("⏰ Scheduler firing: %s → %s", platform, payload.get("clip_id", "?"))

        def _post_sync() -> tuple[int, str]:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=data_bytes,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(data_bytes)),
                    "User-Agent": "ViralBot-Scheduler/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")

        try:
            status, body = await asyncio.to_thread(_post_sync)
            if status in (200, 201, 202):
                logger.info("✅ Scheduler fired OK: %s %s", platform, payload.get("clip_id", ""))
                if platform == "youtube":
                    today = _et_date_str()
                    self._yt_daily[today] = self._yt_daily.get(today, 0) + 1
                    self._save()
                    logger.info("📊 YouTube posts today: %d/%d", self._yt_daily[today], YOUTUBE_DAILY_LIMIT)
            else:
                logger.warning("⚠️ Scheduler HTTP %d for %s: %s", status, platform, body[:200])
        except Exception as exc:
            logger.error("❌ Scheduler fire error for %s: %s", platform, exc)

    def _save(self) -> None:
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(json.dumps({"queue": self._queue, "yt_daily": self._yt_daily}, indent=2))
        except Exception as exc:
            logger.warning("Could not save scheduler state: %s", exc)

    def _load(self) -> None:
        try:
            if STATE_FILE.exists():
                state = json.loads(STATE_FILE.read_text())
                self._queue = state.get("queue", [])
                self._yt_daily = state.get("yt_daily", {})
                today = _et_date_str()
                self._yt_daily = {k: v for k, v in self._yt_daily.items() if k >= today}
                logger.info(
                    "Scheduler loaded: %d pending posts, YouTube today: %d/%d",
                    len(self._queue),
                    self.youtube_count_today(),
                    YOUTUBE_DAILY_LIMIT,
                )
        except Exception as exc:
            logger.warning("Could not load scheduler state: %s", exc)
            self._queue = []
            self._yt_daily = {}


_LIVE_NOTIFIED_CACHE: set[str] = set()


async def run_live_streamer_monitor(bot=None) -> None:
    """Check Top 20 Streamers roster for live status and send Telegram alerts."""
    roster_path = Path("config/streamer_roster.json")
    if not roster_path.exists():
        return

    op_id = os.environ.get("TELEGRAM_OPERATOR_CHAT_ID", "").strip()
    if not op_id or op_id == "0":
        return

    try:
        data = json.loads(roster_path.read_text())
        streamers = data.get("streamers", [])
    except Exception:
        return

    from pipeline.download import check_streamer_live_status

    for s in streamers:
        url = s.get("url", "")
        sid = s.get("id", "")
        if not url or not sid:
            continue

        try:
            is_live, s_name, title = await check_streamer_live_status(url)
            cache_key = f"{sid}_{datetime.now().strftime('%Y%m%d_%H')}"
            if is_live and cache_key not in _LIVE_NOTIFIED_CACHE:
                _LIVE_NOTIFIED_CACHE.add(cache_key)
                plat = s.get("platform", "Stream")
                plat_emoji = "🟣" if plat == "Twitch" else ("🟩" if plat == "Kick" else "▶️")

                alert_text = (
                    f"🔴 <b>LIVE STREAM ALERT: {s_name} is LIVE NOW on {plat}!</b>\n\n"
                    f"• <b>Streamer:</b> {s_name}\n"
                    f"• <b>Title:</b> {title if title else 'N/A'}\n"
                    f"• <b>Platform:</b> {plat_emoji} {plat}\n\n"
                    f"<i>Tap below to clip the live stream instantly!</i>"
                )
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("⚡ Clip Live Stream Now", callback_data=f"clip_streamer:{sid}")]
                ])
                if bot:
                    await bot.send_message(chat_id=op_id, text=alert_text, reply_markup=keyboard, parse_mode="HTML")
                logger.info("📡 Live stream alert sent to Telegram for %s (%s)", s_name, plat)
        except Exception as exc:
            logger.warning("Live monitor check error for %s: %s", sid, exc)


scheduler = PlatformScheduler()


async def run_scheduler_loop(app=None) -> None:
    logger.info("🕐 Peak-hour scheduler & Live Streamer Monitor started (ticks every 60s)")
    bot_instance = getattr(app, "bot", None) if app else None
    while True:
        try:
            await scheduler.tick()
            await run_live_streamer_monitor(bot_instance)
        except Exception as exc:
            logger.exception("Scheduler / Live Monitor tick error: %s", exc)
        await asyncio.sleep(60)
