#!/usr/bin/env python3
"""
VIRAL — automatic Oracle public-IP rotation when YouTube flags the egress IP.

Flow:
    1. Probe YouTube via yt-dlp (cookie-less android_vr client) on a canary VOD.
    2. If probe succeeds → exit 0 (nothing to do).
    3. If bot-check detected → use OCI instance principals (no config needed on
       OCI VMs) to detach current ephemeral public IP and attach a fresh one.
    4. Sleep for sshd/network convergence, re-probe from the *new* egress.
    5. Update .env (PUBLIC_BASE_URL) + rewrite the sslip.io host everywhere.
    6. Restart the bot container so it picks up state.
    7. Append to logs/ip_history.log.

Deploy:  install on the Oracle VM ~/VIRAL/scripts/reset_public_ip.py
         cron/systemd: every 15 min, or triggered from a chat command.
 IAM:    enable instance-principal auth + give the instance's dynamic group
         `manage virtual-network-family` (or at minimum: manage public-ips,
         inspect vnics, inspect private-ips).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import oci  # type: ignore
import requests  # type: ignore

# ── Config ────────────────────────────────────────────────────────────────────
VIRAL_DIR = Path(os.environ.get("VIRAL_DIR", Path.home() / "VIRAL")).resolve()
ENV_FILE = VIRAL_DIR / ".env"
COMPOSE_FILE = VIRAL_DIR / "docker-compose.yml"
BOT_CONTAINER = os.environ.get("BOT_CONTAINER", "viral-viral-1")
HISTORY_LOG = VIRAL_DIR / "logs" / "ip_history.log"

# Public YouTube VOD kept tiny; only used as a bot-check canary.
CANARY_URL = os.environ.get("IPRESET_CANARY_URL", "https://youtu.be/dQw4w9WgXcQ")
PROBE_TIMEOUT_SEC = int(os.environ.get("IPRESET_PROBE_TIMEOUT", "45"))
POST_ROTATE_SETTLE_SEC = int(os.environ.get("IPRESET_SETTLE", "20"))
POST_ROTATE_PROBE_TRIES = int(os.environ.get("IPRESET_RETRIES", "8"))

BOT_CHECK_PATTERNS = (
    "Sign in to confirm you're not a bot",
    "Sign in to confirm you’re not a bot",
    "no longer valid. They have likely been rotated",
)

YT_DLP = shutil.which("yt-dlp") or "yt-dlp"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ip-reset] %(levelname)s %(message)s",
)
log = logging.getLogger("ip-reset")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run(cmd: list[str], capture: bool = True, timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"


def _write_history(line: str) -> None:
    HISTORY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_LOG.open("a") as fh:
        fh.write(f"{line}\n")
    log.info("History: %s", line)


def ip_to_sslip(ip: str) -> str:
    return ip.replace(".", "-") + ".sslip.io"


# ── 1. Bot-check probe ────────────────────────────────────────────────────────

def youtube_probe() -> tuple[bool, str]:
    """Return (is_healthy, detail). Cookie-less android_vr client, tiny range."""
    cmd = [
        YT_DLP,
        "--extractor-args", "youtube:player_client=android_vr",
        "--no-playlist",
        "--no-check-certificates",
        "--skip-download",
        "--print", "OK %(title)s | %(duration)s",
        CANARY_URL,
    ]
    rc, out = _run(cmd, timeout=PROBE_TIMEOUT_SEC)
    ok = rc == 0 and "OK " in out
    return ok, out.strip()


# ── 2. OCI IP rotation via instance principals ───────────────────────────────

def get_current_public_ip() -> str:
    try:
        r = requests.get(
            "http://169.254.169.254/opc/v2/vnics/",
            headers={"Authorization": "Bearer Oracle"},
            timeout=5,
        )
        r.raise_for_status()
        vnics = r.json()
        return vnics[0].get("publicIp", "")
    except Exception as exc:
        log.warning("IMDS public IP lookup failed: %s", exc)
        return ""


def rotate_public_ip(old_ip: str) -> str:
    """Detach the ephemeral public IP bound to the primary VNIC, allocate a new one."""
    signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
    vnet = oci.core.VirtualNetworkClient(config={}, signer=signer)
    compute = oci.core.ComputeClient(config={}, signer=signer)

    md = requests.get(
        "http://169.254.169.254/opc/v2/instance/",
        headers={"Authorization": "Bearer Oracle"},
        timeout=5,
    ).json()
    instance_id = md["id"]
    compartment_id = md["compartmentId"]

    # Find primary VNIC attachment
    attachments = compute.list_vnic_attachments(compartment_id=compartment_id, instance_id=instance_id).data
    if not attachments:
        raise RuntimeError("No VNIC attachments found")
    vnic_id = attachments[0].vnic_id
    log.info("Primary VNIC: %s", vnic_id)

    # Locate the private IP currently carrying a public IP
    privates = vnet.list_private_ips(vnic_id=vnic_id).data
    target_private = None
    current_public = None
    for p in privates:
        if p.public_ip_id:  # type: ignore[attr-defined]
            pub = vnet.get_public_ip(p.public_ip_id).data  # type: ignore[attr-defined]
            current_public = pub
            target_private = p
            break
    if not target_private:
        raise RuntimeError("No private IP with a public IP attached")

    log.info("Detaching old public IP: %s (%s)", current_public.ip_address, current_public.id)
    vnet.delete_public_ip(current_public.id)

    log.info("Creating fresh ephemeral public IP…")
    created = vnet.create_public_ip(
        oci.core.models.CreatePublicIpDetails(
            compartment_id=compartment_id,
            lifetime="EPHEMERAL",
            private_ip_id=target_private.id,
            display_name=f"viral-ephemeral-{int(time.time())}",
        )
    ).data
    new_ip = created.ip_address

    # Wait until assigned
    for _ in range(30):
        time.sleep(2)
        refreshed = vnet.get_public_ip(created.id).data
        if refreshed.lifecycle_state == "AVAILABLE" and refreshed.private_ip_id:
            break

    log.info("New public IP: %s", new_ip)
    return new_ip


# ── 3. Reconfig ──────────────────────────────────────────────────────────────

def patch_env(new_url: str, env_path: Path) -> None:
    """Set PUBLIC_BASE_URL + MAKE_WEBHOOK_URL in env_path to new_url."""
    if not env_path.exists():
        log.warning(".env not found at %s", env_path)
        return
    text = env_path.read_text()
    new_lines = []
    seen = {"PUBLIC_BASE_URL": False, "MAKE_WEBHOOK_URL": False}
    for line in text.splitlines():
        if line.startswith("PUBLIC_BASE_URL="):
            line = f"PUBLIC_BASE_URL={new_url}"
            seen["PUBLIC_BASE_URL"] = True
        elif line.startswith("MAKE_WEBHOOK_URL="):
            line = f"MAKE_WEBHOOK_URL={new_url}/webhook/viral-post"
            seen["MAKE_WEBHOOK_URL"] = True
        new_lines.append(line)
    for key, was_set in seen.items():
        if not was_set:
            if key == "PUBLIC_BASE_URL":
                new_lines.append(f"PUBLIC_BASE_URL={new_url}")
            else:
                new_lines.append(f"MAKE_WEBHOOK_URL={new_url}/webhook/viral-post")
    env_path.write_text("\n".join(new_lines) + "\n")
    log.info("Updated %s", env_path)


def restart_bot() -> None:
    rc, out = _run(["docker", "restart", BOT_CONTAINER], timeout=90)
    log.info("docker restart rc=%d out=%s", rc, out.strip())


# ── 4. Main flow ─────────────────────────────────────────────────────────────

def main() -> int:
    healthy, detail = youtube_probe()
    if healthy:
        log.info("YouTube probe OK, nothing to do.")
        return 0

    old_ip = get_current_public_ip()
    log.warning("Bot-check detected from %s (detail: %s) — rotating IP…", old_ip, detail[:160])

    try:
        new_ip = rotate_public_ip(old_ip)
    except Exception as exc:
        log.error("Rotation failed: %s", exc)
        _write_history(f"{_now()} FAIL rotate from={old_ip} error={exc}")
        return 2

    log.info("Settling %ss for sshd/routing…", POST_ROTATE_SETTLE_SEC)
    time.sleep(POST_ROTATE_SETTLE_SEC)

    # Re-probe from the new egress
    for i in range(1, POST_ROTATE_PROBE_TRIES + 1):
        healthy, detail = youtube_probe()
        if healthy:
            log.info("Post-rotate probe OK (attempt %d)", i)
            break
        log.warning("Post-rotate probe %d/%d still bad: %s", i, POST_ROTATE_PROBE_TRIES, detail[:120])
        time.sleep(10)
    else:
        log.error("Even new IP is bot-checked — YouTube escalating? Manual action needed.")
        _write_history(f"{_now()} FAIL new_ip={new_ip} still_flagged")
        return 3

    new_url = f"https://{ip_to_sslip(new_ip)}"
    patch_env(new_url, ENV_FILE)
    restart_bot()
    _write_history(f"{_now()} OK old={old_ip} new={new_ip} url={new_url}")
    log.info("Rotation complete. New URL: %s", new_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
