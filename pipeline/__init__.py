# pipeline package

import os

DEFAULT_PUBLIC_BASE_URL = "https://150-136-108-208.sslip.io"


def get_public_base_url() -> str:
    """Public-facing base URL for hosted clips/thumbnails (override via PUBLIC_BASE_URL)."""
    return os.environ.get("PUBLIC_BASE_URL", DEFAULT_PUBLIC_BASE_URL).rstrip("/")

