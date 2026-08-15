# pipeline package

import os

DEFAULT_PUBLIC_BASE_URL = "https://132-145-223-32.sslip.io"


def get_public_base_url() -> str:
    """Public-facing base URL for hosted clips/thumbnails.

    Resolution order: PUBLIC_BASE_URL env → singleton default. Set
    PUBLIC_BASE_URL in .env after any IP rotation; never hardcode an IP here
    unless doing so together with a rotation (see scripts/reset_public_ip.sh).
    """
    return os.environ.get("PUBLIC_BASE_URL", DEFAULT_PUBLIC_BASE_URL).rstrip("/")

