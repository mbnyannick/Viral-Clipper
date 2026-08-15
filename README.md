# VIRAL

Personal, self-hosted Telegram bot that takes a long-form VOD link and returns finished,
caption-branded 9:16 vertical clips — no manual editing, no third-party clipping service.

**Pipeline:** download → chunk → transcribe → score → cut → render → composite → deliver → cleanup

Two engines:
- **Classic mode** (`bot/run_pipeline.py`) — full VOD: download → chunk → transcribe → score → cut → composite → deliver.
- **Progressive mode** (`pipeline/streaming_pipeline.py`) — live streams & Kick/Twitch VODs: resolves the raw HLS URL, downloads the full scan-range audio **once**, then slices 10-minute windows locally in parallel, ranks moments globally, and delivers each clip as soon as it's ready.

---

## Requirements

- Docker + Docker Compose (runs on Oracle Cloud Free Tier ARM / any Linux VM)
- Telegram bot token (from [@BotFather](https://t.me/BotFather))
- Your Telegram user ID (from [@userinfobot](https://t.me/userinfobot))
- [Groq API key](https://console.groq.com) (Whisper Large v3 Turbo transcription fallback)
- [Deepgram API key](https://deepgram.com) (Nova-2 primary transcription)
- [DeepSeek API key](https://platform.deepseek.com) (moment scoring)

---

## Setup

### 1. Clone and enter the repo

```bash
git clone <your-repo-url> VIRAL
cd VIRAL
```

### 2. Copy and fill in secrets

```bash
cp .env.example .env
nano .env   # fill in all values
```

### 3. Add fonts

The bot requires two font files in `assets/fonts/`:

**`Bold.ttf`** — main caption font. Montserrat Bold is recommended:
```bash
mkdir -p assets/fonts
# Download from Google Fonts (free, OFL licensed):
wget -O assets/fonts/Bold.ttf \
  "https://github.com/google/fonts/raw/main/ofl/montserrat/static/Montserrat-Bold.ttf"
```

**`NotoColorEmoji.ttf`** — color emoji rendering (~20 MB, installed via apt in Docker):
```bash
# This is installed automatically inside the Docker image via:
#   apt-get install fonts-noto-color-emoji
# No manual step needed for Docker deployment.
#
# For local development outside Docker, either:
#   brew install font-noto-color-emoji          (macOS)
#   sudo apt install fonts-noto-color-emoji     (Debian/Ubuntu)
# or place the TTF at assets/fonts/NotoColorEmoji.ttf manually.
```

### 4. Add a watermark (optional)

Place your logo/handle graphic at `assets/watermark.png`.
It will be composited at the bottom-right of every clip.
If the file is missing, the composite step will fail — create a placeholder:

```bash
# Create a minimal 1×1 transparent PNG as a placeholder
python3 -c "
from PIL import Image
Image.new('RGBA', (1, 1), (0,0,0,0)).save('assets/watermark.png')
"
```

### 5. Build and start

```bash
docker compose up -d --build
docker compose logs -f   # watch startup logs
```

### 6. Verify the bot is running

Send any text to your bot in Telegram. It should reply with:
> 🎬 Send me a YouTube, Kick, or Twitch VOD link and I'll clip it.

---

## Usage

Send a video link (YouTube, Kick, Twitch VOD) to the bot in Telegram:

```
https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

The bot will:
1. Acknowledge receipt immediately.
2. Post a status update when transcription completes.
3. Post a status update when moment scoring completes.
4. Deliver all clips as a batched Telegram album.

If any step fails, the bot reports which step broke and why — no silent failures.

---

## Configuration

All tuning options are in `.env`:

| Variable | Default | Description |
|---|---|---|
| `TOP_N_CLIPS` | `10` | Number of highlight moments to extract |
| `CHUNK_DURATION_MINUTES` | `15` | Audio chunk size for transcription |
| `DEEPSEEK_MODEL` | `deepseek-chat` | `deepseek-chat` (fast) or `deepseek-reasoner` (higher quality) |
| `PUBLIC_BASE_URL` | `https://132-145-223-32.sslip.io` | Public base URL for hosted clips/thumbnails |
| `N8N_WEBHOOK_URL` | `{PUBLIC_BASE_URL}:5678/webhook/viral-post` | n8n Webhook for auto-posting clips |
| `CLIP_WINDOW_MINUTES` | `60` | Progressive mode: scan window size for live streams |

**YouTube cookies:** the container reads `cookies.txt` at the repo root (Netscape format, exported via yt-dlp). This authenticates YouTube requests and avoids the "Sign in to confirm you're not a bot" block. Re-export it periodically as cookies expire.

---

## Project Structure

```
VIRAL/
├── bot/
│   ├── main.py          # Entry point — starts Telegram bot
│   ├── handlers.py      # URL ingestion handler
│   └── run_pipeline.py  # Async pipeline orchestrator
├── pipeline/
│   ├── errors.py        # PipelineError — structured failure type
│   ├── download.py      # yt-dlp download + ffmpeg audio extraction
│   ├── chunk.py         # ffmpeg audio segmenting
│   ├── transcribe.py    # Groq Whisper concurrent transcription
│   ├── score.py         # DeepSeek LLM moment scoring
│   ├── clip.py          # ffmpeg clip cutting (concurrent)
│   ├── caption.py       # Pillow caption PNG rendering
│   ├── composite.py     # ffmpeg pillarbox + overlay compositing
│   └── deliver.py       # Telegram sendMediaGroup delivery
├── assets/
│   ├── fonts/
│   │   └── Bold.ttf     # Your bundled bold font
│   └── watermark.png    # Your handle/logo graphic
├── tests/
│   ├── test_score.py
│   ├── test_transcribe.py
│   └── test_caption.py
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## Running Tests

```bash
# Install deps locally (for development)
pip install -r requirements.txt

# Run tests
pytest
```

Font-dependent caption render tests auto-skip if `assets/fonts/Bold.ttf` is not present.

---

## Cost Estimate

| Source | Cost |
|---|---|
| Deepgram Nova-2 transcription (4-hour VOD) | ~$0.50 |
| DeepSeek moment scoring (one call) | ~$0.01–$0.05 |
| Oracle Always Free ARM VM | $0 |
| **Total per run** | **< $0.60** |

---

## Maintenance Notes

- **yt-dlp updates:** Run `docker compose build --no-cache` periodically to pull the latest
  yt-dlp, which may be needed if YouTube/Kick/Twitch update their anti-scraping measures.
- **YouTube bot-checks:** If you see "Sign in to confirm you're not a bot" in the logs, re-export
  `cookies.txt` from a logged-in browser and `docker compose restart viral`.
- **Logs:** Available at `./logs/viral.log` (host-mounted volume, 10 MB rotating).
- **Crash recovery:** The `restart: always` policy in `docker-compose.yml` ensures the bot
  auto-restarts without manual intervention.
- **Disk:** Each pipeline run cleans up its own scratch files after delivery. The `tmp/`
  directory should stay empty between runs.
