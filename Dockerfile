FROM python:3.11-slim

# Install system deps: ffmpeg for A/V processing, Noto Color Emoji for caption rendering, nodejs/curl/unzip for yt-dlp JS execution
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-noto-color-emoji \
        nodejs \
        curl \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Deno JS runtime for yt-dlp n-sig solver
RUN curl -fsSL https://deno.land/install.sh | sh
ENV PATH="/root/.deno/bin:${PATH}"

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create runtime directories (volumes may mount over these, which is fine)
RUN mkdir -p tmp logs

CMD ["python", "-m", "bot.main"]
