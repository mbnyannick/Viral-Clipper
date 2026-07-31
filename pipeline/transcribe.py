"""
Step 3 — Transcription.

Submits all audio chunks to the Deepgram Nova-2 API
*concurrently* via asyncio.gather. Each chunk's returned segments are
offset by that chunk's start time (in seconds) before being merged into
one flat, chronologically-ordered transcript.

Why Deepgram Nova-2:
  Extremely fast, generous free tier, no restrictive concurrency limits,
  word-level timestamps out of the box.
"""

import asyncio
import logging
import os
from pathlib import Path

import httpx

from .errors import PipelineError

logger = logging.getLogger(__name__)


async def _transcribe_one(
    client: httpx.AsyncClient,
    api_key: str,
    chunk_path: Path,
    offset: float,
    max_retries: int = 3,
) -> list[dict]:
    """Transcribe a single chunk using Deepgram and return offset-adjusted segments."""
    logger.info("  Transcribing %s (offset=%.1fs)", chunk_path.name, offset)
    last_exc = None
    
    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&utterances=true&detect_language=true"
    headers = {
        "Authorization": f"Token {api_key}",
    }

    for attempt in range(1, max_retries + 1):
        try:
            with open(chunk_path, "rb") as fh:
                audio_data = fh.read()
            
            response = await client.post(
                url,
                headers=headers,
                content=audio_data,
                timeout=120.0
            )
            response.raise_for_status()
            data = response.json()
            
            # Deepgram returns utterances when utterances=true is passed
            raw_utterances = data.get("results", {}).get("utterances", [])
            segments = []
            
            for utt in raw_utterances:
                utt_start = utt.get("start", 0.0)
                utt_end = utt.get("end", 0.0)
                utt_text = utt.get("transcript", "")
                
                # Extract words for this utterance
                utt_words = []
                for w in utt.get("words", []):
                    w_text = w.get("word", "").strip()
                    w_start = w.get("start", 0.0)
                    w_end = w.get("end", 0.0)
                    if w_text:
                        utt_words.append({
                            "word": w_text,
                            "start": round(w_start + offset, 3),
                            "end": round(w_end + offset, 3),
                        })
                
                segments.append({
                    "text": utt_text.strip(),
                    "start": round(utt_start + offset, 3),
                    "end": round(utt_end + offset, 3),
                    "words": utt_words,
                })
                
            return segments
            
        except Exception as exc:
            last_exc = exc
            logger.warning("Transcription attempt %d/%d failed for %s: %s", attempt, max_retries, chunk_path.name, exc)
            if attempt < max_retries:
                await asyncio.sleep(attempt * 2.0)

    raise PipelineError("transcribe", f"{chunk_path.name}: {last_exc}")


async def transcribe_chunks(
    chunks: list[tuple[Path, float]],
    api_key: str | None = None,
) -> list[dict]:
    """
    Transcribe all chunks concurrently using Deepgram and return a merged, time-sorted
    list of segment dicts: [{"text": str, "start": float, "end": float, "words": [...]}, …]
    """
    if not api_key:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise PipelineError("transcribe", "DEEPGRAM_API_KEY is not set.")

    logger.info("Submitting %d chunks to Deepgram concurrently", len(chunks))

    # Concurrency limit (Deepgram can handle massive concurrency, but we limit to 20 to not overwhelm local memory/network)
    sem = asyncio.Semaphore(20)

    async with httpx.AsyncClient() as client:
        async def _bounded_transcribe(path: Path, offset: float) -> list[dict]:
            async with sem:
                return await _transcribe_one(client, api_key, path, offset, max_retries=3)

        results: list[list[dict]] = await asyncio.gather(
            *(_bounded_transcribe(path, offset) for path, offset in chunks)
        )

    merged: list[dict] = []
    for segment_list in results:
        merged.extend(segment_list)

    if not merged and chunks:
        raise PipelineError("transcribe", "All audio chunks failed transcription after retries.")

    logger.info("Transcription complete — %d segments total", len(merged))
    return merged
