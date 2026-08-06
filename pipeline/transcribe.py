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


async def _transcribe_groq_one(
    client: httpx.AsyncClient,
    api_key: str,
    chunk_path: Path,
    offset: float,
    max_retries: int = 3,
) -> list[dict]:
    """Transcribe a single chunk using Groq Whisper Large v3 Turbo API."""
    logger.info("  Transcribing %s via Groq Whisper (offset=%.1fs)", chunk_path.name, offset)
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(1, max_retries + 1):
        try:
            with open(chunk_path, "rb") as fh:
                files = {"file": (chunk_path.name, fh, "audio/m4a")}
                data = {
                    "model": "whisper-large-v3-turbo",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": ["word", "segment"],
                }

                response = await client.post(url, headers=headers, files=files, data=data, timeout=120.0)

            response.raise_for_status()
            res_json = response.json()

            segments = []
            words = res_json.get("words") or []
            for seg in (res_json.get("segments") or []):
                seg_start = round(seg.get("start", 0.0) + offset, 3)
                seg_end = round(seg.get("end", 0.0) + offset, 3)
                seg_text = seg.get("text", "").strip()

                seg_words = []
                s_words = seg.get("words") or words
                for w in (s_words or []):
                    w_start = round(w.get("start", 0.0) + offset, 3)
                    w_end = round(w.get("end", 0.0) + offset, 3)
                    if seg_start - 0.5 <= w_start <= seg_end + 0.5:
                        seg_words.append({
                            "word": w.get("word", "").strip(),
                            "start": w_start,
                            "end": w_end,
                            "confidence": 0.95,
                        })

                segments.append({
                    "text": seg_text,
                    "start": seg_start,
                    "end": seg_end,
                    "words": seg_words,
                })

            return segments
        except Exception as exc:
            logger.warning("Groq Whisper attempt %d failed for %s: %s", attempt, chunk_path.name, exc)
            if attempt < max_retries:
                await asyncio.sleep(attempt * 2.0)

    raise PipelineError("transcribe", f"{chunk_path.name}: Groq transcription failed")


async def transcribe_chunks(
    chunks: list[tuple[Path, float]],
    api_key: str | None = None,
) -> list[dict]:
    """
    Transcribe all chunks concurrently using Deepgram or Groq Whisper API with auto-fallback.
    """
    deepgram_key = api_key or os.getenv("DEEPGRAM_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    # Primary try: Deepgram if key is available and not starting with gsk_
    if deepgram_key and not deepgram_key.startswith("gsk_"):
        try:
            logger.info("Submitting %d chunks to Deepgram Nova-2 concurrently", len(chunks))
            sem = asyncio.Semaphore(20)
            async with httpx.AsyncClient() as client:
                async def _bounded_dg(path: Path, offset: float) -> list[dict]:
                    async with sem:
                        return await _transcribe_one(client, deepgram_key, path, offset, max_retries=2)

                results = await asyncio.gather(*(_bounded_dg(p, o) for p, o in chunks))

            merged = []
            for sl in results:
                merged.extend(sl)

            if merged:
                logger.info("Deepgram transcription complete — %d segments total", len(merged))
                return merged
        except Exception as dg_exc:
            logger.warning("Deepgram transcription failed (%s) — switching to Groq Whisper fallback...", dg_exc)

    # Fallback / Primary try: Groq Whisper API
    if groq_key:
        logger.info("Submitting %d chunks to Groq Whisper Large v3 Turbo concurrently", len(chunks))
        sem = asyncio.Semaphore(20)
        async with httpx.AsyncClient() as client:
            async def _bounded_groq(path: Path, offset: float) -> list[dict]:
                async with sem:
                    return await _transcribe_groq_one(client, groq_key, path, offset, max_retries=3)

            results = await asyncio.gather(*(_bounded_groq(p, o) for p, o in chunks))

        merged = []
        for sl in results:
            merged.extend(sl)

        if merged:
            logger.info("Groq Whisper transcription complete — %d segments total", len(merged))
            return merged

    raise PipelineError("transcribe", "All transcription services failed (Deepgram / Groq Whisper).")


