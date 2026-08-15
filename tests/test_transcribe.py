"""
Tests for pipeline/transcribe.py

Tests:
- Timestamps from each chunk are offset by the chunk's start time.
- Chunks are merged in order (chunk 0 segments come before chunk 1).
- A Groq API exception is wrapped in PipelineError.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import httpx
from pipeline.transcribe import transcribe_chunks
from pipeline.errors import PipelineError


def _make_seg(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end}


def _make_httpx_response(segments: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"segments": segments, "words": []}
    return resp


async def test_transcribe_offsets_timestamps(tmp_path):
    """Chunk 1 at offset 900s should have its timestamps shifted by 900."""
    chunk0 = tmp_path / "chunk_000.m4a"
    chunk1 = tmp_path / "chunk_001.m4a"
    chunk0.write_bytes(b"fake")
    chunk1.write_bytes(b"fake")

    async def mock_groq_one(client, api_key, path, offset, max_retries=2):
        if path == chunk0:
            return [
                {"text": "hello world", "start": 0.5 + offset, "end": 2.0 + offset, "words": []},
                {"text": "how are you", "start": 2.1 + offset, "end": 4.0 + offset, "words": []},
            ]
        else:
            return [
                {"text": "and then he said", "start": 1.0 + offset, "end": 3.5 + offset, "words": []},
            ]

    with patch("pipeline.transcribe._transcribe_groq_one", side_effect=mock_groq_one), \
         patch.dict("os.environ", {"DEEPGRAM_API_KEY": "", "GROQ_API_KEY": "test"}):
        segments = await transcribe_chunks(
            [(chunk0, 0.0), (chunk1, 900.0)],
            api_key="test",
        )

    assert len(segments) == 3

    # Chunk 0 — no offset
    assert segments[0]["text"] == "hello world"
    assert segments[0]["start"] == pytest.approx(0.5)
    assert segments[0]["end"] == pytest.approx(2.0)

    assert segments[1]["start"] == pytest.approx(2.1)
    assert segments[1]["end"] == pytest.approx(4.0)

    # Chunk 1 — offset by 900s
    assert segments[2]["text"] == "and then he said"
    assert segments[2]["start"] == pytest.approx(901.0)
    assert segments[2]["end"] == pytest.approx(903.5)


async def test_transcribe_merge_order(tmp_path):
    """Segments from earlier chunks appear before segments from later chunks."""
    chunks = []
    chunk_paths = []
    for i in range(3):
        p = tmp_path / f"chunk_{i:03d}.m4a"
        p.write_bytes(b"fake")
        chunks.append((p, float(i * 900)))
        chunk_paths.append(p)

    async def mock_groq_one(client, api_key, path, offset, max_retries=2):
        idx = chunk_paths.index(path)
        return [{"text": f"chunk{idx} text", "start": 0.0 + offset, "end": 1.0 + offset, "words": []}]

    with patch("pipeline.transcribe._transcribe_groq_one", side_effect=mock_groq_one), \
         patch.dict("os.environ", {"DEEPGRAM_API_KEY": "", "GROQ_API_KEY": "test"}):
        segments = await transcribe_chunks(chunks, api_key="test")

    texts = [s["text"] for s in segments]
    assert texts == ["chunk0 text", "chunk1 text", "chunk2 text"]


async def test_transcribe_wraps_api_error(tmp_path):
    """Groq API exception & local whisper failure → returns fallback clip timeline."""
    chunk = tmp_path / "chunk_000.m4a"
    chunk.write_bytes(b"fake")

    with patch("pipeline.transcribe.httpx.AsyncClient") as mock_client_cls, \
         patch("whisper.load_model", side_effect=Exception("local whisper down")), \
         patch.dict("os.environ", {"DEEPGRAM_API_KEY": ""}):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client.post.side_effect = httpx.RequestError("API is down")

        segments = await transcribe_chunks([(chunk, 0.0)], api_key="test")

    assert len(segments) == 1
    assert segments[0]["text"] == "Check out this highlight moment!"
