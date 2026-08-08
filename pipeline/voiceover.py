"""
pipeline/voiceover.py — AI Voiceover Narration Generator

Synthesizes short hook narration commentary (1-2 sentences) for clips using:
1. OpenAI TTS API (if OPENAI_API_KEY is present)
2. gTTS / Edge-TTS fallback (free offline/lightweight TTS)
"""

import asyncio
import logging
import os
import sys
from pathlib import Path


logger = logging.getLogger(__name__)


DEFAULT_VOICE = os.environ.get("VOICEOVER_SPEAKER", "onyx").strip()  # "onyx" = deep studio male narrator voice


def voiceover_generation_enabled() -> bool:
    """Workflow policy: voiceover narration is permanently disabled."""
    return False


async def generate_voiceover(text: str, output_path: Path, voice: str | None = None) -> Path | None:
    """Voiceover narration synthesis is permanently discontinued for the workflow."""
    return None

    clean_text = text.strip()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fish Audio TTS (s2.1-pro — Primary Consistent Voice Actor)
    fish_key = os.environ.get("FISH_AUDIO_API_KEY", "").strip()
    if fish_key:
        try:
            import json
            import urllib.request
            fish_voice_id = os.environ.get("FISH_AUDIO_VOICE_ID", "").strip()
            req_body = {
                "text": clean_text,
                "format": "mp3",
                "mp3_bitrate": 128,
            }
            if fish_voice_id:
                req_body["reference_id"] = fish_voice_id
            req_data = json.dumps(req_body).encode("utf-8")
            headers = {
                "Authorization": f"Bearer {fish_key}",
                "Content-Type": "application/json",
                "model": "s2.1-pro-free",
            }
            req = urllib.request.Request(
                "https://api.fish.audio/v1/tts",
                data=req_data,
                headers=headers,
                method="POST",
            )
            def _fetch_fish():
                with urllib.request.urlopen(req, timeout=20) as resp:
                    output_path.write_bytes(resp.read())
            await asyncio.to_thread(_fetch_fish)
            if output_path.exists() and output_path.stat().st_size > 1000:
                logger.info("Fish Audio TTS voiceover generated: %s (size=%d bytes)", output_path.name, output_path.stat().st_size)
                return output_path
        except Exception as exc:
            logger.warning("Fish Audio TTS failed: %s", exc)

    # 2. OpenAI TTS-HD Studio Human Voice ("onyx" studio male narrator)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            import json
            import urllib.request
            chosen_voice = voice or DEFAULT_VOICE
            req_data = json.dumps({
                "model": "tts-1-hd",
                "input": clean_text,
                "voice": chosen_voice,
                "speed": 1.15,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.openai.com/v1/audio/speech",
                data=req_data,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST"
            )
            def _fetch_openai():
                with urllib.request.urlopen(req, timeout=15) as resp:
                    output_path.write_bytes(resp.read())
            await asyncio.to_thread(_fetch_openai)
            if output_path.exists() and output_path.stat().st_size > 1000:
                logger.info("OpenAI TTS-HD studio human voiceover generated (%s): %s", chosen_voice, output_path.name)
                return output_path
        except Exception as exc:
            logger.warning("OpenAI TTS-HD failed: %s", exc)




    # 2. ElevenLabs API (Deep Male Voice: "Adam")
    eleven_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    eleven_voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM").strip()  # "Adam" deep voice
    if eleven_key:
        try:
            import json
            import urllib.request
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{eleven_voice_id}"
            req_data = json.dumps({
                "text": clean_text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
            }).encode("utf-8")
            req = urllib.request.Request(
                url, data=req_data,
                headers={"xi-api-key": eleven_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
                method="POST"
            )
            def _fetch_eleven():
                with urllib.request.urlopen(req, timeout=15) as resp:
                    output_path.write_bytes(resp.read())
            await asyncio.to_thread(_fetch_eleven)
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info("ElevenLabs male voiceover generated (%s): %s", eleven_voice_id, output_path.name)
                return output_path
        except Exception as exc:
            logger.warning("ElevenLabs API failed: %s", exc)

    # 3. OpenAI TTS API (Deep Narrator Male Voice: "onyx")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            import json
            import urllib.request
            req_data = json.dumps({
                "model": "tts-1-hd",
                "input": clean_text,
                "voice": "onyx",  # Onyx = Deep viral male voice
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.openai.com/v1/audio/speech",
                data=req_data,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST"
            )
            def _fetch_openai():
                with urllib.request.urlopen(req, timeout=15) as resp:
                    output_path.write_bytes(resp.read())
            await asyncio.to_thread(_fetch_openai)
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info("OpenAI TTS-HD onyx male voiceover generated: %s", output_path.name)
                return output_path
        except Exception as exc:
            logger.warning("OpenAI TTS failed: %s", exc)

    # 3.5 Microsoft Edge-TTS Free Neural Male Voice ("en-US-ChristopherNeural")
    try:
        import edge_tts
        communicate = edge_tts.Communicate(clean_text, "en-US-ChristopherNeural")
        await communicate.save(str(output_path))
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info("Edge-TTS neural male voiceover generated: %s", output_path.name)
            return output_path
    except Exception as edge_exc:
        logger.warning("Edge-TTS failed: %s", edge_exc)

    # 4. Google Translate Free TTS Fallback
    try:
        import urllib.parse
        import urllib.request
        free_url = "https://translate.google.com/translate_tts?ie=UTF-8&q=" + urllib.parse.quote(clean_text) + "&tl=en&client=tw-ob"
        free_req = urllib.request.Request(free_url, headers={"User-Agent": "Mozilla/5.0"})
        def _fetch_free_google():
            with urllib.request.urlopen(free_req, timeout=10) as resp:
                output_path.write_bytes(resp.read())
        await asyncio.to_thread(_fetch_free_google)
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info("Google Free TTS voiceover generated: %s (size=%d bytes)", output_path.name, output_path.stat().st_size)
            return output_path
    except Exception as free_exc:
        logger.warning("Google Free TTS failed: %s", free_exc)

    if output_path.exists() and output_path.stat().st_size == 0:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
    logger.warning("No active TTS engine available for voiceover narration.")
    return None


