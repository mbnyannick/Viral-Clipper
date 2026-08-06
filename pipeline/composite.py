"""
Step 7 — Compositing.

Assembles the final 1080×1920 vertical clip for each moment.

Supports two layout modes:
1. "pillarbox" / square backgrounds: Video is center-cropped to 1:1 and placed in the vertical 9:16 canvas.
   Caption sits in the top card area while the main video is centered in the square frame.
2. "face_crop": OpenCV face detection crops 16:9 video to full 9:16 vertical on speaker.
   Caption sits overlaid at lower 68% position.
"""

import asyncio
import logging
import shutil
import subprocess
import typing
from pathlib import Path


from .crop import detect_crop_offset
from .errors import PipelineError
from .score import Moment

logger = logging.getLogger(__name__)

import cv2

CANVAS_W = 720
CANVAS_H = 1280
GAP_PX = 36
CAPTION_OVERLAP = 60   # px the caption overlaps INTO the top of the video frame @ 1080p

_COMPOSITE_CONCURRENCY = 2


def _detect_action_motion_peak(clip_path: Path) -> float:
    """
    Scan clip frames with OpenCV to find the exact millisecond where physical motion peaks
    (e.g. sudden double-take, laugh, slap, reaction gesture, jump).
    Returns relative timestamp in seconds from start of clip.
    """
    if not clip_path.exists():
        return 1.5

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return 1.5

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 10:
        cap.release()
        return 1.5

    max_diff = -1.0
    best_frame_idx = int(fps * 1.5)
    prev_gray = None

    # Scan frames sequentially for candidate teaser peak (first 15s max)
    max_scan_frames = min(total_frames, int(fps * 15.0))
    step = max(1, int(fps / 5.0))
    curr_frame_idx = 0

    while curr_frame_idx < max_scan_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if curr_frame_idx % step == 0:
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = float(cv2.absdiff(gray, prev_gray).mean())
                if diff > max_diff:
                    max_diff = diff
                    best_frame_idx = curr_frame_idx
            prev_gray = gray
        curr_frame_idx += 1

    cap.release()
    peak_sec = round(best_frame_idx / fps, 2)
    logger.info("  Action motion peak detected in %s @ %.2fs (diff=%.1f)", clip_path.name, peak_sec, max_diff)
    return peak_sec


async def _probe_video_position(clip_path: Path, apply_4_3_crop: bool = False) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(clip_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()

    try:
        w, h = map(int, stdout.decode().strip().split(","))
    except ValueError:
        w, h = 1920, 1080

    if apply_4_3_crop:
        # After center-cropping 16:9 → 4:3: effective width = h * 4/3
        eff_w = int(h * 4 / 3)
        eff_h = h
    else:
        eff_w, eff_h = w, h

    scale = min(CANVAS_W / eff_w, CANVAS_H / eff_h)
    scaled_h = int(eff_h * scale)
    video_top_y = (CANVAS_H - scaled_h) // 2

    return video_top_y, scaled_h



def _probe_video_dims_filter(src_w: int, src_h: int, canvas_w: int, canvas_h: int) -> tuple[int, int]:
    """Recalculate video_top_y / scaled_h for a 4:3 source placed in canvas."""
    # After 16:9→4:3 crop: new dimensions are src_h*(4/3) x src_h
    new_w = int(src_h * 4 / 3)
    new_h = src_h
    scale = min(canvas_w / new_w, canvas_h / new_h)
    scaled_h = int(new_h * scale)
    video_top_y = (canvas_h - scaled_h) // 2
    return video_top_y, scaled_h


def _get_4_3_crop_filter(src_w: int = 1920, src_h: int = 1080) -> str:
    """Return FFmpeg vf snippet that center-crops source video to 4:3 ratio cleanly."""
    return "crop=ih*4/3:ih:(iw-ih*4/3)/2:0"


def _caption_y(video_top_y: int, caption_height: int) -> int:
    y = video_top_y - GAP_PX - caption_height
    return max(10, y)


from PIL import Image


def prepare_watermark(
    watermark_path: Path,
    output_path: Path,
    max_w: int = 240,
    max_h: int = 80,
) -> tuple[Path, int, int]:
    """
    Auto-crop transparent padding around watermark PNG logo and resize to safe bounding box.
    Returns (output_path, logo_width, logo_height).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not watermark_path.exists():
        img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        img.save(str(output_path))
        return output_path, 1, 1

    try:
        img = Image.open(watermark_path).convert("RGBA")
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)

        img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        img.save(str(output_path))
        return output_path, img.width, img.height
    except Exception as exc:
        logger.warning("Watermark preparation failed for %s: %s", watermark_path, exc)
        img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        img.save(str(output_path))
        return output_path, 1, 1


def _get_1_1_crop_filter() -> str:
    """Return FFmpeg vf snippet that center-crops video footage to clean 1:1 square ratio showing full head & chest."""
    return "crop=ih:ih:(iw-ih)/2:0"


async def _composite_one(
    clip_path: Path,
    caption_path: Path,
    caption_height: int,
    watermark_path: Path,
    moment: Moment,
    output_dir: Path,
    layout_mode: str = "pillarbox",
    enable_subtitles: bool = True,
    enable_silence_cut: bool = True,
    segments: list[dict] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"final_{moment.index:02d}.mp4"

    # Safety guard: Ensure input clip_path and output out_path are never identical files
    if clip_path.resolve() == out_path.resolve():
        out_path = output_dir / f"rendered_{moment.index:02d}.mp4"
    norm_wm_path = output_dir / f"wm_norm_{moment.index:02d}.png"
    _, wm_w, wm_h = prepare_watermark(watermark_path, norm_wm_path)

    is_square = False
    if layout_mode == "face_crop":
        video_top_y = 0
        scaled_h = CANVAS_H
        crop_filter_str = "crop=ih*9/16:ih:(iw-ih*9/16)/2:0"
    else:
        # Standard 4:3 Ratio Baseline for BLUR, BLACK, and pillarbox modes (720x540 centered in 720x1280 vertical canvas)
        scaled_h = int(CANVAS_W * 3 / 4)  # 540px height for 4:3 video
        video_top_y = (CANVAS_H - scaled_h) // 2  # 370px top position
        crop_filter_str = _get_4_3_crop_filter()

    video_bottom_y = video_top_y + scaled_h

    wm_x = max(10, min((CANVAS_W - wm_w) // 2, CANVAS_W - wm_w - 10))
    if layout_mode == "face_crop":
        wm_y = CANVAS_H - wm_h - 180
    else:
        wm_y = min(video_bottom_y + 12, CANVAS_H - wm_h - 10)
    # Clean, natural studio profile (matches the BEFORE look — no harsh contrast, over-sharpening, or artificial saturation)
    COLOR_ENHANCE = "eq=contrast=1.00:brightness=0.00:saturation=1.00"

    # ── 1. AI Voiceover generation ───────────────────────────────────────────

    vo_path = output_dir / f"vo_{moment.index:02d}.mp3"
    vo_file = None
    vo_dur = 0.0
    vo_script = getattr(moment, "voiceover", None)
    if not vo_script or not str(vo_script).strip():
        vo_script = f"Wait until you see how this moment unfolded live!"
    try:
        from .voiceover import generate_voiceover
        vo_file = await generate_voiceover(str(vo_script), vo_path)
        if vo_file and vo_file.exists() and vo_file.stat().st_size > 1000:
            res = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(vo_file)],
                capture_output=True, text=True, check=True, timeout=5
            )
            vo_dur = float(res.stdout.strip())
            logger.info("  Voiceover hook duration: %.2fs (teaser concat + 100%% lip sync offset)", vo_dur)
        elif vo_file and vo_file.exists():
            try:
                vo_file.unlink(missing_ok=True)
            except Exception:
                pass
            vo_file = None
    except Exception as vo_exc:
        logger.warning("Voiceover generation exception: %s", vo_exc)

    # ── 2. Mood-matched BGM selection ─────────────────────────────────────────
    bgm_dir = Path("assets/bgm")
    mood = (getattr(moment, "bgm_track", "hype") or "hype").lower().strip()
    if mood in ("suspense", "dark"):
        mood = "drama"
    elif mood in ("funny", "meme"):
        mood = "comedy"
    elif mood in ("lofi", "sad", "relaxed"):
        mood = "chill"

    mood_dir = bgm_dir / mood
    bgm_tracks = []
    if mood_dir.exists():
        bgm_tracks = list(mood_dir.glob("*.mp3")) + list(mood_dir.glob("*.wav"))

    if not bgm_tracks and bgm_dir.exists():
        bgm_tracks = list(bgm_dir.rglob("*.mp3")) + list(bgm_dir.rglob("*.wav"))

    bgm_file = bgm_tracks[moment.index % len(bgm_tracks)] if bgm_tracks else None
    if not bgm_file:
        try:
            from .pixabay_assets import get_pixabay_asset
            bgm_file = await get_pixabay_asset(category="bgm", query=mood)
        except Exception as px_exc:
            logger.warning("Pixabay BGM fetch fallback: %s", px_exc)

    if bgm_file:
        logger.info("  Using BGM track (%s mood): %s", mood, bgm_file.name)

    # ── 2.5. Word-by-word Subtitle & Aura Keyword Generation ───────────────
    sub_file = None
    if enable_subtitles and segments:
        try:
            from .subtitle import build_word_subtitle_filter
            sub_file, _ = build_word_subtitle_filter(
                segments=segments,
                clip_start=moment.start,
                clip_end=moment.end,
                canvas_w=CANVAS_W,
                canvas_h=CANVAS_H,
                time_offset=vo_dur,
            )
        except Exception as sub_exc:
            logger.warning("Subtitle generation exception: %s", sub_exc)

    aura_filter = None
    aura_w = getattr(moment, "aura_word", "")
    if not aura_w and getattr(moment, "title", ""):
        words = [w.strip(".,!?*#") for w in moment.title.split() if w.isupper() and len(w) > 3]
        if words:
            aura_w = words[0]

    if aura_w:
        try:
            from .subtitle import build_aura_keyword_filter
            aura_filter = build_aura_keyword_filter(
                aura_word=aura_w,
                moment_duration=moment.duration,
                canvas_w=CANVAS_W,
                canvas_h=CANVAS_H,
                appear_at=1.2,
                hold_duration=3.5,
            )
        except Exception as aura_exc:
            logger.warning("Aura keyword filter exception: %s", aura_exc)

    def _build_sub_stage(sf: Path | str | None, en_sub: bool, af: str | None) -> str:
        has_sub = en_sub and sf and Path(sf).exists()
        if has_sub:
            safe_sub = str(sf).replace(":", "\\:").replace("'", "\\'")
            if af:
                return f"[out2]{af}[out_aura];[out_aura]subtitles='{safe_sub}'[out]"
            return f"[out2]subtitles='{safe_sub}'[out]"
        else:
            if af:
                return f"[out2]{af}[out]"
            return "[out2]null[out]"

    if layout_mode == "face_crop":
        crop_filter = f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale={CANVAS_W}:{CANVAS_H}:flags=lanczos"
        logger.info(
            "  Compositing clip %02d (face_crop clean full screen, wm_y=%d)",
            moment.index, wm_y,
        )
        sub_stage = _build_sub_stage(sub_file, enable_subtitles, aura_filter)

        concat_v = f"[0:v]{crop_filter},{COLOR_ENHANCE},setpts=PTS-STARTPTS[vbase];"

        vf = (
            f"{concat_v}"
            f"[vbase][1:v]overlay=0:0[v1];"
            f"[v1][2:v]overlay={wm_x}:{wm_y}[out2];"
            f"{sub_stage}"
        )
    elif "blur" in layout_mode or layout_mode == "blurred_frame":
        logger.info(
            "  Compositing clip %02d (%s 720x1280, crop=%s, wm_y=%d)",
            moment.index, layout_mode, "1:1" if is_square else "4:3", wm_y,
        )
        sub_stage = _build_sub_stage(sub_file, enable_subtitles, aura_filter)

        # Silky smooth HD background blur + Lanczos sharp main video scaling
        concat_v = (
            f"[0:v]scale={CANVAS_W}:{CANVAS_H}:force_original_aspect_ratio=increase,crop={CANVAS_W}:{CANVAS_H},boxblur=25:3[bg];"
            f"[0:v]{crop_filter_str},scale={CANVAS_W}:{scaled_h}:flags=lanczos,{COLOR_ENHANCE}[fg];"
            f"[bg][fg]overlay=0:{video_top_y},setpts=PTS-STARTPTS[vbase];"
        )

        vf = (
            f"{concat_v}"
            f"[vbase][1:v]overlay=0:0[v1];"
            f"[v1][2:v]overlay={wm_x}:{wm_y}[out2];"
            f"{sub_stage}"
        )
    else:
        bg_color = "black"
        if "pink" in layout_mode:
            bg_color = "#ff69b4"
        elif "red" in layout_mode:
            bg_color = "#800000"
        elif "blue" in layout_mode:
            bg_color = "#001f3f"
        elif "purple" in layout_mode:
            bg_color = "#2d004d"
        elif "grey" in layout_mode or "gray" in layout_mode:
            bg_color = "#1a1a1a"

        bg_color_pad = bg_color.replace("#", "0x")
        sub_stage = _build_sub_stage(sub_file, enable_subtitles, aura_filter)

        concat_v = f"[0:v]{crop_filter_str},scale={CANVAS_W}:{scaled_h},{COLOR_ENHANCE},pad={CANVAS_W}:{CANVAS_H}:0:{video_top_y}:color={bg_color_pad},setpts=PTS-STARTPTS[vbase];"

        vf = (
            f"{concat_v}"
            f"[vbase][1:v]overlay=0:0[v1];"
            f"[v1][2:v]overlay={wm_x}:{wm_y}[out2];"
            f"{sub_stage}"
        )

    inputs = [
        "-i", str(clip_path),
        "-i", str(caption_path),
        "-i", str(norm_wm_path),
    ]

    # ── 4. Audio Filtergraph Alignment ────────────────────────────────────────
    a_idx = 3
    vo_input_idx = None
    bgm_input_idx = None
    sfx_input_idx = None

    if vo_file:
        inputs.extend(["-i", str(vo_file)])
        vo_input_idx = a_idx
        a_idx += 1

    if bgm_file:
        inputs.extend(["-i", str(bgm_file)])
        bgm_input_idx = a_idx
        a_idx += 1

    # ── 3.5 Dynamic SFX Selection from Full 691+ Sound Effect Library ─────────
    audio_exts = {".mp3", ".wav", ".m4a", ".mp4", ".aac", ".ogg"}
    sfx_search_paths = [Path("assets/sfx"), Path("500+ Sound Effects ")]
    sfx_files = []
    for sp in sfx_search_paths:
        if sp.exists():
            sfx_files.extend([f for f in sp.rglob("*.*") if f.suffix.lower() in audio_exts])

    sfx_file = None
    if sfx_files:
        sfx_file = sfx_files[moment.index % len(sfx_files)]
        logger.info("  Using viral SFX effect (%d total in library): %s", len(sfx_files), sfx_file.name)

    if sfx_file and sfx_file.exists():
        inputs.extend(["-i", str(sfx_file)])
        sfx_input_idx = a_idx
        a_idx += 1


    # Mute/delay original clip audio during Voiceover lead-in (resampled to 48kHz stereo)
    if vo_dur > 0:
        delay_ms = int(vo_dur * 1000)
        audio_mix_filters = [f"[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=1.3,adelay=delays={delay_ms}|{delay_ms}[orig_a]"]
    else:
        audio_mix_filters = ["[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=1.3[orig_a]"]

    mix_inputs = ["[orig_a]"]

    if vo_input_idx is not None:
        audio_mix_filters.append(f"[{vo_input_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=1.5[vo_a]")
        mix_inputs.append("[vo_a]")

    if bgm_input_idx is not None:
        audio_mix_filters.append(f"[{bgm_input_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.25,aloop=loop=-1:size=2e+09[bgm_a]")
        mix_inputs.append("[bgm_a]")

    if sfx_input_idx is not None:
        sfx_peak_sec = _detect_action_motion_peak(clip_path)
        sfx_delay_ms = int((sfx_peak_sec + vo_dur) * 1000)
        audio_mix_filters.append(f"[{sfx_input_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.6,adelay=delays={sfx_delay_ms}|{sfx_delay_ms}[sfx_a]")
        mix_inputs.append("[sfx_a]")

    if len(mix_inputs) > 1:
        mix_str = "".join(mix_inputs)
        audio_mix_filters.append(f"{mix_str}amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0:normalize=0,volume=2.2,aformat=sample_rates=48000:channel_layouts=stereo[outa]")
    else:
        audio_mix_filters.append("[orig_a]aformat=sample_rates=48000:channel_layouts=stereo[outa]")

    af_chain = ";".join(audio_mix_filters)
    filter_complex = f"{vf};{af_chain}"
    a_map = "[outa]"

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", a_map,
        *_get_v_encoder_args(),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "320k",
        "-ar", "48000",
        str(out_path),
    ]


    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=300.0)
        if proc.returncode != 0:
            err_str = stderr.decode(errors='replace')[-500:]
            logger.error("clip_%02d compositing failed (exit %d): %s", moment.index, proc.returncode, err_str)
            raise RuntimeError(f"FFmpeg compositing error: {err_str}")
    except Exception as exc:
        err_msg = repr(exc) if not str(exc) else str(exc)
        logger.warning("Clip %02d compositing error (%s) — running vertical 9:16 fallback", moment.index, err_msg)
        try:
            if 'proc' in locals() and proc:
                proc.kill()
                await proc.wait()
        except Exception:
            pass

        # Robust vertical 9:16 fallback rendering using a unique temp output path to prevent input/output collision
        fb_out = out_path.with_name(f"fb_{out_path.name}")
        cmd_fallback = [
            "ffmpeg", "-y",
            "-i", str(clip_path),
            "-i", str(caption_path),
            "-filter_complex", f"[0:v]scale=108:192:force_original_aspect_ratio=increase,crop=108:192,boxblur=4:1,scale={CANVAS_W}:{CANVAS_H}[bg];[0:v]{crop_filter_str},scale={CANVAS_W}:{scaled_h}[fg];[bg][fg]overlay=0:{video_top_y}[vbase];[vbase][1:v]overlay=0:0[out]",
            "-map", "[out]",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac",
            str(fb_out),
        ]
        try:
            p2 = await asyncio.create_subprocess_exec(*cmd_fallback, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            _, err2 = await asyncio.wait_for(p2.communicate(), timeout=120.0)
            if p2.returncode == 0 and fb_out.exists():
                shutil.move(fb_out, out_path)
            else:
                logger.error("Fallback render failed: %s", err2.decode(errors='replace')[-300:])
        except Exception as fb_exc:
            logger.error("Fallback render exception: %s", fb_exc)

    return out_path if (out_path.exists() and out_path.stat().st_size > 50000) else clip_path

import sys

_COMPOSITE_SEMAPHORE = asyncio.Semaphore(1)


def _get_v_encoder_args() -> list[str]:
    if sys.platform == "darwin":
        return ["-c:v", "h264_videotoolbox", "-b:v", "12000k"]
    return [
        "-c:v", "libx264",
        "-preset", "superfast",
        "-crf", "21",
        "-maxrate", "12000k",
        "-bufsize", "24000k",
        "-threads", "2",
    ]



async def composite_clips(
    clips: list[Path],
    captions: list[tuple[Path, int]],
    watermark_path: Path,
    moments: list[Moment],
    output_dir: Path,
    layout_mode: str = "pillarbox",
    enable_subtitles: bool = True,
    enable_silence_cut: bool = True,
    segments: list[dict] | None = None,
    on_clip_ready: typing.Callable | None = None,
) -> list[Path]:
    """Composite clips concurrently and support immediate per-clip delivery callbacks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    async def _safe_composite(clip, cap_info, moment):
        cap_path, cap_h = cap_info
        async with _COMPOSITE_SEMAPHORE:
            try:
                res = await _composite_one(
                    clip,
                    cap_path,
                    cap_h,
                    watermark_path,
                    moment,
                    output_dir,
                    layout_mode=layout_mode,
                    enable_subtitles=enable_subtitles,
                    enable_silence_cut=enable_silence_cut,
                    segments=segments,
                )
            except Exception as exc:
                logger.warning("Clip %02d compositing exception (%s) — using source clip fallback", moment.index, exc)
                res = clip

            if on_clip_ready and res and Path(res).exists():
                try:
                    if asyncio.iscoroutinefunction(on_clip_ready):
                        await on_clip_ready(Path(res), moment)
                    else:
                        on_clip_ready(Path(res), moment)
                except Exception as callback_exc:
                    logger.warning("on_clip_ready callback failed for clip %02d: %s", moment.index, callback_exc)

            return res

    results = await asyncio.gather(
        *(_safe_composite(c, cap, m) for c, cap, m in zip(clips, captions, moments))
    )
    return [Path(r) for r in results if r is not None and Path(r).exists()]
