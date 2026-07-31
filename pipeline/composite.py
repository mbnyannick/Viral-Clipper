"""
Step 7 — Compositing.

Assembles the final 1080×1920 vertical clip for each moment.

Supports two layout modes:
1. "pillarbox" (default): Video scaled to 1080 wide, centered with white top/bottom space.
   Caption sits in the top white space above the video.
2. "face_crop": OpenCV face detection crops 16:9 video to full 9:16 vertical on speaker.
   Caption sits overlaid at lower 68% position.
"""

import asyncio
import logging
import typing
from pathlib import Path

from .crop import detect_crop_offset
from .errors import PipelineError
from .score import Moment
from .subtitle import build_word_subtitle_filter

logger = logging.getLogger(__name__)

CANVAS_W = 720
CANVAS_H = 1280
GAP_PX = 24

_COMPOSITE_CONCURRENCY = 2


async def _probe_video_position(clip_path: Path) -> tuple[int, int]:
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

    scale = min(CANVAS_W / w, CANVAS_H / h)
    scaled_h = int(h * scale)
    video_top_y = (CANVAS_H - scaled_h) // 2

    return video_top_y, scaled_h


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


async def _composite_one(
    clip_path: Path,
    caption_path: Path,
    caption_height: int,
    watermark_path: Path,
    moment: Moment,
    output_dir: Path,
    layout_mode: str = "pillarbox",
    enable_silence_cut: bool = True,
    enable_subtitles: bool = True,
    segments: list[dict] | None = None,
) -> Path:
    out_path = output_dir / f"final_{moment.index:02d}.mp4"
    norm_wm_path = output_dir / f"wm_norm_{moment.index:02d}.png"
    _, wm_w, wm_h = prepare_watermark(watermark_path, norm_wm_path)

    # Position watermark logo centered horizontally, placed just below the video frame right above the username area
    video_top_y, scaled_h = await _probe_video_position(clip_path)
    video_bottom_y = video_top_y + scaled_h

    wm_x = max(10, min((CANVAS_W - wm_w) // 2, CANVAS_W - wm_w - 10))
    if layout_mode == "face_crop":
        wm_y = CANVAS_H - wm_h - 180
    else:
        wm_y = min(video_bottom_y + 10, CANVAS_H - wm_h - 120)
    wm_y = max(10, min(wm_y, CANVAS_H - wm_h - 10))

    if layout_mode == "face_crop":
        crop_filter = await detect_crop_offset(clip_path)
        logger.info(
            "  Compositing clip %02d (face_crop clean full screen, wm_y=%d, speed=1.10x, sub=%s)",
            moment.index, wm_y, enable_subtitles,
        )
        # Build subtitle filter for face_crop layout if enabled
        sub_filter = ""
        punch_in_expr = None
        if enable_subtitles and segments:
            sf, p_expr = build_word_subtitle_filter(segments, moment.start, moment.end, canvas_w=CANVAS_W, wm_y=wm_y, canvas_h=CANVAS_H)
            if sf:
                sub_filter = f",subtitles='{sf}':fontsdir=assets/fonts"
            punch_in_expr = p_expr
            
        if punch_in_expr:
            # 1.2x Zoom Punch-In Crop during high-emotion keywords!
            zoom_crop = crop_filter.replace("crop=ih*9/16:ih:", "crop=ih*9/16*0.833:ih*0.833:")
            zoom_crop = zoom_crop.replace(":0,scale", "+ih*9/16*0.0835:0,scale")
            vf = (
                f"[0:v]{crop_filter},setpts=PTS/1.10[base];"
                f"[0:v]{zoom_crop},setpts=PTS/1.10[zoom];"
                f"[base][zoom]overlay=x=0:y=0:enable='{punch_in_expr}'[cropped];"
                f"[cropped][2:v]overlay={wm_x}:{wm_y}[out2];"
                f"[out2]null{sub_filter}[out]"
            )
        else:
            vf = (
                f"[0:v]{crop_filter},setpts=PTS/1.10[cropped];"
                f"[cropped][2:v]overlay={wm_x}:{wm_y}[out2];"
                f"[out2]null{sub_filter}[out]"
            )
    elif layout_mode == "blurred_frame":
        cap_y = max(15, video_top_y - caption_height - 15)
        logger.info(
            "  Compositing clip %02d (blurred_frame 720x1280, caption_y=%d, wm_y=%d, speed=1.10x)",
            moment.index, cap_y, wm_y,
        )
        sub_filter = ""
        if enable_subtitles and segments:
            sf, _ = build_word_subtitle_filter(segments, moment.start, moment.end, canvas_w=CANVAS_W, wm_y=wm_y, canvas_h=CANVAS_H)
            if sf:
                sub_filter = f",subtitles='{sf}':fontsdir=assets/fonts"
        vf = (
            f"[0:v]scale=108:192:force_original_aspect_ratio=increase,crop=108:192,boxblur=4:1,scale={CANVAS_W}:{CANVAS_H}[bg];"
            f"[0:v]scale={CANVAS_W}:-2[fg];"
            f"[bg][fg]overlay=0:{video_top_y},setpts=PTS/1.10[vbase];"
            f"[vbase][1:v]overlay=0:{cap_y}[v1];"
            f"[v1][2:v]overlay={wm_x}:{wm_y}[out2];"
            f"[out2]null{sub_filter}[out]"
        )
    else:
        # Custom Canvas Background Color (black, red, blue, purple, dark_red)
        bg_color = "black"
        if "red" in layout_mode:
            bg_color = "#800000"  # Deep Crimson Red
        elif "blue" in layout_mode:
            bg_color = "#001f3f"  # Midnight Navy Blue
        elif "purple" in layout_mode:
            bg_color = "#2d004d"  # Dark Velvet Purple
        elif "grey" in layout_mode or "gray" in layout_mode:
            bg_color = "#1a1a1a"  # Dark Charcoal Grey

        cap_y = max(15, video_top_y - caption_height - 15)
        logger.info(
            "  Compositing clip %02d (%s 720x1280, color=%s, caption_y=%d, wm_y=%d, speed=1.10x, sub=%s)",
            moment.index, layout_mode, bg_color, cap_y, wm_y, enable_subtitles,
        )
        sub_filter = ""
        if enable_subtitles and segments:
            sf, _ = build_word_subtitle_filter(segments, moment.start, moment.end, canvas_w=CANVAS_W, wm_y=wm_y, canvas_h=CANVAS_H)
            if sf:
                sub_filter = f",subtitles='{sf}':fontsdir=assets/fonts"
        vf = (
            f"[0:v]scale={CANVAS_W}:-2,pad={CANVAS_W}:{CANVAS_H}:0:{video_top_y}:color={bg_color},setpts=PTS/1.10[vbase];"
            f"[vbase][1:v]overlay=0:{cap_y}[v1];"
            f"[v1][2:v]overlay={wm_x}:{wm_y}[out2];"
            f"[out2]null{sub_filter}[out]"
        )

    inputs = [
        "-i", str(clip_path),
        "-i", str(caption_path),
        "-i", str(norm_wm_path),
    ]

    af_chain = ["[0:a]atempo=1.10,volume=1.5[voice]"]
    amix_inputs = ["[voice]"]
    
    input_count = 3  # We start with 3 inputs (clip, caption, watermark)
    
    # BGM
    bgm_track = getattr(moment, "bgm_track", "none")
    bgm_path = Path(__file__).parent.parent / "assets" / "audio" / "bgm" / f"{bgm_track}.mp3"
    if bgm_track and bgm_track != "none" and bgm_path.exists():
        bgm_idx = input_count
        input_count += 1
        inputs.extend(["-stream_loop", "-1", "-i", str(bgm_path)])
        af_chain.append(f"[{bgm_idx}:a]volume=0.10[bgm]")
        amix_inputs.append("[bgm]")
        
    # SFX
    sfx_events = getattr(moment, "sfx_events", [])
    if not sfx_events:
        # Guarantee highly engaging SFX even if LLM fails to generate them
        sfx_events = [
            {"type": "whoosh", "time_offset": 0.2},
            {"type": "boom", "time_offset": max(2.5, (moment.end - moment.start) * 0.3)} # 30% into the clip
        ]
        
    for event in sfx_events:
        sfx_type = event.get("type")
        sfx_time = event.get("time_offset", 0.0)
        sfx_path = Path(__file__).parent.parent / "assets" / "audio" / "sfx" / f"{sfx_type}.wav"
        if sfx_path.exists():
            sfx_idx = input_count
            input_count += 1
            inputs.extend(["-i", str(sfx_path)])
            delay_ms = int((sfx_time / 1.10) * 1000)
            # Volume increased to 2.5 so it pierces through the 1.5x voice audio
            af_chain.append(f"[{sfx_idx}:a]adelay={delay_ms}|{delay_ms},volume=2.5[sfx{sfx_idx}]")
            amix_inputs.append(f"[sfx{sfx_idx}]")
            
    # Combine audio
    if len(amix_inputs) > 1:
        mix_str = "".join(amix_inputs)
        af_chain.append(f"{mix_str}amix=inputs={len(amix_inputs)}:duration=first:dropout_transition=2[aout]")
        a_map = "[aout]"
    else:
        af_chain.append(f"[voice]anull[aout]")
        a_map = "[aout]"
        
    filter_complex = f"{vf}; " + "; ".join(af_chain)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-map", a_map,
        *_get_v_encoder_args(),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
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
            logger.warning("clip_%02d compositing error: %s", moment.index, stderr.decode(errors='replace')[-300:])
    except Exception as exc:
        err_msg = repr(exc) if not str(exc) else str(exc)
        logger.warning("Clip %02d compositing timeout/error (%s) — using fast fallback copy", moment.index, err_msg)
        try:
            if 'proc' in locals() and proc:
                proc.kill()
                await proc.wait()
        except Exception:
            pass
        # Fallback fast render with strict 9:16 static center crop
        cmd_fallback = [
            "ffmpeg", "-y", "-i", str(clip_path),
            "-vf", "crop=ih*9/16:ih,scale=720:1280",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-c:a", "aac",
            str(out_path),
        ]
        try:
            p2 = await asyncio.create_subprocess_exec(*cmd_fallback, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await asyncio.wait_for(p2.communicate(), timeout=60.0)
        except Exception:
            pass

import sys

_COMPOSITE_SEMAPHORE = asyncio.Semaphore(2)


def _get_v_encoder_args() -> list[str]:
    if sys.platform == "darwin":
        return ["-c:v", "h264_videotoolbox", "-b:v", "4500k"]
    return [
        "-c:v", "libx264",
        "-preset", "superfast",
        "-crf", "20",
        "-threads", "4",
    ]


async def composite_clips(
    clips: list[Path],
    captions: list[tuple[Path, int]],
    watermark_path: Path,
    moments: list[Moment],
    output_dir: Path,
    layout_mode: str = "pillarbox",
    enable_silence_cut: bool = True,
    enable_subtitles: bool = True,
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
                    enable_silence_cut=enable_silence_cut,
                    enable_subtitles=enable_subtitles,
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
    return list(results)
