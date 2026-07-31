"""
Step 6.5 — Dynamic AI Face Tracking & Multi-Angle Speaker Alignment.

Uses MediaPipe Face Mesh to perform Active Speaker Detection (ASD) 
based on Mouth Aspect Ratio (MAR) variance. Automatically jump-cuts 
the camera to whoever is actively speaking in a multi-speaker layout!
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)

try:
    import cv2
    import mediapipe as mp
    # Verify solutions API is available (some architectures like ARM64 might lack it)
    _ = mp.solutions.face_mesh
    HAS_MEDIAPIPE = True
except (ImportError, AttributeError) as e:
    HAS_MEDIAPIPE = False
    logger.warning("MediaPipe Face Mesh not available (%s), falling back to static crop.", e)


async def detect_crop_offset(clip_path: Path) -> str:
    """
    Locate active speakers using 3D Face Mesh and return an optimized,
    high-performance FFmpeg crop filter string.
    """
    return await asyncio.get_event_loop().run_in_executor(
        None, _detect_dynamic_crop_sync, clip_path
    )


def _detect_dynamic_crop_sync(clip_path: Path) -> str:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return "crop=ih*9/16:ih:(iw-ow)/2:0,scale=720:1280"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width == 0 or height == 0:
        cap.release()
        return "crop=ih*9/16:ih:(iw-ow)/2:0,scale=720:1280"

    # Target 9:16 crop width for the frame's height
    crop_w = int(height * 9 / 16)
    if crop_w >= width:
        cap.release()
        return "scale=720:1280"

    default_x = max(0, (width - crop_w) // 2)
    step_frames = max(1, int(fps))  # sample 1 frame per second for high-performance detection

    seats = [] # list of dicts: {"center_x": int, "frames": {frame_idx: metric}}

    face_cascade_path = str(Path(__file__).parent.parent / "assets" / "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(face_cascade_path) if Path(face_cascade_path).exists() else None

    if HAS_MEDIAPIPE:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

    import math

    for frame_idx in range(0, total_frames, step_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        scale = 640.0 / w if w > 640 else 1.0
        if scale != 1.0:
            small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small_frame = frame

        face_found = False
        if HAS_MEDIAPIPE:
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                face_found = True
                for face_landmarks in results.multi_face_landmarks:
                    # Nose tip is roughly landmark 1
                    nx = face_landmarks.landmark[1].x
                    real_x = int(nx * w)
                    
                    # Mouth Aspect Ratio (MAR)
                    p13 = face_landmarks.landmark[13] # Upper lip inner
                    p14 = face_landmarks.landmark[14] # Lower lip inner
                    p78 = face_landmarks.landmark[78] # Left mouth corner
                    p308 = face_landmarks.landmark[308] # Right mouth corner
                    
                    vert_dist = math.hypot(p13.x - p14.x, p13.y - p14.y)
                    horz_dist = math.hypot(p78.x - p308.x, p78.y - p308.y)
                    metric = vert_dist / (horz_dist + 1e-6) # Use MAR as metric
                    
                    matched = False
                    for seat in seats:
                        if abs(seat["center_x"] - real_x) < (width * 0.15):
                            seat["center_x"] = int((seat["center_x"] * len(seat["frames"]) + real_x) / (len(seat["frames"]) + 1))
                            seat["frames"][frame_idx] = metric
                            matched = True
                            break
                    if not matched:
                        seats.append({"center_x": real_x, "frames": {frame_idx: metric}})
        
        # Dual-Engine Fallback: If MediaPipe fails to find the face (e.g. face is tiny or turned), instantly fallback to HAAR
        if not face_found and face_cascade is not None:
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (fx, fy, fw, fh) in faces:
                real_x = int((fx + fw/2) / scale)
                metric = (fw * fh) / (scale * scale) # Use face size variance as fallback metric

                matched = False
                for seat in seats:
                    if abs(seat["center_x"] - real_x) < (width * 0.15):
                        seat["center_x"] = int((seat["center_x"] * len(seat["frames"]) + real_x) / (len(seat["frames"]) + 1))
                        seat["frames"][frame_idx] = metric
                        matched = True
                        break
                if not matched:
                    seats.append({"center_x": real_x, "frames": {frame_idx: metric}})

    cap.release()
    if HAS_MEDIAPIPE:
        face_mesh.close()

    if not seats:
        logger.warning("No faces detected in %s, falling back to static center crop.", clip_path.name)
        return f"crop=ih*9/16:ih:{default_x}:0,scale=720:1280"

    # Active Speaker Detection (ASD) via Rolling Window Face Size Variance
    # Window size: 1.5 seconds
    window_frames = int(fps * 1.5)
    points = [] # list of (time_sec, target_x)
    current_active_seat = None

    for start_frame in range(0, total_frames, step_frames):
        end_frame = start_frame + window_frames
        
        best_seat = None
        max_activity = -1.0
        
        for idx, s in enumerate(seats):
            # Extract MAR values in this time window
            sorted_frames = sorted([f for f in s["frames"].keys() if start_frame <= f < end_frame])
            
            # Calculate total mouth movement (sum of absolute differences)
            activity = 0.0
            for i in range(1, len(sorted_frames)):
                activity += abs(s["frames"][sorted_frames[i]] - s["frames"][sorted_frames[i-1]])
                
            # Hysteresis: give a 20% boost to the currently active speaker to prevent rapid flickering camera cuts
            if current_active_seat == idx:
                activity *= 1.2
                
            if activity > max_activity:
                max_activity = activity
                best_seat = idx
                
        if best_seat is not None and max_activity > 0:
            current_active_seat = best_seat
            
        # If no activity (silence), hold the last known active seat
        if current_active_seat is not None:
            sec = start_frame / fps
            target_x = max(0, min(seats[current_active_seat]["center_x"] - crop_w // 2, width - crop_w))
            points.append((sec, target_x))

    if not points:
        return f"crop=ih*9/16:ih:{default_x}:0,scale=720:1280"

    # Simplify points: only generate a cut when the active speaker actually changes
    simplified_points = []
    last_x = None
    for t, x in points:
        if x != last_x:
            simplified_points.append((t, x))
            last_x = x
            
    # Add clip end bound
    duration = total_frames / fps if fps > 0 else 0
    simplified_points.append((duration, simplified_points[-1][1]))
    
    # Generate TV-style FFmpeg Jump-Cut Camera Expressions!
    # We use a flat sum of boolean evaluations to prevent FFmpeg OOM parsing deeply nested IF statements
    expr_parts = []
    for i in range(len(simplified_points) - 1):
        t0, x0 = simplified_points[i]
        t1, x1 = simplified_points[i+1]
        expr_parts.append(f"{x0}*between(t,{t0:.3f},{t1:.3f})")
    
    # Handle time after the last point
    last_t, last_x = simplified_points[-1]
    expr_parts.append(f"{last_x}*gte(t,{last_t:.3f})")

    expr_str = "+".join(expr_parts)

    logger.info("Face tracking for %s: Generated %d dynamic camera jump-cuts!", clip_path.name, len(simplified_points)-1)
    return f"crop=ih*9/16:ih:'{expr_str}':0,scale=720:1280"
