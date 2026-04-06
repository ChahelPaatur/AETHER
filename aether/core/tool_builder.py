"""
ToolBuilder: reads the ToolDiscovery manifest and constructs working tool
objects for every detected capability. Each tool exposes methods that return
structured dicts with {success, result, error} fields.

Usage:
    from aether.core.tool_discovery import ToolDiscovery
    from aether.core.tool_builder import ToolBuilder

    discovery = ToolDiscovery()
    manifest = discovery.discover()
    builder = ToolBuilder(manifest)
    tools = builder.build_all()
    # tools["system"].get_cpu_percent()
    # tools["camera"].capture_image()
"""
import atexit
import glob as glob_mod
import logging
import os
import shutil
import socket
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

_cam_log = logging.getLogger("Camera")


def _ok(result: Any = None) -> Dict:
    return {"success": True, "result": result, "error": ""}


def _err(msg: str) -> Dict:
    return {"success": False, "result": None, "error": msg}


# ── Raspberry Pi detection ────────────────────────────────────────────

def _is_raspberry_pi() -> bool:
    """Detect if running on a Raspberry Pi."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            if "Raspberry Pi" in f.read():
                return True
    except (FileNotFoundError, OSError, PermissionError):
        pass
    return False


_IS_PI = _is_raspberry_pi()
_ON_PI = _IS_PI  # alias used by other modules


def _extract_image_path(val) -> str:
    if isinstance(val, str) and val: return val
    if isinstance(val, dict):
        for key in ('filepath','image_path','path','image','file'):
            v = val.get(key)
            if isinstance(v, str) and v: return v
            if isinstance(v, dict):
                for k2 in ('filepath','image_path','path'):
                    v2 = v.get(k2)
                    if isinstance(v2, str) and v2: return v2
    return ''

# ── Picamera2 thread lock — fresh instance per capture ────────────────

_picam_lock = threading.Lock()


def _capture_frame_any():
    """Capture a frame from whatever camera is available.

    Returns (frame_ndarray, backend_name_or_error).
    On Pi creates a fresh Picamera2 instance each time (under lock) to
    avoid singleton corruption ('Camera in Running state' / allocator errors).
    Elsewhere tries cv2.
    """
    if _ON_PI:
        with _picam_lock:
            try:
                from picamera2 import Picamera2
                cam = Picamera2()
                config = cam.create_still_configuration(
                    main={"size": (1920, 1080), "format": "RGB888"}
                )
                cam.configure(config)
                cam.start()
                time.sleep(0.3)
                frame = cam.capture_array()
                cam.stop()
                cam.close()
                return frame[:, :, :3], "picamera2"
            except Exception as e:
                return None, f"picamera2 failed: {e}"
    else:
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return frame, "cv2"
            cap.release()
        except Exception:
            pass
        return None, "no camera available"


# ── Camera backend detection ──────────────────────────────────────────

def _detect_camera_backend():
    """Detect available camera backend: 'picamera2' on Pi, 'cv2' otherwise.

    On Raspberry Pi, cv2.VideoCapture(0) causes V4L2 timeouts, so we
    skip cv2 entirely and use picamera2 directly.
    """
    if _IS_PI:
        # Pi: use picamera2 ONLY, never attempt cv2.VideoCapture
        try:
            from picamera2 import Picamera2
            return "picamera2", Picamera2
        except (ImportError, RuntimeError):
            pass
        return None, None

    # Non-Pi: prefer cv2
    try:
        import cv2
        return "cv2", cv2
    except ImportError:
        pass
    try:
        from picamera2 import Picamera2
        return "picamera2", Picamera2
    except (ImportError, RuntimeError):
        pass
    return None, None


# ── CameraTool ────────────────────────────────────────────────────────

class CameraTool:
    """Camera operations via cv2 or picamera2 (auto-detected).

    Uses cv2 when available, falls back to picamera2 on Raspberry Pi.
    Camera is initialized once (singleton) and released on shutdown.
    """

    def __init__(self):
        self._backend_name, self._backend = _detect_camera_backend()
        self._cv2 = None
        self._cap = None          # cv2 VideoCapture
        self._picam = None        # picamera2 instance
        self._picam_started = False
        self._prev_frame = None   # for motion detection (numpy array)

        if self._backend_name == "cv2":
            self._cv2 = self._backend
            _cam_log.info("Using cv2 backend")
        elif self._backend_name == "picamera2":
            _cam_log.info("Using picamera2 backend")
        else:
            _cam_log.warning("No camera backend available (neither cv2 nor picamera2)")

        atexit.register(self.close)

    # ── picamera2 singleton management ────────────────────────────────

    def _ensure_picamera(self):
        """Lazily initialize picamera2 once with RGB888 config."""
        if self._picam is not None and self._picam_started:
            return None  # already running
        try:
            Picamera2 = self._backend
            self._picam = Picamera2()
            config = self._picam.create_still_configuration(
                main={"size": (1920, 1080), "format": "RGB888"}
            )
            self._picam.configure(config)
            self._picam.start()
            time.sleep(0.5)  # warmup
            self._picam_started = True
            return None
        except Exception as e:
            self._picam = None
            self._picam_started = False
            return f"picamera2 init failed: {e}"

    # ── Frame capture (backend-agnostic) ──────────────────────────────

    def _grab_frame(self):
        """Capture a frame as a numpy array. Returns (frame, error_str)."""
        if self._backend_name == "cv2":
            return self._grab_frame_cv2()
        elif self._backend_name == "picamera2":
            return self._grab_frame_picamera2()
        return None, "no camera backend available"

    def _grab_frame_cv2(self):
        if _ON_PI:
            frame, backend = _capture_frame_any()
            if frame is None:
                return None, backend
            return frame, None
        if self._cv2 is None:
            return None, "cv2 not installed"
        if self._cap is None or not self._cap.isOpened():
            self._cap = self._cv2.VideoCapture(0)
            if not self._cap.isOpened():
                self._cap = None
                return None, "camera not available"
        ret, frame = self._cap.read()
        if not ret:
            return None, "failed to read frame"
        return frame, None

    def _grab_frame_picamera2(self):
        # Always use _capture_frame_any() — avoids double Picamera2 conflict
        frame, backend = _capture_frame_any()
        if frame is None:
            return None, backend
        return frame, None

    def _save_frame(self, frame, prefix="frame") -> str:
        """Save a numpy frame to disk as JPEG. Returns filepath.

        Handles RGB→BGR conversion when saving picamera2 frames with cv2,
        since picamera2 returns RGB and cv2.imwrite expects BGR.
        """
        out_dir = os.path.join("logs", "captures")
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        path = os.path.join(out_dir, f"{prefix}_{ts}.jpg")

        try:
            if self._cv2 is not None:
                # picamera2 returns RGB; cv2.imwrite expects BGR
                if self._backend_name == "picamera2" and len(frame.shape) == 3:
                    frame_bgr = frame[:, :, ::-1]
                    self._cv2.imwrite(path, frame_bgr)
                else:
                    self._cv2.imwrite(path, frame)
            else:
                # Use PIL (handles RGB natively)
                from PIL import Image
                img = Image.fromarray(frame)
                img.save(path, "JPEG")
        except Exception:
            # Last resort: save raw numpy data
            np.save(path.replace(".jpg", ".npy"), frame)
            path = path.replace(".jpg", ".npy")
        return path

    # ── Public API ────────────────────────────────────────────────────

    def capture_image(self) -> Dict:
        """Capture a frame and save to logs/captures/frame_TIMESTAMP.jpg."""
        try:
            frame, err = self._grab_frame()
            if err:
                return _err(err)
            path = self._save_frame(frame, "frame")
            h, w = frame.shape[:2]
            _cam_log.info("Captured image: %s", path)
            _cam_log.info("Resolution: %dx%d", w, h)
            return _ok({
                "filepath": path,
                "resolution": f"{w}x{h}",
                "timestamp": datetime.now().isoformat(),
                "backend": self._backend_name,
            })
        except Exception as e:
            return _err(str(e))

    def detect_motion(self) -> Dict:
        """Compare consecutive frames via absdiff, return motion_score 0-1.

        Works with both cv2 and pure numpy (picamera2).
        """
        try:
            scores = []
            prev_gray = None
            for _ in range(5):
                frame, err = self._grab_frame()
                if err:
                    return _err(err)
                # Convert to grayscale
                gray = self._to_grayscale(frame)
                # Simple box blur (numpy-only gaussian approx)
                gray = self._blur_gray(gray)
                if prev_gray is not None:
                    delta = np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)).astype(np.uint8)
                    thresh = (delta > 25).astype(np.uint8) * 255
                    motion_px = np.count_nonzero(thresh)
                    total_px = thresh.shape[0] * thresh.shape[1]
                    scores.append(motion_px / total_px)
                prev_gray = gray
                time.sleep(0.08)

            avg_score = sum(scores) / len(scores) if scores else 0.0
            return _ok({"motion_score": round(avg_score, 4),
                         "frame_scores": [round(s, 4) for s in scores],
                         "backend": self._backend_name})
        except Exception as e:
            return _err(str(e))

    def measure_brightness(self) -> Dict:
        """Return average pixel brightness 0-1."""
        try:
            frame, err = self._grab_frame()
            if err:
                return _err(err)
            gray = self._to_grayscale(frame)
            brightness = float(np.mean(gray)) / 255.0
            return _ok({"brightness": round(brightness, 4),
                         "backend": self._backend_name})
        except Exception as e:
            return _err(str(e))

    def detect_color(self, color_name: str = "red", **kwargs) -> Dict:
        """Return percentage of frame matching named color.

        Uses cv2 HSV when available; falls back to simple RGB thresholding.
        """
        try:
            # Extract image path if provided
            image_path = _extract_image_path(
                kwargs.get("image_path","") or kwargs.get("image","") or
                kwargs.get("filepath","") or kwargs.get("path",""))

            # RGB threshold ranges (simpler, works without cv2)
            rgb_ranges = {
                "red":    {"min": (150, 0, 0), "max": (255, 100, 100)},
                "green":  {"min": (0, 100, 0), "max": (100, 255, 100)},
                "blue":   {"min": (0, 0, 150), "max": (100, 100, 255)},
                "yellow": {"min": (180, 180, 0), "max": (255, 255, 100)},
            }
            if color_name.lower() not in rgb_ranges:
                return _err(f"unknown color '{color_name}', "
                            f"supported: {', '.join(rgb_ranges)}")

            if image_path and os.path.exists(str(image_path)):
                import cv2 as _cv2_load
                frame = _cv2_load.imread(str(image_path))
                if frame is None:
                    return _err(f"could not read image: {image_path}")
                err = None
            else:
                frame, err = self._grab_frame()
            if err:
                return _err(err)

            if self._cv2 is not None:
                # Use cv2 HSV for better accuracy
                hsv_ranges = {
                    "red":    ((0, 120, 70),   (10, 255, 255)),
                    "green":  ((36, 80, 70),   (85, 255, 255)),
                    "blue":   ((86, 120, 70),  (130, 255, 255)),
                    "yellow": ((26, 120, 70),  (35, 255, 255)),
                }
                bounds = hsv_ranges[color_name.lower()]
                hsv = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2HSV)
                mask = self._cv2.inRange(hsv, np.array(bounds[0]), np.array(bounds[1]))
            else:
                # Pure numpy RGB thresholding
                rgb = frame[:, :, :3]  # ensure 3 channels
                bounds = rgb_ranges[color_name.lower()]
                lo = np.array(bounds["min"])
                hi = np.array(bounds["max"])
                mask = np.all((rgb >= lo) & (rgb <= hi), axis=2).astype(np.uint8)

            color_px = np.count_nonzero(mask)
            total_px = mask.shape[0] * mask.shape[1]
            pct = color_px / total_px * 100

            return _ok({"color": color_name,
                         "percentage": round(pct, 2),
                         "detected": pct > 0.5,
                         "backend": self._backend_name})
        except Exception as e:
            return _err(str(e))

    # ── Numpy-only image utilities ────────────────────────────────────

    def _to_grayscale(self, frame) -> np.ndarray:
        """Convert frame to grayscale using cv2 if available, else numpy."""
        if self._cv2 is not None:
            if len(frame.shape) == 3:
                return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
            return frame
        # Numpy: weighted average (ITU-R 601-2 luma)
        if len(frame.shape) == 3:
            return np.dot(frame[:, :, :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return frame

    def _blur_gray(self, gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Simple box blur using numpy (approximates Gaussian blur)."""
        if self._cv2 is not None:
            return self._cv2.GaussianBlur(gray, (21, 21), 0)
        # Numpy box blur via cumulative sum
        k = kernel_size
        pad = k // 2
        padded = np.pad(gray.astype(np.float32), pad, mode='edge')
        # Use uniform_filter equivalent via cumsum
        cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        result = (cumsum[k:, k:] - cumsum[:-k, k:] - cumsum[k:, :-k] + cumsum[:-k, :-k]) / (k * k)
        # Trim to original size
        return result[:gray.shape[0], :gray.shape[1]].astype(np.uint8)

    def close(self):
        """Release camera resources."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None
            self._picam_started = False


# ── YOLOTool ─────────────────────────────────────────────────────────

class YOLOTool:
    """Object detection and scene description.

    When ultralytics is available, uses YOLOv8 for local inference.
    When ultralytics is missing (e.g. Raspberry Pi), falls back to the
    Anthropic vision API (claude-sonnet-4-20250514) for scene description.

    Methods:
        detect(image_path)      — run inference on an image file
        detect_from_camera()    — capture frame, run inference, return detections
        count_objects(class_name) — count detections matching a class name
        describe_scene()        — natural language description of everything visible
    """

    def __init__(self, camera_tool: Optional["CameraTool"] = None):
        self._model = None
        self._ultralytics = None
        self._camera = camera_tool
        self._has_ultralytics = False
        try:
            import ultralytics
            self._ultralytics = ultralytics
            self._has_ultralytics = True
        except ImportError:
            pass

    def _ensure_model(self):
        """Lazy-load the YOLOv8n model on first use."""
        if self._model is not None:
            return None
        if self._ultralytics is None:
            return "ultralytics not installed"
        try:
            from ultralytics import YOLO
            self._model = YOLO("yolov8n.pt")  # auto-downloads ~6MB first run
            return None
        except Exception as e:
            return f"YOLO init failed: {e}"

    def _capture_frame(self):
        """Capture a frame via CameraTool, return (filepath, error).

        On Raspberry Pi, creating a second Picamera2 instance causes
        'Device or resource busy', so we require self._camera to be set.
        """
        if self._camera is not None:
            cap_result = self._camera.capture_image()
        elif _IS_PI:
            return None, ("No shared CameraTool — cannot create a second "
                          "Picamera2 instance on Pi (device busy)")
        else:
            cam = CameraTool()
            cap_result = cam.capture_image()
            cam.close()
        if not cap_result.get("success"):
            return None, cap_result.get("error", "capture failed")
        return cap_result["result"]["filepath"], None

    def detect(self, image_path: str = "", **kwargs) -> Dict:
        """Run YOLOv8n on an image. Returns top 10 detections by confidence.

        Parameters
        ----------
        image_path : str, optional
            Path to an image file.  Falls back to camera capture if empty.

        Each detection: class_name, confidence, bounding_box {x1,y1,x2,y2},
        center {x, y}.
        """
        try:
            # Accept image_path / filepath / image from kwargs
            image_path = (image_path
                          or kwargs.get("filepath", "")
                          or kwargs.get("image_path", "")
                          or kwargs.get("image", ""))
            err = self._ensure_model()
            if err:
                return _err(err)

            if not image_path or not os.path.exists(image_path):
                image_path, cap_err = self._capture_frame()
                if cap_err:
                    return _err(f"no image and capture failed: {cap_err}")

            results = self._model(image_path, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    cx = round((xyxy[0] + xyxy[2]) / 2, 1)
                    cy = round((xyxy[1] + xyxy[3]) / 2, 1)
                    detections.append({
                        "class_name": r.names[cls_id],
                        "confidence": round(conf, 4),
                        "bounding_box": {
                            "x1": round(xyxy[0], 1),
                            "y1": round(xyxy[1], 1),
                            "x2": round(xyxy[2], 1),
                            "y2": round(xyxy[3], 1),
                        },
                        "center": {"x": cx, "y": cy},
                    })

            detections.sort(key=lambda d: d["confidence"], reverse=True)
            top = detections[:10]
            return _ok({
                "image_path": image_path,
                "detections": top,
                "count": len(top),
                "total_raw": len(detections),
            })
        except Exception as e:
            return _err(str(e))

    def detect_from_camera(self, **kwargs) -> Dict:
        """Capture a frame from camera, run YOLO, return detections."""
        try:
            filepath, cap_err = self._capture_frame()
            if cap_err:
                return _err(f"camera capture failed: {cap_err}")
            result = self.detect(filepath)
            # Clean up temp capture
            try:
                os.remove(filepath)
            except OSError:
                pass
            return result
        except Exception as e:
            return _err(str(e))

    def count_objects(self, class_name: str = "", **kwargs) -> Dict:
        """Detect objects from camera and count how many match class_name."""
        try:
            det_result = self.detect_from_camera()
            if not det_result.get("success"):
                return det_result

            detections = det_result["result"]["detections"]
            if not class_name:
                return _ok({"total": len(detections)})

            target = class_name.lower().strip()
            matches = [d for d in detections
                       if d["class_name"].lower() == target]
            confidences = [d["confidence"] for d in matches]
            return _ok({
                "class_name": class_name,
                "count": len(matches),
                "confidences": confidences,
                "total_objects": len(detections),
            })
        except Exception as e:
            return _err(str(e))

    def _describe_via_vision_api(self, image_path: str,
                                query: str = "") -> Dict:
        """Describe a scene using the Anthropic vision API (fallback).

        Used when ultralytics/YOLO is not available (e.g. Raspberry Pi).
        Sends the image to claude-sonnet-4-20250514 with vision capability.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        query : str, optional
            If provided, used as the text prompt instead of the default.
            Used by ``count_fingers`` for targeted finger-counting prompts.
        """
        import base64
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return _err("ANTHROPIC_API_KEY not set — cannot use vision API")
        if not image_path or not os.path.exists(image_path):
            return _err(f"Image not found: {image_path}")

        # Determine media type from extension
        ext = os.path.splitext(image_path)[1].lower()
        media_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_map.get(ext, "image/jpeg")

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        prompt_text = query if query else (
            "Describe what you see in this image in detail. "
            "Include people, objects, hands, fingers, text, "
            "colors, and positions.")

        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }],
        )
        description = message.content[0].text
        return _ok({
            "description": description,
            "backend": "anthropic_vision",
            "model": "claude-sonnet-4-20250514",
            "image_path": image_path,
        })

    def describe_scene(self, image_path="", **kwargs) -> Dict:
        """Describe everything visible using the Anthropic vision API.

        Always uses claude-sonnet-4-20250514 for rich scene understanding
        (people, fingers, objects, text, colours, spatial layout).

        Extracts filepath from string or dict args FIRST — before any
        camera code runs.  On Raspberry Pi the camera is NEVER opened;
        a filepath from a prior ``capture_image`` call is required.

        Parameters
        ----------
        image_path : str or dict
            Path to image, or a dict from ``capture_image`` containing
            ``filepath`` / ``image_path``.
        query : str, optional (via kwargs)
            Custom prompt for the vision API.  When the query mentions
            fingers, an explicit finger-counting prompt is used instead
            of the default scene description.
        """
        # ── 0. Extract path from string or dict FIRST ─────────────
        def _get_path(v):
            if isinstance(v, str) and v:
                return v
            if isinstance(v, dict):
                fp = v.get("filepath") or v.get("image_path") or ""
                if fp:
                    return fp
                r = v.get("result")
                if isinstance(r, dict):
                    return r.get("filepath") or r.get("image_path") or ""
            return ""

        image_path = (_get_path(image_path)
                      or _get_path(kwargs.get("image", ""))
                      or _get_path(kwargs.get("filepath", ""))
                      or _get_path(kwargs.get("image_path", "")))

        # ── 0b. Resolve query / detect finger-counting intent ─────
        query = kwargs.get("query", "")
        _FINGER_KEYWORDS = ("finger", "fingers", "how many fingers",
                            "count fingers", "count_fingers")
        if query and any(kw in query.lower() for kw in _FINGER_KEYWORDS):
            query = ("Count exactly how many fingers are being held up "
                     "in this image. Just respond with a number.")

        # ── 1. If we have a valid path, use vision API directly ───
        if image_path and os.path.exists(str(image_path)):
            return self._describe_via_vision_api(str(image_path),
                                                 query=query)

        # ── 2. On Pi — NEVER open camera, require filepath ────────
        if _ON_PI:
            return _err("describe_scene requires image_path on Pi — "
                        "call capture_image first")

        # ── 3. Non-Pi — capture fresh frame as last resort ────────
        frame, backend = _capture_frame_any()
        if frame is None:
            return _err(f"capture failed: {backend}")
        try:
            import cv2
            cap_dir = os.path.join("logs", "captures")
            os.makedirs(cap_dir, exist_ok=True)
            tmp = os.path.join(
                cap_dir,
                f"describe_{int(time.time() * 1000)}.jpg",
            )
            if backend == "picamera2":
                frame = frame[:, :, ::-1]
            cv2.imwrite(tmp, frame)
        except Exception as e:
            return _err(f"failed to save captured frame: {e}")

        return self._describe_via_vision_api(tmp, query=query)

    def count_fingers(self, image_path="", **kwargs) -> Dict:
        """Count fingers held up in an image using Anthropic vision API.

        Routes through ``describe_scene`` with a targeted finger-counting
        prompt.  YOLO/count_objects cannot detect fingers — this method
        uses the vision API which can.

        Returns {finger_count, raw_response, image_path, backend, model}.
        """
        # ── Extract path exactly like describe_scene ──────────────
        def _get_path(v):
            if isinstance(v, str) and v:
                return v
            if isinstance(v, dict):
                fp = v.get("filepath") or v.get("image_path") or ""
                if fp:
                    return fp
                r = v.get("result")
                if isinstance(r, dict):
                    return r.get("filepath") or r.get("image_path") or ""
            return ""

        image_path = (_get_path(image_path)
                      or _get_path(kwargs.get("image", ""))
                      or _get_path(kwargs.get("filepath", ""))
                      or _get_path(kwargs.get("image_path", "")))

        finger_query = ("Count exactly how many fingers are being held up "
                        "in this image. Just respond with a number.")
        result = self.describe_scene(
            image_path=image_path, query=finger_query)
        if not result.get("success"):
            return result

        # Parse the number from the vision API response
        raw = result["result"]["description"].strip()
        import re
        numbers = re.findall(r'\d+', raw)
        finger_count = int(numbers[0]) if numbers else -1

        return _ok({
            "finger_count": finger_count,
            "raw_response": str(finger_count),
        })


# ── SystemTool ────────────────────────────────────────────────────────

class SystemTool:
    """System metrics. Always built."""

    def __init__(self):
        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            pass

    def get_cpu_percent(self) -> Dict:
        try:
            if self._psutil:
                return _ok({"cpu_percent": self._psutil.cpu_percent(interval=None)})
            return _err("psutil not installed")
        except Exception as e:
            return _err(str(e))

    def get_ram_percent(self) -> Dict:
        try:
            if self._psutil:
                mem = self._psutil.virtual_memory()
                return _ok({"ram_percent": mem.percent,
                             "used_gb": round(mem.used / (1024**3), 1),
                             "total_gb": round(mem.total / (1024**3), 1)})
            return _err("psutil not installed")
        except Exception as e:
            return _err(str(e))

    def get_cpu_temp(self) -> Dict:
        """Read CPU temperature. Linux: /sys/class/thermal. Mac: psutil."""
        try:
            # Linux sysfs
            thermal = "/sys/class/thermal/thermal_zone0/temp"
            if os.path.exists(thermal):
                with open(thermal) as f:
                    temp_c = int(f.read().strip()) / 1000.0
                return _ok({"cpu_temp_c": round(temp_c, 1), "source": "sysfs"})

            # psutil sensors_temperatures (macOS / other)
            if self._psutil:
                fn = getattr(self._psutil, "sensors_temperatures", None)
                if fn:
                    temps = fn()
                    if temps:
                        for name, entries in temps.items():
                            if entries:
                                return _ok({"cpu_temp_c": entries[0].current,
                                             "source": name})

            return _err("no temperature source available")
        except Exception as e:
            return _err(str(e))

    def get_disk_space(self) -> Dict:
        try:
            usage = shutil.disk_usage("/")
            return _ok({
                "free_gb": round(usage.free / (1024**3), 1),
                "total_gb": round(usage.total / (1024**3), 1),
                "used_gb": round(usage.used / (1024**3), 1),
                "percent_used": round(usage.used / usage.total * 100, 1),
            })
        except Exception as e:
            return _err(str(e))

    def get_battery(self) -> Dict:
        try:
            if self._psutil:
                batt = self._psutil.sensors_battery()
                if batt is not None:
                    return _ok({
                        "percent": batt.percent,
                        "plugged_in": batt.power_plugged,
                        "seconds_left": batt.secsleft if batt.secsleft >= 0 else None,
                    })
                return _ok({"percent": None, "detail": "no battery (desktop)"})
            return _err("psutil not installed")
        except Exception as e:
            return _err(str(e))


# ── GPUTool ───────────────────────────────────────────────────────────

class GPUTool:
    """GPU operations via PyTorch. Built when torch + gpu available."""

    def __init__(self):
        self._torch = None
        self._device = None
        try:
            import torch
            self._torch = torch
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        except ImportError:
            pass

    def get_gpu_memory(self) -> Dict:
        """Report GPU memory usage."""
        try:
            torch = self._torch
            if torch is None:
                return _err("torch not installed")

            if self._device and self._device.type == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024**2)
                reserved = torch.cuda.memory_reserved() / (1024**2)
                total = torch.cuda.get_device_properties(0).total_mem / (1024**2)
                return _ok({
                    "device": "cuda",
                    "name": torch.cuda.get_device_name(0),
                    "allocated_mb": round(allocated, 1),
                    "reserved_mb": round(reserved, 1),
                    "total_mb": round(total, 1),
                })

            if self._device and self._device.type == "mps":
                allocated = torch.mps.current_allocated_memory() / (1024**2)
                return _ok({
                    "device": "mps",
                    "name": "Apple Silicon",
                    "allocated_mb": round(allocated, 1),
                })

            return _ok({"device": "cpu", "detail": "no GPU acceleration"})
        except Exception as e:
            return _err(str(e))

    def run_inference(self, data: Optional[Any] = None) -> Dict:
        """Run a test inference: push a numpy array through a linear layer."""
        try:
            torch = self._torch
            if torch is None:
                return _err("torch not installed")
            import numpy as np

            if data is None:
                data = np.random.randn(1, 16).astype(np.float32)
            elif isinstance(data, list):
                data = np.array(data, dtype=np.float32)
                if data.ndim == 1:
                    data = data.reshape(1, -1)

            in_features = data.shape[-1]
            tensor = torch.tensor(data, device=self._device)
            layer = torch.nn.Linear(in_features, 4).to(self._device)

            with torch.no_grad():
                output = layer(tensor)

            result_np = output.cpu().numpy().tolist()
            return _ok({
                "input_shape": list(data.shape),
                "output_shape": list(output.shape),
                "output": result_np,
                "device": str(self._device),
            })
        except Exception as e:
            return _err(str(e))


# ── StorageTool ───────────────────────────────────────────────────────

class StorageTool:
    """File system operations. Always built."""

    def read_file(self, path: str) -> Dict:
        try:
            path = os.path.realpath(path)
            if not os.path.exists(path):
                return _err(f"file not found: {path}")
            with open(path, "r") as f:
                content = f.read()
            return _ok({"path": path, "size_bytes": len(content),
                         "content": content[:5000]})
        except Exception as e:
            return _err(str(e))

    def write_file(self, path: str, content: str) -> Dict:
        try:
            path = os.path.realpath(path)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return _ok({"path": path, "bytes_written": len(content)})
        except Exception as e:
            return _err(str(e))

    def list_directory(self, path: str = ".") -> Dict:
        try:
            path = os.path.realpath(path)
            if not os.path.isdir(path):
                return _err(f"not a directory: {path}")
            entries = []
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                entries.append({
                    "name": name,
                    "is_dir": os.path.isdir(full),
                    "size": os.path.getsize(full) if os.path.isfile(full) else 0,
                })
            return _ok({"path": path, "count": len(entries),
                         "entries": entries[:200]})
        except Exception as e:
            return _err(str(e))

    def search_files(self, pattern: str, root: str = ".") -> Dict:
        try:
            root = os.path.realpath(root)
            matches = glob_mod.glob(os.path.join(root, "**", pattern),
                                    recursive=True)
            return _ok({"pattern": pattern, "root": root,
                         "count": len(matches),
                         "files": matches[:200]})
        except Exception as e:
            return _err(str(e))


# ── NetworkTool ───────────────────────────────────────────────────────

class NetworkTool:
    """Network operations. Built when manifest network.internet is true."""

    def __init__(self):
        self._requests = None
        try:
            import requests
            self._requests = requests
        except ImportError:
            pass

    def web_fetch(self, url: str, timeout: float = 10.0) -> Dict:
        """Fetch a URL and return status + body."""
        try:
            if self._requests:
                resp = self._requests.get(url, timeout=timeout)
                return _ok({
                    "url": url,
                    "status_code": resp.status_code,
                    "content_length": len(resp.text),
                    "body": resp.text[:5000],
                })
            # Fallback to urllib
            from urllib.request import urlopen, Request
            req = Request(url, headers={"User-Agent": "AETHER/3.0"})
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return _ok({
                    "url": url,
                    "status_code": resp.status,
                    "content_length": len(body),
                    "body": body[:5000],
                })
        except Exception as e:
            return _err(str(e))

    def check_connectivity(self) -> Dict:
        """Ping 8.8.8.8 on port 53."""
        try:
            t0 = time.time()
            sock = socket.create_connection(("8.8.8.8", 53), timeout=5)
            latency_ms = (time.time() - t0) * 1000
            sock.close()
            return _ok({"reachable": True, "target": "8.8.8.8:53",
                         "latency_ms": round(latency_ms, 1)})
        except (OSError, socket.timeout) as e:
            return _ok({"reachable": False, "error": str(e)})


# ── MotorTool ─────────────────────────────────────────────────────────

class MotorTool:
    """Unified motor control built from auto-detected motor config.

    Reads configs/auto_detected_motors.json and builds the correct
    implementation for whatever was detected: MAVLink, PCA9685, GPIO, ROS.
    Falls back to simulation logging if nothing detected.
    """

    def __init__(self, motor_config: Optional[Dict] = None):
        self._config = motor_config or self._load_config()
        self._primary = self._config.get("primary") if self._config else None
        self._controller_type = (self._primary or {}).get("controller_type", "simulation")
        self._connection = None
        self._initialized = False

    @staticmethod
    def _load_config() -> Optional[Dict]:
        """Load auto_detected_motors.json from configs/."""
        import json
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            "configs", "auto_detected_motors.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _lazy_init(self):
        """Initialize the hardware connection on first use."""
        if self._initialized or self._primary is None:
            return
        self._initialized = True
        ctype = self._controller_type
        method = self._primary.get("connection_method", "")

        if ctype == "mavlink_fc" and method == "serial":
            self._init_mavlink()
        elif ctype == "pca9685_driver" and method == "i2c":
            self._init_pca9685()
        elif ctype == "gpio_direct_motors" and method == "gpio":
            self._init_gpio()
        elif ctype == "ros_motors":
            self._init_ros()

    def _init_mavlink(self):
        try:
            from pymavlink import mavutil
            port = self._primary["port"]
            baud = self._primary.get("baud", 115200)
            self._connection = mavutil.mavlink_connection(port, baud=baud)
            self._connection.wait_heartbeat(timeout=5)
        except Exception:
            self._connection = None

    def _init_pca9685(self):
        try:
            import smbus2
            self._connection = smbus2.SMBus(1)
            addr = int(self._primary.get("address", "0x40"), 16)
            # Set MODE1 to auto-increment
            self._connection.write_byte_data(addr, 0x00, 0x20)
        except Exception:
            self._connection = None

    def _init_gpio(self):
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self._connection = GPIO
        except (ImportError, RuntimeError):
            try:
                import gpiozero
                self._connection = gpiozero
            except ImportError:
                self._connection = None

    def _init_ros(self):
        try:
            import rospy
            from geometry_msgs.msg import Twist
            self._connection = {"rospy": rospy, "Twist": Twist}
        except ImportError:
            self._connection = None

    def get_motor_info(self) -> Dict:
        """Return detected motor controller info."""
        if self._primary is None:
            return _ok({"controller": "simulation",
                         "detail": "no motor controller detected"})
        return _ok({
            "controller": self._controller_type,
            "connection": self._primary.get("connection_method", "?"),
            "capabilities": self._primary.get("detected_capabilities", []),
            "actions": self._primary.get("suggested_actions", []),
        })

    def motor_forward(self, speed: float = 50.0, duration: float = 1.0) -> Dict:
        """Drive forward at given speed (0-100) for duration seconds."""
        self._lazy_init()
        try:
            if self._controller_type == "mavlink_fc" and self._connection:
                return self._mavlink_velocity(speed / 100.0, 0, 0, duration)
            elif self._controller_type == "pca9685_driver" and self._connection:
                return self._pca9685_drive(speed, duration)
            elif self._controller_type == "gpio_direct_motors" and self._connection:
                return self._gpio_drive(speed, duration)
            elif self._controller_type == "ros_motors" and self._connection:
                return self._ros_drive(speed, duration)
            else:
                return self._sim_log("motor_forward",
                                     {"speed": speed, "duration": duration})
        except Exception as e:
            return _err(str(e))

    def motor_turn(self, direction: str = "left",
                   speed: float = 50.0, duration: float = 0.5) -> Dict:
        """Turn in given direction at speed for duration."""
        self._lazy_init()
        yaw = -speed / 100.0 if direction == "left" else speed / 100.0
        try:
            if self._controller_type == "mavlink_fc" and self._connection:
                return self._mavlink_velocity(0, 0, yaw, duration)
            elif self._controller_type == "pca9685_driver" and self._connection:
                return self._pca9685_turn(direction, speed, duration)
            elif self._controller_type == "gpio_direct_motors" and self._connection:
                return self._gpio_turn(direction, speed, duration)
            elif self._controller_type == "ros_motors" and self._connection:
                return self._ros_turn(direction, speed, duration)
            else:
                return self._sim_log("motor_turn",
                                     {"direction": direction, "speed": speed})
        except Exception as e:
            return _err(str(e))

    def motor_stop(self) -> Dict:
        """Stop all motors immediately."""
        self._lazy_init()
        try:
            if self._controller_type == "mavlink_fc" and self._connection:
                return self._mavlink_velocity(0, 0, 0, 0)
            elif self._controller_type == "pca9685_driver" and self._connection:
                addr = int(self._primary.get("address", "0x40"), 16)
                # Set all channels to 0
                for ch in range(16):
                    self._pca9685_set_pwm(addr, ch, 0)
                return _ok({"stopped": True, "controller": "pca9685"})
            elif self._controller_type == "gpio_direct_motors" and self._connection:
                return _ok({"stopped": True, "controller": "gpio"})
            elif self._controller_type == "ros_motors" and self._connection:
                return self._ros_drive(0, 0)
            else:
                return self._sim_log("motor_stop", {})
        except Exception as e:
            return _err(str(e))

    def arm(self) -> Dict:
        """Arm the flight controller (MAVLink only)."""
        if self._controller_type != "mavlink_fc":
            return _err("arm requires MAVLink flight controller")
        self._lazy_init()
        if not self._connection:
            return _err("MAVLink connection not established")
        try:
            from pymavlink import mavutil
            self._connection.arducopter_arm()
            self._connection.motors_armed_wait()
            return _ok({"armed": True})
        except Exception as e:
            return _err(str(e))

    def disarm(self) -> Dict:
        """Disarm the flight controller (MAVLink only)."""
        if self._controller_type != "mavlink_fc":
            return _err("disarm requires MAVLink flight controller")
        self._lazy_init()
        if not self._connection:
            return _err("MAVLink connection not established")
        try:
            self._connection.arducopter_disarm()
            self._connection.motors_disarmed_wait()
            return _ok({"armed": False})
        except Exception as e:
            return _err(str(e))

    # ── MAVLink implementation ──

    def _mavlink_velocity(self, vx: float, vy: float, yaw_rate: float,
                          duration: float) -> Dict:
        from pymavlink import mavutil
        self._connection.mav.set_position_target_local_ned_send(
            0, self._connection.target_system, self._connection.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000111111000111,  # velocity mask
            0, 0, 0, vx, vy, 0, 0, 0, 0, 0, yaw_rate)
        if duration > 0:
            time.sleep(duration)
            # Send stop
            self._connection.mav.set_position_target_local_ned_send(
                0, self._connection.target_system, self._connection.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000111111000111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return _ok({"vx": vx, "vy": vy, "yaw_rate": yaw_rate,
                     "duration": duration, "controller": "mavlink"})

    # ── PCA9685 implementation ──

    def _pca9685_set_pwm(self, addr: int, channel: int, value: int):
        """Set PWM value (0-4095) on a PCA9685 channel."""
        reg = 0x06 + 4 * channel
        self._connection.write_byte_data(addr, reg, 0)       # ON_L
        self._connection.write_byte_data(addr, reg + 1, 0)   # ON_H
        self._connection.write_byte_data(addr, reg + 2, value & 0xFF)  # OFF_L
        self._connection.write_byte_data(addr, reg + 3, value >> 8)    # OFF_H

    def _pca9685_drive(self, speed: float, duration: float) -> Dict:
        addr = int(self._primary.get("address", "0x40"), 16)
        pwm_val = int(speed / 100.0 * 4095)
        self._pca9685_set_pwm(addr, 0, pwm_val)  # left motor
        self._pca9685_set_pwm(addr, 1, pwm_val)  # right motor
        if duration > 0:
            time.sleep(duration)
            self._pca9685_set_pwm(addr, 0, 0)
            self._pca9685_set_pwm(addr, 1, 0)
        return _ok({"speed": speed, "duration": duration, "controller": "pca9685"})

    def _pca9685_turn(self, direction: str, speed: float,
                      duration: float) -> Dict:
        addr = int(self._primary.get("address", "0x40"), 16)
        pwm_val = int(speed / 100.0 * 4095)
        if direction == "left":
            self._pca9685_set_pwm(addr, 0, 0)
            self._pca9685_set_pwm(addr, 1, pwm_val)
        else:
            self._pca9685_set_pwm(addr, 0, pwm_val)
            self._pca9685_set_pwm(addr, 1, 0)
        if duration > 0:
            time.sleep(duration)
            self._pca9685_set_pwm(addr, 0, 0)
            self._pca9685_set_pwm(addr, 1, 0)
        return _ok({"direction": direction, "speed": speed,
                     "controller": "pca9685"})

    # ── GPIO implementation ──

    def _gpio_drive(self, speed: float, duration: float) -> Dict:
        # Simple simulation-safe GPIO drive
        return _ok({"speed": speed, "duration": duration,
                     "controller": "gpio"})

    def _gpio_turn(self, direction: str, speed: float,
                   duration: float) -> Dict:
        return _ok({"direction": direction, "speed": speed,
                     "controller": "gpio"})

    # ── ROS implementation ──

    def _ros_drive(self, speed: float, duration: float) -> Dict:
        try:
            rospy = self._connection["rospy"]
            Twist = self._connection["Twist"]
            pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
            msg = Twist()
            msg.linear.x = speed / 100.0
            pub.publish(msg)
            if duration > 0:
                time.sleep(duration)
                msg.linear.x = 0
                pub.publish(msg)
            return _ok({"speed": speed, "duration": duration,
                         "controller": "ros"})
        except Exception as e:
            return _err(str(e))

    def _ros_turn(self, direction: str, speed: float,
                  duration: float) -> Dict:
        try:
            rospy = self._connection["rospy"]
            Twist = self._connection["Twist"]
            pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
            msg = Twist()
            msg.angular.z = (speed / 100.0) * (-1 if direction == "right" else 1)
            pub.publish(msg)
            if duration > 0:
                time.sleep(duration)
                msg.angular.z = 0
                pub.publish(msg)
            return _ok({"direction": direction, "speed": speed,
                         "controller": "ros"})
        except Exception as e:
            return _err(str(e))

    # ── Simulation fallback ──

    @staticmethod
    def _sim_log(action: str, params: Dict) -> Dict:
        """Log the command for simulation mode."""
        import json as _json
        log_dir = os.path.join("logs", "motor_commands")
        os.makedirs(log_dir, exist_ok=True)
        entry = {"time": time.time(), "action": action, "params": params}
        log_path = os.path.join(log_dir, "commands.json")
        existing = []
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    existing = _json.load(f)
            except Exception:
                existing = []
        existing.append(entry)
        if len(existing) > 500:
            existing = existing[-500:]
        try:
            with open(log_path, "w") as f:
                _json.dump(existing, f, indent=2)
        except OSError:
            pass
        return _ok({"action": action, "params": params,
                     "controller": "simulation",
                     "detail": "no motor controller — command logged"})


# ── TFLiteTool (lightweight Pi-friendly object detection) ─────────────

_TFLITE_MODEL_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/"
    "tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
)
_TFLITE_MODEL_DIR = os.path.join("models", "tflite")
_TFLITE_MODEL_PATH = os.path.join(_TFLITE_MODEL_DIR, "detect.tflite")
_TFLITE_LABELS_PATH = os.path.join(_TFLITE_MODEL_DIR, "labelmap.txt")

# COCO SSD MobileNet v1 labels (91 classes, index 0 = background)
_COCO_LABELS = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "", "backpack",
    "umbrella", "", "", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "", "dining table", "", "",
    "toilet", "", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class TFLiteTool:
    """Lightweight object detection using TFLite MobileNet-SSD.

    Designed for Raspberry Pi where PyTorch / YOLO are too heavy.
    Uses tflite-runtime for fast CPU inference (~100-200ms per frame).

    Methods:
        detect(image_path) — run MobileNet-SSD on an image, return detections
    """

    def __init__(self, camera_tool: Optional["CameraTool"] = None):
        self._interpreter = None
        self._camera = camera_tool
        self._labels = list(_COCO_LABELS)

    def _ensure_model(self):
        """Download model if missing, then load interpreter."""
        if self._interpreter is not None:
            return None
        # Download model zip if not present
        if not os.path.exists(_TFLITE_MODEL_PATH):
            try:
                os.makedirs(_TFLITE_MODEL_DIR, exist_ok=True)
                import urllib.request
                import zipfile
                zip_path = os.path.join(_TFLITE_MODEL_DIR, "model.zip")
                urllib.request.urlretrieve(_TFLITE_MODEL_URL, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(_TFLITE_MODEL_DIR)
                os.remove(zip_path)
                # The zip contains detect.tflite and labelmap.txt
            except Exception as e:
                return f"model download failed: {e}"
        # Load labels if file exists
        if os.path.exists(_TFLITE_LABELS_PATH):
            try:
                with open(_TFLITE_LABELS_PATH, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        self._labels = lines
            except Exception:
                pass
        # Load interpreter
        try:
            from tflite_runtime.interpreter import Interpreter
            self._interpreter = Interpreter(model_path=_TFLITE_MODEL_PATH)
            self._interpreter.allocate_tensors()
            return None
        except Exception as e:
            return f"TFLite init failed: {e}"

    def detect(self, image_path: str = "", **kwargs) -> Dict:
        """Run MobileNet-SSD on an image. Returns detections with bounding boxes.

        Parameters
        ----------
        image_path : str, optional
            Path to an image file. Falls back to camera capture if empty.
        """
        try:
            image_path = (image_path
                          or kwargs.get("filepath", "")
                          or kwargs.get("image_path", "")
                          or kwargs.get("image", ""))
            # Extract path from dict if needed
            if isinstance(image_path, dict):
                image_path = (image_path.get("filepath")
                              or image_path.get("image_path") or "")
                r = image_path if isinstance(image_path, str) else ""
                if not r and isinstance(image_path, dict):
                    rr = image_path.get("result", {})
                    r = rr.get("filepath", "") if isinstance(rr, dict) else ""
                image_path = r

            err = self._ensure_model()
            if err:
                return _err(err)

            # Capture if no path
            if not image_path or not os.path.exists(str(image_path)):
                if self._camera is not None:
                    cap_result = self._camera.capture_image()
                    if not cap_result.get("success"):
                        return _err(cap_result.get("error", "capture failed"))
                    image_path = cap_result["result"]["filepath"]
                else:
                    return _err("no image_path and no camera available")

            # Load and preprocess image
            from PIL import Image
            img = Image.open(str(image_path)).convert("RGB")
            input_details = self._interpreter.get_input_details()
            output_details = self._interpreter.get_output_details()
            h, w = input_details[0]["shape"][1], input_details[0]["shape"][2]
            img_resized = img.resize((w, h))
            input_data = np.expand_dims(np.array(img_resized, dtype=np.uint8), axis=0)

            # Run inference
            self._interpreter.set_tensor(input_details[0]["index"], input_data)
            self._interpreter.invoke()

            # Parse outputs: boxes, classes, scores, count
            boxes = self._interpreter.get_tensor(output_details[0]["index"])[0]
            classes = self._interpreter.get_tensor(output_details[1]["index"])[0]
            scores = self._interpreter.get_tensor(output_details[2]["index"])[0]
            count = int(self._interpreter.get_tensor(output_details[3]["index"])[0])

            orig_w, orig_h = img.size
            detections = []
            for i in range(min(count, 10)):
                score = float(scores[i])
                if score < 0.4:
                    continue
                cls_id = int(classes[i])
                label = (self._labels[cls_id]
                         if cls_id < len(self._labels) else f"class_{cls_id}")
                ymin, xmin, ymax, xmax = boxes[i]
                detections.append({
                    "class_name": label,
                    "confidence": round(score, 3),
                    "bounding_box": {
                        "x1": int(xmin * orig_w),
                        "y1": int(ymin * orig_h),
                        "x2": int(xmax * orig_w),
                        "y2": int(ymax * orig_h),
                    },
                })

            return _ok({
                "detections": detections,
                "count": len(detections),
                "model": "mobilenet_ssd_v1_quant",
                "image_path": str(image_path),
            })
        except Exception as e:
            return _err(str(e))


# ── OLEDTool ──────────────────────────────────────────────────────────

_OLED_SIM_DIR = os.path.join("logs", "oled_sim")
_OLED_W = 128
_OLED_H = 64

_FACE_DEFS = {
    "neutral":  {"eyes": "open",    "mouth": "flat"},
    "happy":    {"eyes": "open",    "mouth": "smile"},
    "thinking": {"eyes": "squint",  "mouth": "flat"},
    "alert":    {"eyes": "wide",    "mouth": "open"},
    "sleeping": {"eyes": "closed",  "mouth": "flat"},
    "speaking": {"eyes": "open",    "mouth": "open"},
}


class OLEDTool:
    """Controls an SSD1306/SSD1309 128x64 SPI OLED display (with simulation fallback).

    Modes:
      - real: luma.oled SSD1306 driver on SPI (DC=GPIO25, RST=GPIO24, CE0)
      - sim: renders frames to PNG files under logs/oled_sim/ for review
    """

    def __init__(self, spi_port: int = 0, spi_device: int = 0,
                 gpio_dc: int = 25, gpio_rst: int = 24,
                 width: int = _OLED_W, height: int = _OLED_H,
                 force_sim: bool = False):
        self._width = width
        self._height = height
        self._spi_port = spi_port
        self._spi_device = spi_device
        self._gpio_dc = gpio_dc
        self._gpio_rst = gpio_rst
        self._mode = "sim"
        self._device = None
        self._Image = None
        self._ImageDraw = None
        self._ImageFont = None
        self._frame_counter = 0

        # PIL is required for both real and sim modes
        try:
            from PIL import Image, ImageDraw, ImageFont
            self._Image = Image
            self._ImageDraw = ImageDraw
            self._ImageFont = ImageFont
        except ImportError:
            self._Image = None

        if not force_sim and _ON_PI and self._Image is not None:
            try:
                from luma.core.interface.serial import spi
                from luma.oled.device import ssd1306
                serial = spi(port=spi_port, device=spi_device,
                             gpio_DC=gpio_dc, gpio_RST=gpio_rst)
                self._device = ssd1306(serial, width=width, height=height)
                self._mode = "real"
            except Exception:
                self._device = None
                self._mode = "sim"

        if self._mode == "sim":
            try:
                os.makedirs(_OLED_SIM_DIR, exist_ok=True)
            except OSError:
                pass

    @property
    def mode(self) -> str:
        return self._mode

    # ── Internal rendering helpers ─────────────────────────────────

    def _new_image(self):
        return self._Image.new("1", (self._width, self._height), 0)

    def _font(self, size: int = 12):
        try:
            return self._ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (OSError, IOError):
            return self._ImageFont.load_default()

    def _text_size(self, draw, text, font):
        try:
            l, t, r, b = draw.textbbox((0, 0), text, font=font)
            return r - l, b - t
        except AttributeError:
            return draw.textsize(text, font=font)

    def _wrap_text(self, draw, text, font, max_w):
        words = text.split()
        lines: list = []
        current = ""
        for w in words:
            trial = (current + " " + w).strip()
            tw, _ = self._text_size(draw, trial, font)
            if tw <= max_w or not current:
                current = trial
            else:
                lines.append(current)
                current = w
        if current:
            lines.append(current)
        return lines

    def _push(self, img, tag: str = "frame"):
        if self._mode == "real" and self._device is not None:
            try:
                self._device.display(img)
                return _ok({"mode": "real", "tag": tag})
            except Exception as e:
                return _err(f"oled display failed: {e}")
        # sim mode: save PNG (scaled 4x for visibility)
        try:
            self._frame_counter += 1
            fname = f"{tag}_{self._frame_counter:04d}.png"
            path = os.path.join(_OLED_SIM_DIR, fname)
            big = img.resize((self._width * 4, self._height * 4),
                             self._Image.NEAREST)
            big.convert("L").save(path)
            return _ok({"mode": "sim", "path": path, "tag": tag})
        except Exception as e:
            return _err(f"oled sim save failed: {e}")

    # ── Public methods ─────────────────────────────────────────────

    def clear(self) -> Dict:
        if self._Image is None:
            return _err("PIL not available")
        img = self._new_image()
        return self._push(img, tag="clear")

    def clear_oled(self) -> Dict:
        return self.clear()

    def display_text(self, text: str = "", font_size: int = 12,
                     x: int = 0, y: int = 0, clear: bool = True, **kwargs) -> Dict:
        # Parse string-repr dict or extract from dict
        if isinstance(text, str) and text.strip().startswith('{'):
            try:
                import ast
                text = ast.literal_eval(text)
            except:
                pass
        if isinstance(text, dict):
            SENSOR_KEYS = ('cpu_temp_c','cpu_percent','ram_percent','brightness',
                           'finger_count','percentage','temp','temperature')
            found = False
            for key in SENSOR_KEYS:
                if key in text:
                    text = f"{key.replace('_',' ').title()}: {text[key]}"
                    found = True
                    break
            if not found:
                text = (text.get('content') or text.get('body') or
                        text.get('description') or text.get('result') or
                        text.get('summary') or text.get('text') or
                        str(list(text.values())[0]))
        # Strip emoji and non-ASCII for OLED bitmap font
        import re
        text = str(text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'[^\x20-\x7E]', '', text).strip()
        text = text[:120]
        if self._Image is None:
            return _err("PIL not available")
        img = self._new_image() if clear else self._new_image()
        draw = self._ImageDraw.Draw(img)
        font = self._font(font_size)
        max_w = self._width - x
        lines = self._wrap_text(draw, str(text), font, max_w)
        line_h = font_size + 2
        cy = y
        for line in lines:
            if cy + line_h > self._height:
                break
            draw.text((x, cy), line, fill=1, font=font)
            cy += line_h
        return self._push(img, tag="text")

    def display_image(self, image_path: str = "") -> Dict:
        if self._Image is None:
            return _err("PIL not available")
        path = _extract_image_path(image_path)
        if not path or not os.path.exists(path):
            return _err(f"image not found: {path}")
        try:
            src = self._Image.open(path).convert("1")
            src = src.resize((self._width, self._height))
            return self._push(src, tag="image")
        except Exception as e:
            return _err(f"load image failed: {e}")

    def show_animation(self, frames: list = None, fps: int = 10) -> Dict:
        if self._Image is None:
            return _err("PIL not available")
        frames = frames or []
        if not frames:
            return _err("no frames provided")
        delay = 1.0 / max(fps, 1)
        last = None
        for f in frames:
            if isinstance(f, str):
                if not os.path.exists(f):
                    continue
                img = self._Image.open(f).convert("1").resize(
                    (self._width, self._height))
            else:
                img = f.convert("1").resize((self._width, self._height))
            last = self._push(img, tag="anim")
            time.sleep(delay)
        return last or _err("no playable frames")

    # ── Face drawing ───────────────────────────────────────────────

    def _draw_eye_pair(self, draw, state: str):
        """Draw both eyes. Positions: left (35,25), right (93,25)."""
        lx, rx, ey = 35, 93, 25
        r = 10
        pr = 4  # pupil radius

        if state == "closed":
            draw.line((lx - r, ey, lx + r, ey), fill=1, width=3)
            draw.line((rx - r, ey, rx + r, ey), fill=1, width=3)
        elif state == "squint":
            # Smaller ovals
            draw.ellipse((lx - r, ey - 5, lx + r, ey + 5), fill=1)
            draw.ellipse((rx - r, ey - 5, rx + r, ey + 5), fill=1)
            draw.ellipse((lx - pr, ey - 2, lx + pr, ey + 2), fill=0)
            draw.ellipse((rx - pr, ey - 2, rx + pr, ey + 2), fill=0)
        elif state == "wide":
            wr = r + 3
            draw.ellipse((lx - wr, ey - wr, lx + wr, ey + wr), fill=1)
            draw.ellipse((rx - wr, ey - wr, rx + wr, ey + wr), fill=1)
            draw.ellipse((lx - 5, ey - 5, lx + 5, ey + 5), fill=0)
            draw.ellipse((rx - 5, ey - 5, rx + 5, ey + 5), fill=0)
        elif state == "wink_left":
            draw.line((lx - r, ey, lx + r, ey), fill=1, width=3)
            draw.ellipse((rx - r, ey - r, rx + r, ey + r), fill=1)
            draw.ellipse((rx - pr, ey - pr, rx + pr, ey + pr), fill=0)
        elif state == "wink_right":
            draw.ellipse((lx - r, ey - r, lx + r, ey + r), fill=1)
            draw.ellipse((lx - pr, ey - pr, lx + pr, ey + pr), fill=0)
            draw.line((rx - r, ey, rx + r, ey), fill=1, width=3)
        else:  # open
            draw.ellipse((lx - r, ey - r, lx + r, ey + r), fill=1)
            draw.ellipse((rx - r, ey - r, rx + r, ey + r), fill=1)
            draw.ellipse((lx - pr, ey - pr, lx + pr, ey + pr), fill=0)
            draw.ellipse((rx - pr, ey - pr, rx + pr, ey + pr), fill=0)

    def _draw_mouth(self, draw, kind: str):
        my = 52
        if kind == "smile":
            draw.arc((44, my - 8, 84, my + 8), 0, 180, fill=1, width=2)
        elif kind == "open":
            draw.ellipse((54, my - 5, 74, my + 5), outline=1, width=2)
        elif kind == "frown":
            draw.arc((44, my - 8, 84, my + 8), 180, 0, fill=1, width=2)
        else:  # flat
            draw.line((44, my, 84, my), fill=1, width=2)

    def draw_face(self, expression: str = "neutral") -> Dict:
        if self._Image is None:
            return _err("PIL not available")
        spec = _FACE_DEFS.get(expression, _FACE_DEFS["neutral"])
        img = self._new_image()
        draw = self._ImageDraw.Draw(img)
        self._draw_eye_pair(draw, spec["eyes"])
        mouth = spec["mouth"]
        self._draw_mouth(draw, mouth)
        return self._push(img, tag=f"face_{expression}")

    def draw_eyes(self, state: str = "open", blink: bool = False) -> Dict:
        if self._Image is None:
            return _err("PIL not available")
        img = self._new_image()
        draw = self._ImageDraw.Draw(img)
        self._draw_eye_pair(draw, "closed" if blink else state)
        return self._push(img, tag=f"eyes_{state}")

    def animate_blink(self, times: int = 2) -> Dict:
        if self._Image is None:
            return _err("PIL not available")
        last = None
        for _ in range(max(1, int(times))):
            # Open eyes + smile mouth
            img = self._new_image()
            draw = self._ImageDraw.Draw(img)
            self._draw_eye_pair(draw, "open")
            self._draw_mouth(draw, "smile")
            self._push(img, tag="blink_open")
            time.sleep(0.4)
            # Closed eyes + smile mouth
            img = self._new_image()
            draw = self._ImageDraw.Draw(img)
            self._draw_eye_pair(draw, "closed")
            self._draw_mouth(draw, "smile")
            last = self._push(img, tag="blink_closed")
            time.sleep(0.15)
        # End with neutral face
        last = self.draw_face("neutral")
        return last or _ok({"mode": self._mode, "blinks": times})

    def animate_speaking(self, duration: float = 2.0) -> Dict:
        if self._Image is None:
            return _err("PIL not available")
        end = time.time() + max(0.2, float(duration))
        last = None
        while time.time() < end:
            last = self.draw_face("speaking")
            time.sleep(0.15)
            last = self.draw_face("neutral")
            time.sleep(0.15)
        return last or _ok({"mode": self._mode, "duration": duration})

    def scroll_text(self, text: str = "", speed: int = 3) -> Dict:
        if self._Image is None:
            return _err("PIL not available")
        font = self._font(16)
        tmp = self._new_image()
        draw = self._ImageDraw.Draw(tmp)
        tw, th = self._text_size(draw, str(text), font)
        y = (self._height - th) // 2
        step_px = max(1, int(speed))
        last = None
        for offset in range(self._width, -tw - 1, -step_px):
            img = self._new_image()
            d2 = self._ImageDraw.Draw(img)
            d2.text((offset, y), str(text), fill=1, font=font)
            last = self._push(img, tag="scroll")
            time.sleep(0.03)
        return last or _ok({"mode": self._mode})

    def show_value(self, label: str = "", value: Any = "",
                   unit: str = "") -> Dict:
        # Extract clean value from dict or string-repr of dict
        if isinstance(value, str) and value.strip().startswith('{'):
            try:
                import ast
                value = ast.literal_eval(value.strip())
            except:
                pass
        # Also handle string like "{'cpu_temp_c': 45.3, ...}"
        if isinstance(value, str) and 'cpu_temp_c' in value:
            try:
                import re
                m = re.search(r"cpu_temp_c.?\s*:\s*([\d.]+)", value)
                if m: value = float(m.group(1))
            except: pass
        if isinstance(value, dict):
            SKIP = {'image_path','filepath','path','file','raw_response','backend','tag','mode','resolution','timestamp'}
            for key in ('finger_count','count','cpu_temp_c','cpu_percent','ram_percent','brightness','percentage','total','result'):
                if key in value:
                    value = value[key]
                    break
            else:
                for k, v in value.items():
                    if k not in SKIP and isinstance(v, (int, float)):
                        value = v
                        break
                else:
                    value = '?'
        if self._Image is None:
            return _err("PIL not available")
        img = self._new_image()
        draw = self._ImageDraw.Draw(img)
        label_font = self._font(12)
        value_font = self._font(28)
        lw, lh = self._text_size(draw, str(label), label_font)
        draw.text(((self._width - lw) // 2, 2), str(label),
                  fill=1, font=label_font)
        display_val = f"{value}{unit}" if unit else str(value)
        vw, vh = self._text_size(draw, display_val, value_font)
        draw.text(((self._width - vw) // 2,
                   (self._height - vh) // 2 + 6),
                  display_val, fill=1, font=value_font)
        return self._push(img, tag="value")

    def show_startup(self) -> Dict:
        if self._Image is None:
            return _err("PIL not available")
        lx, rx, ey = 35, 93, 25
        # Eyes grow open animation
        for size in (2, 5, 8, 10):
            img = self._new_image()
            draw = self._ImageDraw.Draw(img)
            draw.ellipse((lx - size, ey - size, lx + size, ey + size), fill=1)
            draw.ellipse((rx - size, ey - size, rx + size, ey + size), fill=1)
            self._push(img, tag=f"startup_{size}")
            time.sleep(0.1)
        # Happy face
        self.draw_face("happy")
        time.sleep(0.5)
        # AETHER v3 text
        self.display_text("AETHER v3", clear=True)
        time.sleep(1.0)
        return self.draw_face("neutral")


# ── ToolBuilder ───────────────────────────────────────────────────────

class ToolBuilder:
    """
    Reads the ToolDiscovery manifest and constructs tool objects for every
    detected capability.

    Usage:
        builder = ToolBuilder(manifest)
        tools = builder.build_all()   # dict of name → tool object
    """

    def __init__(self, manifest: Dict):
        self._manifest = manifest

    def build_all(self) -> Dict[str, Any]:
        """Build and return all tools that the manifest supports."""
        tools: Dict[str, Any] = {}
        hw = self._manifest.get("hardware", {})
        sw = self._manifest.get("software", {})
        net = self._manifest.get("network", {})

        # SystemTool — always
        tools["system"] = SystemTool()

        # StorageTool — always
        tools["storage"] = StorageTool()

        # CameraTool — if camera detected
        if hw.get("camera", {}).get("available", False):
            tools["camera"] = CameraTool()

        # GPUTool — if torch installed AND gpu available
        if sw.get("torch", False) and hw.get("gpu", {}).get("available", False):
            tools["gpu"] = GPUTool()

        # NetworkTool — if internet available
        if net.get("internet", False):
            tools["network"] = NetworkTool()

        # YOLOTool — if ultralytics available OR camera available (vision API fallback)
        if sw.get("ultralytics", False) or "camera" in tools:
            cam = tools.get("camera")
            tools["yolo"] = YOLOTool(camera_tool=cam)

        # TFLiteTool — lightweight detection for Pi (no PyTorch needed)
        if sw.get("tflite_runtime", False):
            cam = tools.get("camera")
            tools["tflite"] = TFLiteTool(camera_tool=cam)

        # OLEDTool — if OLED detected OR PIL installed (sim fallback)
        oled_info = hw.get("oled", {})
        oled_available = bool(oled_info.get("available", False))
        if oled_available or sw.get("PIL", False):
            pins = oled_info.get("pins", {})
            tools["oled"] = OLEDTool(
                spi_port=pins.get("port", 0),
                spi_device=pins.get("device", 0),
                gpio_dc=pins.get("DC", 25),
                gpio_rst=pins.get("RST", 24),
                force_sim=not oled_available,
            )

        # MotorTool — always (falls back to simulation logging)
        motor_ctrls = self._manifest.get("motor_controllers", [])
        motor_config = {
            "primary": motor_ctrls[0] if motor_ctrls else None,
            "controllers": motor_ctrls,
        } if motor_ctrls else None
        tools["motor"] = MotorTool(motor_config)

        return tools

    def build_summary(self) -> str:
        """Return a one-line summary of what would be built."""
        tools = self.build_all()
        names = sorted(tools.keys())
        return f"{len(names)} tools: {', '.join(names)}"
