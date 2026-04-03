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

# ── Singleton picamera2 instance — stays open across all captures ─────

_picam_instance = None


def _get_picamera():
    """Return the module-level picamera2 singleton, creating it if needed."""
    global _picam_instance
    if _picam_instance is None:
        from picamera2 import Picamera2
        _picam_instance = Picamera2()
        config = _picam_instance.create_still_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}
        )
        _picam_instance.configure(config)
        _picam_instance.start()
        time.sleep(0.5)
    return _picam_instance


def _capture_frame_any():
    """Capture a frame from whatever camera is available.

    Returns (frame_ndarray, backend_name_or_error).
    On Pi uses the picamera2 singleton; elsewhere tries cv2.
    """
    if _ON_PI:
        try:
            global _picam_instance
            picam = _get_picamera()
            frame = picam.capture_array()
            return frame[:, :, :3], "picamera2"
        except Exception as e:
            # Reset singleton on error so next call retries
            try:
                if _picam_instance:
                    _picam_instance.stop()
                    _picam_instance.close()
            except Exception:
                pass
            _picam_instance = None
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
        err = self._ensure_picamera()
        if err:
            return None, err
        try:
            frame = self._picam.capture_array()
            return frame, None
        except Exception as e:
            return None, f"picamera2 capture failed: {e}"

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

    def detect_color(self, color_name: str = "red") -> Dict:
        """Return percentage of frame matching named color.

        Uses cv2 HSV when available; falls back to simple RGB thresholding.
        """
        try:
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

    def _describe_via_vision_api(self, image_path: str) -> Dict:
        """Describe a scene using the Anthropic vision API (fallback).

        Used when ultralytics/YOLO is not available (e.g. Raspberry Pi).
        Sends the image to claude-sonnet-4-20250514 with vision capability.
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
                        "text": ("Describe what you see in this image in detail. "
                                 "Include people, objects, hands, fingers, text, "
                                 "colors, and positions."),
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

    def describe_scene(self, image_path: str = "", **kwargs) -> Dict:
        """Describe everything visible in a scene.

        When ultralytics is available, uses YOLOv8 for local detection.
        When ultralytics is missing (e.g. Raspberry Pi), falls back to
        the Anthropic vision API (claude-sonnet-4-20250514).

        Parameters
        ----------
        image_path : str, optional
            Path to an image file.  If provided, that image is used;
            otherwise a fresh frame is captured from the camera.
            Also accepted as ``filepath`` or ``image`` in kwargs (the LLM
            planner often passes the output of capture_image this way).
        """
        try:
            # Accept image_path / filepath / image from kwargs
            image_path = (image_path
                          or kwargs.get("filepath", "")
                          or kwargs.get("image_path", "")
                          or kwargs.get("image", ""))

            # ── Fallback: no ultralytics → use Anthropic vision API ──
            if not self._has_ultralytics:
                if not image_path or not os.path.exists(str(image_path)):
                    # Use _capture_frame_any() directly — on Pi this uses
                    # the picamera2 singleton which is proven to work,
                    # avoiding the "device busy" issue with _capture_frame().
                    frame, backend = _capture_frame_any()
                    if frame is None:
                        return _err(f"no ultralytics and capture failed: {backend}")
                    try:
                        import cv2
                        cap_dir = os.path.join("logs", "captures")
                        os.makedirs(cap_dir, exist_ok=True)
                        image_path = os.path.join(
                            cap_dir,
                            f"describe_{int(time.time() * 1000)}.jpg",
                        )
                        cv2.imwrite(image_path, frame)
                    except Exception as e:
                        return _err(f"failed to save captured frame: {e}")
                return self._describe_via_vision_api(image_path)

            # ── Primary: YOLOv8 local detection ──────────────────────
            if image_path and os.path.exists(image_path):
                det_result = self.detect(image_path)
            else:
                det_result = self.detect_from_camera()
            if not det_result.get("success"):
                return det_result

            detections = det_result["result"]["detections"]
            if not detections:
                return _ok({"description": "I cannot see any recognizable objects.",
                            "detections": []})

            # Group by class name
            groups: Dict[str, list] = {}
            for d in detections:
                name = d["class_name"]
                groups.setdefault(name, []).append(d["confidence"])

            # Build description string
            parts = []
            for name, confs in sorted(groups.items(),
                                       key=lambda x: max(x[1]), reverse=True):
                count = len(confs)
                conf_strs = ", ".join(f"{c*100:.0f}%" for c in sorted(confs, reverse=True))
                if count == 1:
                    parts.append(f"1 {name} ({conf_strs} confident)")
                else:
                    parts.append(f"{count} {name}s ({conf_strs})")

            description = "I can see: " + ", ".join(parts)
            return _ok({
                "description": description,
                "backend": "yolov8",
                "detections": detections,
                "object_count": len(detections),
                "unique_classes": len(groups),
            })
        except Exception as e:
            return _err(str(e))


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
