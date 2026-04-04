"""
NavigationEngine: hardware-agnostic navigation with three capability levels.

Level 1 — Camera only (no motors): visual scanning, object tracking, distance
           estimation, surroundings reporting.
Level 2 — Camera + GPIO motors: color navigation, obstacle avoidance, object
           following, search patterns, marker docking.
Level 3 — Full MAVLink flight controller: takeoff, waypoint navigation, visual
           navigation, RTL, emergency landing.

Detects which level is available from the ToolDiscovery manifest and exposes
only valid methods. The Planner queries available_actions() to avoid planning
impossible actions.

Also provides a unified observe() returning the same 15-dim observation vector
as SimulationEnvironment, with missing sensor slots filled by safe defaults.

Observation vector mapping (same as environment.py / real_perception.py):
  [0]  battery_level       [1]  solar_power        [2]  bus_voltage
  [3]  imu_x               [4]  imu_y              [5]  imu_z
  [6]  attitude_error       [7]  temperature_panel  [8]  temperature_core
  [9]  obstacle_dist_front [10] obstacle_dist_left [11] obstacle_dist_right
  [12] target_dist         [13] target_bearing     [14] mission_progress
"""
import json
import math
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

OBS_DIM = 15

# ── Logging ───────────────────────────────────────────────────────────

_TOOL_LOG_PATH = os.path.join("logs", "tool_calls.json")


def _log_call(action: str, params: Dict, result) -> None:
    """Append every navigation call to logs/tool_calls.json."""
    if isinstance(result, dict):
        success = result.get("success", False)
        error = result.get("error", "")
    else:
        # String or other return (e.g. report_surroundings)
        success = not (isinstance(result, str) and result.startswith("Error"))
        error = ""
    entry = {
        "timestamp": time.time(),
        "action": action,
        "params": {k: str(v)[:200] for k, v in params.items()},
        "success": success,
        "error": error,
    }
    os.makedirs(os.path.dirname(_TOOL_LOG_PATH) or ".", exist_ok=True)
    existing = []
    if os.path.exists(_TOOL_LOG_PATH):
        try:
            with open(_TOOL_LOG_PATH) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append(entry)
    # Keep last 1000 entries
    if len(existing) > 1000:
        existing = existing[-1000:]
    try:
        with open(_TOOL_LOG_PATH, "w") as f:
            json.dump(existing, f, indent=2)
    except OSError:
        pass


def _ok(data: Dict) -> Dict:
    data["success"] = True
    data["error"] = ""
    return data


def _err(msg: str) -> Dict:
    return {"success": False, "error": msg}


# ── Safe imports ──────────────────────────────────────────────────────

import atexit
import logging

_nav_cam_log = logging.getLogger("Camera")


def _try_import(name: str):
    try:
        return __import__(name)
    except ImportError:
        return None


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


def _detect_nav_camera_backend():
    """Detect camera backend for navigation.

    On Raspberry Pi, cv2.VideoCapture(0) causes V4L2 timeouts, so we
    skip cv2 entirely and use picamera2 directly.
    """
    if _IS_PI:
        try:
            from picamera2 import Picamera2
            return "picamera2", Picamera2
        except (ImportError, RuntimeError):
            pass
        return None, None

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


# ── Level 1: Camera-only navigation ──────────────────────────────────

class _CameraNav:
    """Vision-based observation and analysis. No motor output.

    Supports both cv2 and picamera2 backends. Automatically detects which
    is available and uses it. Falls back to numpy-only image analysis when
    cv2 is not installed.
    """

    def __init__(self):
        self._backend_name, self._backend_mod = _detect_nav_camera_backend()
        self._cv2 = None
        self._cap = None           # cv2 VideoCapture
        self._picam = None         # picamera2 instance
        self._picam_started = False
        self._prev_gray = None

        if self._backend_name == "cv2":
            self._cv2 = self._backend_mod
            _nav_cam_log.info("Using cv2 backend")
        elif self._backend_name == "picamera2":
            _nav_cam_log.info("Using picamera2 backend")
        else:
            _nav_cam_log.warning("No camera backend available")

        atexit.register(self.close)

    def _ensure_picamera(self) -> Optional[str]:
        """Lazily initialize picamera2 singleton with RGB888 config."""
        if self._picam is not None and self._picam_started:
            return None
        try:
            Picamera2 = self._backend_mod
            self._picam = Picamera2()
            config = self._picam.create_still_configuration(
                main={"size": (1920, 1080), "format": "RGB888"}
            )
            self._picam.configure(config)
            self._picam.start()
            time.sleep(0.5)
            self._picam_started = True
            return None
        except Exception as e:
            self._picam = None
            self._picam_started = False
            return f"picamera2 init failed: {e}"

    def _ensure_camera(self) -> Optional[str]:
        if self._backend_name == "cv2":
            if _IS_PI:
                # On Pi, use the shared picamera2 singleton
                from aether.core.tool_builder import _capture_frame_any
                frame, backend = _capture_frame_any()
                if frame is None:
                    return backend
                self._last_pi_frame = frame
                return None
            if self._cv2 is None:
                return "cv2 not installed"
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._cv2.VideoCapture(0)
                if not self._cap.isOpened():
                    self._cap = None
                    return "camera not available"
            return None
        elif self._backend_name == "picamera2":
            return self._ensure_picamera()
        return "no camera backend available"

    def _grab_frame(self):
        if self._backend_name == "cv2":
            err = self._ensure_camera()
            if err:
                return None, err
            ret, frame = self._cap.read()
            if not ret:
                return None, "failed to read frame"
            return frame, None
        elif self._backend_name == "picamera2":
            err = self._ensure_picamera()
            if err:
                return None, err
            try:
                frame = self._picam.capture_array()
                return frame, None
            except Exception as e:
                return None, f"picamera2 capture failed: {e}"
        return None, "no camera backend available"

    def _to_grayscale(self, frame) -> np.ndarray:
        """Convert to grayscale using cv2 or numpy."""
        if self._cv2 is not None and len(frame.shape) == 3:
            return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        if len(frame.shape) == 3:
            return np.dot(frame[:, :, :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        return frame

    def _save_frame(self, frame, prefix="scan") -> str:
        """Save frame to disk as JPEG.

        Handles RGB→BGR conversion for picamera2 frames saved via cv2.
        """
        out_dir = os.path.join("logs", "captures")
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        filepath = os.path.join(out_dir, f"{prefix}_{ts}.jpg")
        try:
            if self._cv2 is not None:
                if self._backend_name == "picamera2" and len(frame.shape) == 3:
                    frame_bgr = frame[:, :, ::-1]
                    self._cv2.imwrite(filepath, frame_bgr)
                else:
                    self._cv2.imwrite(filepath, frame)
            else:
                from PIL import Image
                img = Image.fromarray(frame)
                img.save(filepath, "JPEG")
        except Exception:
            np.save(filepath.replace(".jpg", ".npy"), frame)
            filepath = filepath.replace(".jpg", ".npy")
        _nav_cam_log.info("Captured image: %s", filepath)
        return filepath

    def visual_scan(self) -> Dict:
        """Capture a frame, analyze it, and return structured metrics.

        Returns dict with: filepath, resolution, brightness, edge_density,
        dominant_colors, contour_count, motion_score, timestamp.
        Works with both cv2 and pure numpy (picamera2 fallback).
        """
        frame, err = self._grab_frame()
        if err:
            return _err(err)

        h, w = frame.shape[:2]
        ts = int(time.time() * 1000)
        filepath = self._save_frame(frame, "scan")

        gray = self._to_grayscale(frame)
        brightness = round(float(np.mean(gray)) / 255.0, 4)
        total_pixels = h * w

        cv2 = self._cv2

        if cv2 is not None:
            # cv2 path: full Canny + contour + k-means analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_count = int(np.count_nonzero(edges))
            edge_density = round(edge_count / total_pixels, 4)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            min_area = total_pixels * 0.002
            significant = [c for c in contours if cv2.contourArea(c) > min_area]
            contour_count = len(significant)

            # Dominant colors via k-means
            dominant_colors = []
            try:
                pixels = frame.reshape(-1, 3).astype(np.float32)
                if len(pixels) > 50000:
                    indices = np.random.choice(len(pixels), 50000, replace=False)
                    pixels = pixels[indices]
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            10, 1.0)
                _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 3,
                                                cv2.KMEANS_PP_CENTERS)
                counts = np.bincount(labels.flatten())
                order = np.argsort(-counts)
                for idx in order:
                    bgr = centers[idx]
                    pct = round(float(counts[idx]) / len(labels) * 100, 1)
                    dominant_colors.append({
                        "rgb": [int(bgr[2]), int(bgr[1]), int(bgr[0])],
                        "percentage": pct,
                    })
            except Exception:
                dominant_colors = []

            gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        else:
            # Pure numpy path: simple gradient-based edge detection
            gy = np.abs(np.diff(gray.astype(np.int16), axis=0))
            gx = np.abs(np.diff(gray.astype(np.int16), axis=1))
            edge_map = np.zeros_like(gray)
            edge_map[:-1, :] += gy.astype(np.uint8)
            edge_map[:, :-1] += gx.astype(np.uint8)
            edge_count = int(np.count_nonzero(edge_map > 30))
            edge_density = round(edge_count / total_pixels, 4)

            # Estimate contour count from connected regions
            contour_count = max(1, edge_count // 500)

            # Simple dominant color via mean RGB
            dominant_colors = []
            if len(frame.shape) == 3:
                mean_rgb = np.mean(frame[:, :, :3].reshape(-1, 3), axis=0)
                dominant_colors.append({
                    "rgb": [int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2])],
                    "percentage": 100.0,
                })

            # Simple box blur for motion
            k = 5
            pad = k // 2
            padded = np.pad(gray.astype(np.float32), pad, mode='edge')
            cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
            gray_blur = (cumsum[k:, k:] - cumsum[:-k, k:] - cumsum[k:, :-k] + cumsum[:-k, :-k]) / (k * k)
            gray_blur = gray_blur[:gray.shape[0], :gray.shape[1]].astype(np.uint8)

        # Motion score — compare to previous frame
        motion_score = 0.0
        if self._prev_gray is not None:
            delta = np.abs(gray_blur.astype(np.int16) - self._prev_gray.astype(np.int16)).astype(np.uint8)
            motion_pixels = np.count_nonzero(delta > 25)
            motion_score = round(motion_pixels / total_pixels, 4)
        self._prev_gray = gray_blur

        # Dominant color as string
        if dominant_colors:
            top_rgb = dominant_colors[0]["rgb"]
            dominant_color = self._rgb_to_name(top_rgb)
        else:
            dominant_color = "unknown"

        _nav_cam_log.info("Resolution: %dx%d", w, h)

        return _ok({
            "filepath": filepath,
            "resolution": f"{w}x{h}",
            "brightness": brightness,
            "edge_count": edge_count,
            "edge_density": edge_density,
            "dominant_colors": dominant_colors,
            "dominant_color": dominant_color,
            "contour_count": contour_count,
            "motion_score": motion_score,
            "timestamp": ts,
            "backend": self._backend_name,
        })

    @staticmethod
    def _rgb_to_name(rgb: list) -> str:
        """Convert an RGB triplet to a human-readable color name."""
        r, g, b = rgb
        if max(r, g, b) - min(r, g, b) < 30:
            avg = (r + g + b) / 3
            if avg < 60:
                return "black"
            elif avg < 180:
                return "grey"
            else:
                return "white"
        if r > g and r > b:
            return "red" if r > 180 else "dark red"
        if g > r and g > b:
            return "green" if g > 180 else "dark green"
        if b > r and b > g:
            return "blue" if b > 180 else "dark blue"
        if r > 150 and g > 150 and b < 100:
            return "yellow"
        return "mixed"

    def track_object(self, color_or_shape: str = "red") -> Dict:
        """Track a colored object across frames, report position.

        Uses cv2 HSV when available; falls back to simple RGB thresholding.
        """
        frame, err = self._grab_frame()
        if err:
            return _err(err)

        cv2 = self._cv2
        if cv2 is not None:
            color_ranges = _get_color_hsv_range(color_or_shape)
            if color_ranges is None:
                return _err(f"unknown color: {color_or_shape}")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, color_ranges[0], color_ranges[1])
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return _ok({"found": False, "color": color_or_shape})
            biggest = max(contours, key=cv2.contourArea)
            M = cv2.moments(biggest)
            if M["m00"] == 0:
                return _ok({"found": False, "color": color_or_shape})
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            area = cv2.contourArea(biggest)
        else:
            # Pure numpy RGB color tracking
            mask = self._numpy_color_mask(frame, color_or_shape)
            if mask is None:
                return _err(f"unknown color: {color_or_shape}")
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                return _ok({"found": False, "color": color_or_shape})
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            area = float(len(ys))

        frame_area = frame.shape[0] * frame.shape[1]
        return _ok({
            "found": True,
            "color": color_or_shape,
            "center_x": round(cx / frame.shape[1], 4),
            "center_y": round(cy / frame.shape[0], 4),
            "area_fraction": round(area / frame_area, 6),
            "pixel_area": int(area),
            "estimated_distance": self.estimate_distance(area)["distance_m"],
        })

    def _numpy_color_mask(self, frame, color_name: str):
        """Create a binary mask for a named color using numpy RGB ranges."""
        rgb_ranges = {
            "red":    {"min": (150, 0, 0), "max": (255, 100, 100)},
            "green":  {"min": (0, 100, 0), "max": (100, 255, 100)},
            "blue":   {"min": (0, 0, 150), "max": (100, 100, 255)},
            "yellow": {"min": (180, 180, 0), "max": (255, 255, 100)},
        }
        bounds = rgb_ranges.get(color_name.lower())
        if bounds is None:
            return None
        rgb = frame[:, :, :3]
        lo = np.array(bounds["min"])
        hi = np.array(bounds["max"])
        return np.all((rgb >= lo) & (rgb <= hi), axis=2).astype(np.uint8)

    def estimate_distance(self, object_pixels: float) -> Dict:
        """Estimate distance from pixel area using inverse-square heuristic.

        Assumes a reference object of ~0.3m at ~5000px area.
        """
        if object_pixels <= 0:
            return _ok({"distance_m": None, "confidence": 0.0})
        # Reference calibration: 0.3m object = ~5000px at 1m distance
        ref_area = 5000.0
        ref_dist = 1.0
        distance = ref_dist * math.sqrt(ref_area / object_pixels)
        confidence = min(1.0, object_pixels / 500.0)  # low area = low confidence
        return _ok({"distance_m": round(distance, 3),
                     "confidence": round(confidence, 3)})

    def detect_objects(self) -> Dict:
        """Contour-based shape detection on current frame.

        Uses cv2 contours when available; falls back to numpy gradient regions.
        """
        frame, err = self._grab_frame()
        if err:
            return _err(err)

        cv2 = self._cv2
        gray = self._to_grayscale(frame)
        objects = []
        min_area = frame.shape[0] * frame.shape[1] * 0.003

        if cv2 is not None:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                vertices = len(approx)
                shape = _classify_shape(vertices, w, h)
                objects.append({
                    "shape": shape, "vertices": vertices,
                    "x": x, "y": y, "w": w, "h": h,
                    "area": int(area),
                    "center_x": round((x + w / 2) / frame.shape[1], 4),
                    "center_y": round((y + h / 2) / frame.shape[0], 4),
                })
        else:
            # Numpy fallback: simple gradient-based region detection
            gy = np.abs(np.diff(gray.astype(np.int16), axis=0))
            gx = np.abs(np.diff(gray.astype(np.int16), axis=1))
            edge_map = np.zeros_like(gray, dtype=np.uint8)
            edge_map[:-1, :] += np.minimum(gy, 255).astype(np.uint8)
            edge_map[:, :-1] += np.minimum(gx, 255).astype(np.uint8)
            # Threshold to get "bright" regions
            bright_mask = (gray > np.mean(gray) + 30).astype(np.uint8)
            bright_pixels = np.count_nonzero(bright_mask)
            if bright_pixels > min_area:
                ys, xs = np.where(bright_mask > 0)
                if len(ys) > 0:
                    objects.append({
                        "shape": "region", "vertices": 0,
                        "x": int(np.min(xs)), "y": int(np.min(ys)),
                        "w": int(np.max(xs) - np.min(xs)),
                        "h": int(np.max(ys) - np.min(ys)),
                        "area": int(bright_pixels),
                        "center_x": round(float(np.mean(xs)) / frame.shape[1], 4),
                        "center_y": round(float(np.mean(ys)) / frame.shape[0], 4),
                    })

        objects.sort(key=lambda o: o["area"], reverse=True)
        return _ok({"objects": objects[:30],
                     "total_detected": len(objects)})

    def measure_brightness(self) -> Dict:
        """Average frame brightness 0-1."""
        frame, err = self._grab_frame()
        if err:
            return _err(err)
        gray = self._to_grayscale(frame)
        brightness = float(np.mean(gray)) / 255.0
        return _ok({"brightness": round(brightness, 4)})

    def detect_color(self, color_name: str = "red") -> Dict:
        """Detect percentage of frame occupied by a named color."""
        frame, err = self._grab_frame()
        if err:
            return _err(err)

        cv2 = self._cv2
        if cv2 is not None:
            color_ranges = _get_color_hsv_range(color_name)
            if color_ranges is None:
                return _err(f"unknown color: {color_name}")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, color_ranges[0], color_ranges[1])
        else:
            mask = self._numpy_color_mask(frame, color_name)
            if mask is None:
                return _err(f"unknown color: {color_name}")

        color_pixels = np.count_nonzero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        percentage = color_pixels / total_pixels

        return _ok({"color": color_name,
                     "percentage": round(percentage * 100, 2),
                     "detected": percentage > 0.005})

    def capture_image(self) -> Dict:
        """Capture a single frame and save to disk."""
        frame, err = self._grab_frame()
        if err:
            return _err(err)

        filepath = self._save_frame(frame, "aether_cap")
        h, w = frame.shape[:2]
        _nav_cam_log.info("Resolution: %dx%d", w, h)
        return _ok({"filepath": filepath,
                     "resolution": f"{w}x{h}",
                     "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "backend": self._backend_name})

    def get_motion_bearing(self) -> Tuple[bool, float]:
        """Single-frame motion check. Returns (motion_detected, bearing 0-1).

        Works with both cv2 and pure numpy.
        """
        frame, err = self._grab_frame()
        if err:
            return False, 0.5

        gray = self._to_grayscale(frame)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False, 0.5

        delta = np.abs(gray.astype(np.int16) - self._prev_gray.astype(np.int16)).astype(np.uint8)
        thresh = (delta > 25).astype(np.uint8)
        self._prev_gray = gray

        min_pixels = frame.shape[0] * frame.shape[1] * 0.005
        motion_pixels = np.count_nonzero(thresh)
        if motion_pixels < min_pixels:
            return False, 0.5

        # Find center of motion
        ys, xs = np.where(thresh > 0)
        if len(xs) > 0:
            cx = float(np.mean(xs))
            return True, cx / frame.shape[1]
        return True, 0.5

    def report_surroundings(self, scan_data=None, path=None, **kwargs) -> str:
        """Analyze a camera image and return a natural language description.

        Accepts either:
          - scan_data: dict from visual_scan() (must have 'filepath'), or
                       a string filepath directly
          - path: string filepath to a JPEG/PNG image
          - neither: captures a new frame via camera

        Works with both cv2 and pure numpy (picamera2 fallback).
        Computes brightness, edge count, and contour count, then calls the
        Anthropic API for a description. Falls back to a locally composed
        description.

        Returns the description string directly (not wrapped in a dict).
        """
        if self._backend_name is None:
            return "Error: no camera backend available"

        # ── Resolve filepath from whichever input was provided ──
        filepath = None
        if path:
            filepath = path
        elif scan_data and isinstance(scan_data, dict):
            filepath = scan_data.get("filepath")
        elif scan_data and isinstance(scan_data, str):
            filepath = scan_data

        # If no image available, capture one directly
        if not filepath or not os.path.exists(filepath):
            cap_result = self.capture_image()
            if not cap_result.get("success"):
                return f"Error: could not capture image — {cap_result.get('error', 'unknown')}"
            filepath = cap_result.get("filepath", "")

        # ── Read and analyze image ──
        cv2 = self._cv2
        img = None
        if cv2 is not None:
            img = cv2.imread(filepath)

        if img is not None:
            # cv2 path
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = float(cv2.mean(gray)[0]) / 255.0
            edges = cv2.Canny(gray, 50, 150)
            edge_count = int(cv2.countNonZero(edges))
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            contour_count = len(contours)
        else:
            # Numpy-only path: capture a fresh frame for analysis
            frame, err = self._grab_frame()
            if err:
                return f"Error: {err}"
            gray = self._to_grayscale(frame)
            brightness = float(np.mean(gray)) / 255.0
            # Simple gradient edge detection
            gy = np.abs(np.diff(gray.astype(np.int16), axis=0))
            gx = np.abs(np.diff(gray.astype(np.int16), axis=1))
            edge_count = int(np.count_nonzero(gy > 30)) + int(np.count_nonzero(gx > 30))
            contour_count = max(1, edge_count // 500)

        # ── Generate description via Anthropic API ──
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=200,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Describe what a camera sees based on these metrics "
                            f"in exactly one paragraph: brightness={brightness:.2f} "
                            f"(0=very dark, 1=very bright), edge_complexity="
                            f"{edge_count} edges detected (higher means more "
                            f"complex scene), distinct_regions={contour_count}. "
                            f"Be specific and natural sounding."
                        ),
                    }],
                )
                return message.content[0].text
            except Exception as e:
                # Fall through to local description
                pass

        # ── Local fallback ──
        return self._local_describe(brightness, edge_count, contour_count)

    @staticmethod
    def _local_describe(brightness: float, edge_count: int,
                        contour_count: int) -> str:
        """Compose a description locally without an API call."""
        if brightness < 0.15:
            light = "very dark"
        elif brightness < 0.35:
            light = "dimly lit"
        elif brightness < 0.65:
            light = "moderately bright"
        else:
            light = "brightly lit"

        if edge_count < 5000:
            complexity = "simple, mostly featureless"
        elif edge_count < 50000:
            complexity = "moderately complex"
        else:
            complexity = "highly complex and textured"

        return (
            f"The camera shows a {light}, {complexity} environment "
            f"with {edge_count} detected edges suggesting "
            f"{'minimal' if edge_count < 5000 else 'significant'} detail. "
            f"{contour_count} distinct objects or regions detected. "
            f"{'No significant motion detected.' if contour_count < 3 else f'{contour_count} separate regions suggest a cluttered or busy scene.'}"
        )

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


# ── Level 2: Camera + GPIO motors ────────────────────────────────────

class _MotorNav:
    """GPIO-based motor control combined with camera feedback."""

    def __init__(self, camera_nav: _CameraNav):
        self._camera = camera_nav
        self._gpio = None
        self._pwm_instances: Dict[int, Any] = {}
        self._initialized_pins: set = set()
        self._init_gpio()

    def _init_gpio(self):
        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
        except (ImportError, RuntimeError):
            try:
                import gpiozero  # noqa: F401
                self._gpio = None  # gpiozero uses different API
            except ImportError:
                self._gpio = None

    def _setup_pin(self, pin: int, mode: str = "out"):
        if self._gpio is None:
            return
        if pin not in self._initialized_pins:
            if mode == "out":
                self._gpio.setup(pin, self._gpio.OUT)
            else:
                self._gpio.setup(pin, self._gpio.IN)
            self._initialized_pins.add(pin)

    def set_pin(self, pin: int, value: bool) -> Dict:
        if self._gpio is None:
            return _err("GPIO not available")
        self._setup_pin(pin, "out")
        self._gpio.output(pin, self._gpio.HIGH if value else self._gpio.LOW)
        return _ok({"pin": pin, "value": value})

    def read_pin(self, pin: int) -> Dict:
        if self._gpio is None:
            return _err("GPIO not available")
        self._setup_pin(pin, "in")
        val = self._gpio.input(pin)
        return _ok({"pin": pin, "value": bool(val)})

    def pwm_set(self, pin: int, frequency: float, duty_cycle: float) -> Dict:
        if self._gpio is None:
            return _err("GPIO not available")
        self._setup_pin(pin, "out")
        duty_cycle = max(0.0, min(100.0, duty_cycle))
        if pin in self._pwm_instances:
            self._pwm_instances[pin].ChangeDutyCycle(duty_cycle)
            self._pwm_instances[pin].ChangeFrequency(frequency)
        else:
            pwm = self._gpio.PWM(pin, frequency)
            pwm.start(duty_cycle)
            self._pwm_instances[pin] = pwm
        return _ok({"pin": pin, "frequency": frequency,
                     "duty_cycle": duty_cycle})

    def motor_forward(self, left_pin: int, right_pin: int,
                      speed: float = 100.0) -> Dict:
        if self._gpio is None:
            return _err("GPIO not available")
        self.pwm_set(left_pin, 1000, speed)
        self.pwm_set(right_pin, 1000, speed)
        return _ok({"action": "forward", "speed": speed})

    def motor_turn(self, direction: str, left_pin: int, right_pin: int,
                   speed: float = 70.0) -> Dict:
        if self._gpio is None:
            return _err("GPIO not available")
        if direction == "left":
            self.pwm_set(left_pin, 1000, 0)
            self.pwm_set(right_pin, 1000, speed)
        else:
            self.pwm_set(left_pin, 1000, speed)
            self.pwm_set(right_pin, 1000, 0)
        return _ok({"action": "turn", "direction": direction, "speed": speed})

    def motor_stop(self, left_pin: int, right_pin: int) -> Dict:
        if self._gpio is None:
            return _err("GPIO not available")
        self.pwm_set(left_pin, 1000, 0)
        self.pwm_set(right_pin, 1000, 0)
        return _ok({"action": "stop"})

    def servo_set(self, pin: int, angle: float) -> Dict:
        if self._gpio is None:
            return _err("GPIO not available")
        angle = max(0, min(180, angle))
        # Standard servo: 2-12% duty cycle at 50Hz for 0-180 degrees
        duty = 2.0 + (angle / 180.0) * 10.0
        return self.pwm_set(pin, 50, duty)

    def navigate_to_color(self, color: str, left_pin: int = 17,
                          right_pin: int = 18, speed: float = 60.0,
                          timeout: float = 30.0) -> Dict:
        """Move toward a colored blob using camera feedback."""
        start = time.time()
        steps = 0
        while time.time() - start < timeout:
            steps += 1
            track = self._camera.track_object(color)
            if not track.get("success") or not track.get("found"):
                # Object lost — stop and search
                self.motor_stop(left_pin, right_pin)
                time.sleep(0.3)
                self.motor_turn("left", left_pin, right_pin, 40)
                time.sleep(0.4)
                self.motor_stop(left_pin, right_pin)
                continue

            cx = track["center_x"]
            area_frac = track.get("area_fraction", 0)

            # Close enough — arrived
            if area_frac > 0.15:
                self.motor_stop(left_pin, right_pin)
                return _ok({"action": "navigate_to_color", "color": color,
                            "arrived": True, "steps": steps})

            # Steer toward object
            if cx < 0.35:
                self.motor_turn("left", left_pin, right_pin, speed * 0.6)
            elif cx > 0.65:
                self.motor_turn("right", left_pin, right_pin, speed * 0.6)
            else:
                self.motor_forward(left_pin, right_pin, speed)
            time.sleep(0.2)

        self.motor_stop(left_pin, right_pin)
        return _ok({"action": "navigate_to_color", "color": color,
                     "arrived": False, "steps": steps, "timeout": True})

    def avoid_obstacle(self, left_pin: int = 17, right_pin: int = 18,
                       speed: float = 50.0) -> Dict:
        """If obstacle in center, turn until clear then forward."""
        scan = self._camera.detect_objects()
        objects = scan.get("objects", []) if scan.get("success") else []
        center_objs = [o for o in objects if 0.33 <= o.get("center_x", 0.5) <= 0.66]

        if not center_objs:
            self.motor_forward(left_pin, right_pin, speed)
            time.sleep(0.5)
            self.motor_stop(left_pin, right_pin)
            return _ok({"action": "avoid_obstacle", "obstacle": False,
                         "moved_forward": True})

        # Obstacle detected — turn away from most cluttered side
        left_objs = [o for o in objects if o.get("center_x", 0.5) < 0.33]
        right_objs = [o for o in objects if o.get("center_x", 0.5) > 0.66]
        turn_dir = "right" if len(left_objs) <= len(right_objs) else "left"

        for _ in range(10):  # max 10 turn increments
            self.motor_turn(turn_dir, left_pin, right_pin, speed * 0.7)
            time.sleep(0.3)
            self.motor_stop(left_pin, right_pin)
            time.sleep(0.1)
            rescan = self._camera.detect_objects()
            re_objs = rescan.get("objects", []) if rescan.get("success") else []
            re_center = [o for o in re_objs if 0.33 <= o.get("center_x", 0.5) <= 0.66]
            if not re_center:
                break

        self.motor_forward(left_pin, right_pin, speed)
        time.sleep(0.5)
        self.motor_stop(left_pin, right_pin)
        return _ok({"action": "avoid_obstacle", "obstacle": True,
                     "turn_direction": turn_dir})

    def follow_object(self, color: str, left_pin: int = 17,
                      right_pin: int = 18, speed: float = 50.0,
                      duration: float = 30.0) -> Dict:
        """Follow a colored object for duration seconds."""
        start = time.time()
        steps = 0
        while time.time() - start < duration:
            steps += 1
            track = self._camera.track_object(color)
            if not track.get("success") or not track.get("found"):
                self.motor_stop(left_pin, right_pin)
                time.sleep(0.2)
                continue

            cx = track["center_x"]
            area_frac = track.get("area_fraction", 0)

            # Too close — back off (not implemented without reverse pin)
            # Instead just stop
            if area_frac > 0.25:
                self.motor_stop(left_pin, right_pin)
            elif cx < 0.35:
                self.motor_turn("left", left_pin, right_pin, speed * 0.6)
            elif cx > 0.65:
                self.motor_turn("right", left_pin, right_pin, speed * 0.6)
            else:
                self.motor_forward(left_pin, right_pin, speed)
            time.sleep(0.15)

        self.motor_stop(left_pin, right_pin)
        return _ok({"action": "follow_object", "color": color, "steps": steps,
                     "duration": round(time.time() - start, 1)})

    def search_pattern(self, left_pin: int = 17, right_pin: int = 18,
                       speed: float = 50.0, legs: int = 4) -> Dict:
        """Execute systematic expanding-square search with camera scan at each leg."""
        scans = []
        for leg in range(legs):
            # Forward leg (increasing length)
            self.motor_forward(left_pin, right_pin, speed)
            time.sleep(1.0 + leg * 0.5)
            self.motor_stop(left_pin, right_pin)
            time.sleep(0.3)

            # Scan
            scan = self._camera.detect_objects()
            scans.append({"leg": leg, "scan": scan})

            # Turn 90 degrees (approximate)
            self.motor_turn("right", left_pin, right_pin, speed * 0.8)
            time.sleep(0.5)
            self.motor_stop(left_pin, right_pin)
            time.sleep(0.2)

        return _ok({"action": "search_pattern", "legs": legs,
                     "scans": scans})

    def dock_to_marker(self, marker_color: str, left_pin: int = 17,
                       right_pin: int = 18, speed: float = 30.0,
                       timeout: float = 30.0) -> Dict:
        """Slowly approach and stop at a colored marker."""
        start = time.time()
        while time.time() - start < timeout:
            track = self._camera.track_object(marker_color)
            if not track.get("success") or not track.get("found"):
                self.motor_stop(left_pin, right_pin)
                time.sleep(0.3)
                continue

            area_frac = track.get("area_fraction", 0)
            cx = track["center_x"]

            # Docked when marker fills >20% of frame
            if area_frac > 0.20:
                self.motor_stop(left_pin, right_pin)
                return _ok({"action": "dock_to_marker", "color": marker_color,
                            "docked": True, "area_fraction": area_frac})

            # Slow approach with steering
            approach_speed = speed * min(1.0, 0.3 / max(area_frac, 0.001))
            approach_speed = min(approach_speed, speed)
            if cx < 0.4:
                self.motor_turn("left", left_pin, right_pin, approach_speed * 0.5)
            elif cx > 0.6:
                self.motor_turn("right", left_pin, right_pin, approach_speed * 0.5)
            else:
                self.motor_forward(left_pin, right_pin, approach_speed)
            time.sleep(0.15)

        self.motor_stop(left_pin, right_pin)
        return _ok({"action": "dock_to_marker", "color": marker_color,
                     "docked": False, "timeout": True})

    def cleanup(self):
        for pwm in self._pwm_instances.values():
            try:
                pwm.stop()
            except Exception:
                pass
        self._pwm_instances.clear()
        if self._gpio is not None:
            try:
                self._gpio.cleanup()
            except Exception:
                pass


# ── Level 3: MAVLink flight controller ───────────────────────────────

class _FlightNav:
    """MAVLink-based flight control via dronekit or pymavlink."""

    def __init__(self, camera_nav: _CameraNav):
        self._camera = camera_nav
        self._vehicle = None
        self._dronekit = _try_import("dronekit")
        self._connected = False

    def connect(self, port: str = "/dev/ttyUSB0",
                baud: int = 57600) -> Dict:
        if self._dronekit is None:
            return _err("dronekit not installed")
        try:
            self._vehicle = self._dronekit.connect(port, baud=baud,
                                                    wait_ready=True,
                                                    timeout=30)
            self._connected = True
            return _ok({"port": port, "baud": baud,
                         "firmware": str(self._vehicle.version)})
        except Exception as e:
            return _err(f"connect failed: {e}")

    def arm(self) -> Dict:
        if not self._connected:
            return _err("not connected")
        v = self._vehicle
        v.mode = self._dronekit.VehicleMode("GUIDED")
        v.armed = True
        timeout = 15
        while not v.armed and timeout > 0:
            time.sleep(1)
            timeout -= 1
        if v.armed:
            return _ok({"armed": True})
        return _err("arming timed out")

    def takeoff(self, altitude: float = 5.0) -> Dict:
        if not self._connected:
            return _err("not connected")
        arm_result = self.arm()
        if not arm_result["success"]:
            return arm_result
        self._vehicle.simple_takeoff(altitude)
        # Wait for altitude
        while True:
            alt = self._vehicle.location.global_relative_frame.alt
            if alt >= altitude * 0.95:
                break
            time.sleep(1)
        return _ok({"altitude": altitude, "actual_alt": alt})

    def navigate_waypoint(self, lat: float, lon: float,
                          alt: float = 10.0) -> Dict:
        if not self._connected:
            return _err("not connected")
        dk = self._dronekit
        target = dk.LocationGlobalRelative(lat, lon, alt)
        self._vehicle.simple_goto(target)
        return _ok({"target_lat": lat, "target_lon": lon, "target_alt": alt})

    def visual_navigate(self, target_color: str = "red",
                        altitude: float = 5.0) -> Dict:
        """Fly toward a visual target using camera feedback + yaw control."""
        if not self._connected:
            return _err("not connected")

        track = self._camera.track_object(target_color)
        if not track.get("success") or not track.get("found"):
            return _ok({"found": False, "color": target_color})

        cx = track["center_x"]
        # Convert center_x (0-1) to yaw adjustment
        yaw_offset = (cx - 0.5) * 60  # ±30 degrees
        return _ok({"found": True, "color": target_color,
                     "yaw_adjustment": round(yaw_offset, 1),
                     "center_x": cx})

    def move(self, direction: str, speed: float = 1.0,
             duration: float = 2.0) -> Dict:
        """Move in a direction for duration seconds."""
        if not self._connected:
            return _err("not connected")
        # Send velocity command
        vx, vy, vz = 0.0, 0.0, 0.0
        d = direction.lower()
        if d == "forward":
            vx = speed
        elif d == "backward":
            vx = -speed
        elif d == "left":
            vy = -speed
        elif d == "right":
            vy = speed
        elif d == "up":
            vz = -speed
        elif d == "down":
            vz = speed
        else:
            return _err(f"unknown direction: {direction}")

        # pymavlink velocity message
        try:
            from pymavlink import mavutil
            msg = self._vehicle.message_factory.set_position_target_local_ned_encode(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000111111000111,
                0, 0, 0, vx, vy, vz,
                0, 0, 0, 0, 0)
            self._vehicle.send_mavlink(msg)
            time.sleep(duration)
            # Stop
            stop_msg = self._vehicle.message_factory.set_position_target_local_ned_encode(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000111111000111,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0)
            self._vehicle.send_mavlink(stop_msg)
        except Exception as e:
            return _err(f"velocity command failed: {e}")

        return _ok({"direction": direction, "speed": speed,
                     "duration": duration})

    def land(self) -> Dict:
        if not self._connected:
            return _err("not connected")
        self._vehicle.mode = self._dronekit.VehicleMode("LAND")
        return _ok({"action": "land"})

    def get_telemetry(self) -> Dict:
        if not self._connected:
            return _err("not connected")
        v = self._vehicle
        loc = v.location.global_relative_frame
        att = v.attitude
        bat = v.battery
        return _ok({
            "battery_voltage": bat.voltage,
            "battery_level": bat.level,
            "altitude": loc.alt,
            "latitude": loc.lat,
            "longitude": loc.lon,
            "roll": round(math.degrees(att.roll), 2),
            "pitch": round(math.degrees(att.pitch), 2),
            "yaw": round(math.degrees(att.yaw), 2),
            "groundspeed": v.groundspeed,
            "airspeed": v.airspeed,
        })

    def return_to_launch(self) -> Dict:
        if not self._connected:
            return _err("not connected")
        self._vehicle.mode = self._dronekit.VehicleMode("RTL")
        return _ok({"action": "return_to_launch"})

    def emergency_land(self) -> Dict:
        if not self._connected:
            return _err("not connected")
        self._vehicle.mode = self._dronekit.VehicleMode("LAND")
        self._vehicle.channels.overrides = {}
        return _ok({"action": "emergency_land"})

    def close(self):
        if self._vehicle is not None:
            try:
                self._vehicle.close()
            except Exception:
                pass
            self._vehicle = None
            self._connected = False


# ── System sensor reader ─────────────────────────────────────────────

class _SystemSensors:
    """Read system metrics for the observation vector. Always available."""

    def __init__(self):
        self._psutil = _try_import("psutil")

    def get_cpu_temp(self) -> float:
        """Return CPU temperature in Celsius, or -1 if unavailable."""
        # Linux: /sys/class/thermal
        thermal = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(thermal):
            try:
                with open(thermal) as f:
                    return int(f.read().strip()) / 1000.0
            except (OSError, ValueError):
                pass
        # psutil (macOS / fallback)
        if self._psutil:
            temps = getattr(self._psutil, "sensors_temperatures", lambda: None)()
            if temps:
                for entries in temps.values():
                    if entries:
                        return entries[0].current
        return -1.0

    def get_cpu_percent(self) -> float:
        if self._psutil:
            return self._psutil.cpu_percent(interval=None) / 100.0
        return 0.0

    def get_ram_percent(self) -> float:
        if self._psutil:
            return self._psutil.virtual_memory().percent / 100.0
        return 0.0

    def get_disk_space(self) -> Dict:
        try:
            usage = __import__("shutil").disk_usage("/")
            return _ok({"free_gb": round(usage.free / (1024**3), 1),
                         "total_gb": round(usage.total / (1024**3), 1),
                         "percent_used": round(usage.used / usage.total * 100, 1)})
        except OSError:
            return _err("cannot read disk")

    def get_battery(self) -> float:
        """Battery percent 0-1, or 0.95 if no battery (desktop)."""
        if self._psutil:
            batt = self._psutil.sensors_battery()
            if batt is not None:
                return batt.percent / 100.0
        return 0.95


# ── IMU reader ────────────────────────────────────────────────────────

class _IMUSensor:
    """Read MPU6050 or similar I2C IMU."""

    def __init__(self):
        self._smbus = _try_import("smbus2")
        self._bus = None
        self._addr = 0x68
        if self._smbus:
            try:
                self._bus = self._smbus.SMBus(1)
                # Wake up MPU6050
                self._bus.write_byte_data(self._addr, 0x6B, 0)
            except (OSError, IOError):
                self._bus = None

    def _read_word(self, reg: int) -> int:
        if not self._bus:
            return 0
        high = self._bus.read_byte_data(self._addr, reg)
        low = self._bus.read_byte_data(self._addr, reg + 1)
        val = (high << 8) | low
        if val >= 0x8000:
            val = val - 0x10000
        return val

    def read_accelerometer(self) -> Dict:
        if not self._bus:
            return _err("IMU not available")
        ax = self._read_word(0x3B) / 16384.0
        ay = self._read_word(0x3D) / 16384.0
        az = self._read_word(0x3F) / 16384.0
        return _ok({"x": round(ax, 4), "y": round(ay, 4), "z": round(az, 4)})

    def read_gyroscope(self) -> Dict:
        if not self._bus:
            return _err("IMU not available")
        gx = self._read_word(0x43) / 131.0
        gy = self._read_word(0x45) / 131.0
        gz = self._read_word(0x47) / 131.0
        return _ok({"x": round(gx, 4), "y": round(gy, 4), "z": round(gz, 4)})

    def get_orientation(self) -> Dict:
        accel = self.read_accelerometer()
        if not accel.get("success"):
            return _err("IMU not available")
        ax, ay, az = accel["x"], accel["y"], accel["z"]
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        return _ok({"roll": round(math.degrees(roll), 2),
                     "pitch": round(math.degrees(pitch), 2),
                     "yaw": 0.0})  # yaw requires magnetometer

    def detect_vibration(self) -> Dict:
        accel = self.read_accelerometer()
        if not accel.get("success"):
            return _err("IMU not available")
        magnitude = math.sqrt(accel["x"]**2 + accel["y"]**2 + accel["z"]**2)
        # Gravity is ~1g, so vibration = deviation from 1.0
        vibration = abs(magnitude - 1.0)
        return _ok({"magnitude": round(vibration, 4),
                     "raw_accel_mag": round(magnitude, 4)})

    @property
    def available(self) -> bool:
        return self._bus is not None


# ── Color HSV helpers ─────────────────────────────────────────────────

def _get_color_hsv_range(color_name: str):
    """Return (lower, upper) HSV bounds for a named color, or None."""
    ranges = {
        "red":    (np.array([0, 120, 70]),   np.array([10, 255, 255])),
        "orange": (np.array([11, 120, 70]),  np.array([25, 255, 255])),
        "yellow": (np.array([26, 120, 70]),  np.array([35, 255, 255])),
        "green":  (np.array([36, 80, 70]),   np.array([85, 255, 255])),
        "blue":   (np.array([86, 120, 70]),  np.array([130, 255, 255])),
        "purple": (np.array([131, 80, 70]),  np.array([160, 255, 255])),
        "pink":   (np.array([161, 80, 70]),  np.array([175, 255, 255])),
        "white":  (np.array([0, 0, 200]),    np.array([180, 30, 255])),
        "black":  (np.array([0, 0, 0]),      np.array([180, 255, 50])),
    }
    return ranges.get(color_name.lower())


def _classify_shape(vertices: int, w: int, h: int) -> str:
    """Classify shape from vertex count and bounding rect."""
    if vertices == 3:
        return "triangle"
    elif vertices == 4:
        aspect = w / max(h, 1)
        return "square" if 0.8 <= aspect <= 1.2 else "rectangle"
    elif vertices == 5:
        return "pentagon"
    elif vertices > 6:
        return "circle"
    return f"polygon_{vertices}"


# ── Navigation Engine ─────────────────────────────────────────────────

# Level definitions
LEVEL_CAMERA = 1
LEVEL_MOTOR = 2
LEVEL_FLIGHT = 3

# Action registry per level — each level adds to all lower levels
_LEVEL_ACTIONS: Dict[int, List[str]] = {
    LEVEL_CAMERA: [
        "visual_scan", "track_object", "measure_brightness",
        "report_surroundings", "detect_color",
        # Extended camera actions (always available at level 1+)
        "estimate_distance", "detect_objects", "capture_image",
        "yolo_detect", "describe_scene", "count_objects", "count_fingers",
    ],
    LEVEL_MOTOR: [
        "navigate_to_color", "avoid_obstacle", "follow_object",
        "search_pattern",
        # Extended motor actions
        "dock_to_marker",
        "set_pin", "read_pin", "pwm_set",
        "motor_forward", "motor_turn", "motor_stop", "servo_set",
    ],
    LEVEL_FLIGHT: [
        "takeoff_and_hover", "navigate_waypoint", "visual_navigate",
        "return_home", "emergency_land",
        # Extended flight actions
        "connect", "arm", "takeoff", "move", "land", "get_telemetry",
        "return_to_launch",
    ],
}

# System actions always available
_SYSTEM_ACTIONS = [
    "get_cpu_temp", "get_cpu_percent", "get_ram_percent",
    "get_disk_space", "get_battery",
]

# IMU actions
_IMU_ACTIONS = [
    "read_accelerometer", "read_gyroscope",
    "get_orientation", "detect_vibration",
]


class NavigationEngine:
    """
    Hardware-agnostic navigation engine with three capability levels.

    Reads the ToolDiscovery manifest to determine which level is available
    and exposes only valid actions. Accepts an optional tools dict from
    ToolBuilder — if provided, uses those tool objects for observe() and
    execute(); otherwise falls back to internal subsystem classes.

    Provides a unified observe() returning the 15-dim observation vector
    with real sensor data where available and safe defaults otherwise.
    """

    def __init__(self, manifest: Dict, tools: Optional[Dict] = None):
        self._manifest = manifest
        self._tools = tools or {}
        self._level = self._detect_level(manifest)
        self._step_count = 0
        self._max_steps = 300

        # Initialize subsystems based on level
        # Use ToolBuilder tools if provided, otherwise create internal ones
        self._camera: Optional[_CameraNav] = None
        self._motors: Optional[_MotorNav] = None
        self._flight: Optional[_FlightNav] = None
        self._system = _SystemSensors()
        self._imu = _IMUSensor()

        hw = manifest.get("hardware", {})

        if hw.get("camera", {}).get("available", False):
            self._camera = _CameraNav()

        if self._level >= LEVEL_MOTOR:
            self._motors = _MotorNav(self._camera)

        if self._level >= LEVEL_FLIGHT:
            self._flight = _FlightNav(self._camera)

    @staticmethod
    def _detect_level(manifest: Dict) -> int:
        """Determine highest navigation level from manifest.

        Also checks motor_controllers from auto-detection:
        - mavlink_fc → level 3 (flight)
        - pca9685_driver, gpio_direct_motors, ros_motors, odrive, vesc,
          roboclaw → level 2 (ground motors)
        """
        hw = manifest.get("hardware", {})
        has_camera = hw.get("camera", {}).get("available", False)
        has_gpio = hw.get("gpio", {}).get("available", False)
        has_mavlink = hw.get("mavlink", {}).get("available", False)

        # Check auto-detected motor controllers
        motor_ctrls = manifest.get("motor_controllers", [])
        ctrl_types = {c.get("controller_type", "") for c in motor_ctrls}

        has_flight_ctrl = has_mavlink or "mavlink_fc" in ctrl_types
        has_motor_ctrl = (has_gpio
                          or bool(ctrl_types & {"pca9685_driver",
                                                "gpio_direct_motors",
                                                "ros_motors", "odrive",
                                                "vesc", "roboclaw"}))

        if has_flight_ctrl and has_camera:
            return LEVEL_FLIGHT
        if (has_motor_ctrl or has_gpio) and has_camera:
            return LEVEL_MOTOR
        if has_camera:
            return LEVEL_CAMERA
        return 0  # no camera — system-only

    @property
    def level(self) -> int:
        return self._level

    @property
    def level_name(self) -> str:
        names = {0: "system-only", 1: "camera", 2: "motor", 3: "flight"}
        return names.get(self._level, "unknown")

    def available_actions(self) -> List[str]:
        """Return all actions valid at the current level."""
        actions = list(_SYSTEM_ACTIONS)
        if self._imu.available:
            actions.extend(_IMU_ACTIONS)
        for lvl in range(1, self._level + 1):
            actions.extend(_LEVEL_ACTIONS.get(lvl, []))
        actions.append("observe")
        return sorted(set(actions))

    def execute(self, action: str, params: Optional[Dict] = None):
        """Execute a navigation action by name.

        Returns a structured dict for most actions, or a plain string
        for report_surroundings (so the executor can print it directly).
        """
        params = params or {}
        valid = self.available_actions()
        if action not in valid:
            result = _err(f"action '{action}' not available at level "
                          f"{self._level} ({self.level_name})")
            _log_call(action, params, result)
            return result

        # Dispatch to the correct subsystem
        try:
            result = self._dispatch(action, params)
        except Exception as e:
            result = _err(f"execution error: {e}")

        _log_call(action, params, result)
        return result

    def _dispatch(self, action: str, params: Dict) -> Dict:
        """Route action to the correct handler."""
        # System actions
        if action == "get_cpu_temp":
            return _ok({"cpu_temp": self._system.get_cpu_temp()})
        if action == "get_cpu_percent":
            return _ok({"cpu_percent": round(self._system.get_cpu_percent() * 100, 1)})
        if action == "get_ram_percent":
            return _ok({"ram_percent": round(self._system.get_ram_percent() * 100, 1)})
        if action == "get_disk_space":
            return self._system.get_disk_space()
        if action == "get_battery":
            return _ok({"battery_percent": round(self._system.get_battery() * 100, 1)})
        if action == "observe":
            return _ok({"observation": self.observe().tolist()})

        # IMU actions
        if action == "read_accelerometer":
            return self._imu.read_accelerometer()
        if action == "read_gyroscope":
            return self._imu.read_gyroscope()
        if action == "get_orientation":
            return self._imu.get_orientation()
        if action == "detect_vibration":
            return self._imu.detect_vibration()

        # Level 1: Camera
        if self._camera:
            cam_map = {
                "visual_scan": lambda: self._camera.visual_scan(),
                "track_object": lambda: self._camera.track_object(
                    params.get("color", params.get("color_or_shape", "red"))),
                "estimate_distance": lambda: self._camera.estimate_distance(
                    params.get("object_pixels", 1000)),
                "detect_objects": lambda: self._camera.detect_objects(),
                "measure_brightness": lambda: self._camera.measure_brightness(),
                "detect_color": lambda: self._camera.detect_color(
                    params.get("color_name", params.get("color", "red"))),
                "capture_image": lambda: self._camera.capture_image(),
                "report_surroundings": lambda: self._camera.report_surroundings(
                    scan_data=params.get("scan_data"),
                    path=params.get("path", "")),
            }
            if action in cam_map:
                return cam_map[action]()

        # YOLO actions (level 1+ when yolo tool available)
        yolo = self._tools.get("yolo")
        if yolo:
            yolo_map = {
                "yolo_detect": lambda: yolo.detect(
                    params.get("image_path", "")),
                "describe_scene": lambda: yolo.describe_scene(**params),
                "count_objects": lambda: yolo.count_objects(
                    params.get("class_name", "")),
                "count_fingers": lambda: yolo.count_fingers(**params),
            }
            if action in yolo_map:
                return yolo_map[action]()

        # Level 2: Motor
        if self._motors:
            lp = params.get("left_pin", 17)
            rp = params.get("right_pin", 18)
            sp = params.get("speed", 50.0)
            motor_map = {
                "set_pin": lambda: self._motors.set_pin(
                    params["pin"], params.get("value", True)),
                "read_pin": lambda: self._motors.read_pin(params["pin"]),
                "pwm_set": lambda: self._motors.pwm_set(
                    params["pin"], params.get("frequency", 1000),
                    params.get("duty_cycle", 50)),
                "motor_forward": lambda: self._motors.motor_forward(lp, rp, sp),
                "motor_turn": lambda: self._motors.motor_turn(
                    params.get("direction", "left"), lp, rp, sp),
                "motor_stop": lambda: self._motors.motor_stop(lp, rp),
                "servo_set": lambda: self._motors.servo_set(
                    params["pin"], params.get("angle", 90)),
                "navigate_to_color": lambda: self._motors.navigate_to_color(
                    params.get("color", params.get("class_name", "red")),
                    lp, rp, sp, params.get("timeout", 30)),
                "avoid_obstacle": lambda: self._motors.avoid_obstacle(lp, rp, sp),
                "follow_object": lambda: self._motors.follow_object(
                    params.get("color", "red"), lp, rp, sp,
                    params.get("duration", 30)),
                "search_pattern": lambda: self._motors.search_pattern(
                    lp, rp, sp, params.get("legs", 4)),
                "dock_to_marker": lambda: self._motors.dock_to_marker(
                    params.get("marker_color", params.get("color", "red")),
                    lp, rp, sp, params.get("timeout", 30)),
            }
            if action in motor_map:
                return motor_map[action]()

        # Level 3: Flight
        if self._flight:
            flight_map = {
                "connect": lambda: self._flight.connect(
                    params.get("port", "/dev/ttyUSB0"),
                    params.get("baud", 57600)),
                "arm": lambda: self._flight.arm(),
                "takeoff": lambda: self._flight.takeoff(
                    params.get("altitude", 5.0)),
                "takeoff_and_hover": lambda: self._flight.takeoff(
                    params.get("altitude", 5.0)),
                "navigate_waypoint": lambda: self._flight.navigate_waypoint(
                    params["lat"], params["lon"],
                    params.get("alt", 10.0)),
                "visual_navigate": lambda: self._flight.visual_navigate(
                    params.get("target_color", "red"),
                    params.get("altitude", 5.0)),
                "move": lambda: self._flight.move(
                    params.get("direction", "forward"),
                    params.get("speed", 1.0),
                    params.get("duration", 2.0)),
                "land": lambda: self._flight.land(),
                "get_telemetry": lambda: self._flight.get_telemetry(),
                "return_to_launch": lambda: self._flight.return_to_launch(),
                "return_home": lambda: self._flight.return_to_launch(),
                "emergency_land": lambda: self._flight.emergency_land(),
            }
            if action in flight_map:
                return flight_map[action]()

        return _err(f"no handler for action: {action}")

    # ── Unified observation vector ────────────────────────────────

    def observe(self) -> np.ndarray:
        """Return 15-dim observation vector using real sensor data where available.

        Uses ToolBuilder tools (self._tools) when provided, falls back to
        internal subsystem classes. Missing sensors get safe defaults (0.5).

        Index mapping:
          [0]  battery       — from system tool or psutil, else 1.0
          [1]  solar_power   — 0.5 default (no sensor)
          [2]  bus_voltage   — 0.5 default (no sensor)
          [3]  imu_x         — from IMU if available, else 0.5
          [4]  imu_y         — from IMU if available, else 0.5
          [5]  imu_z         — from IMU if available, else 0.5
          [6]  attitude_error — from IMU if available, else 0.5
          [7]  cpu_percent   — from system tool or psutil
          [8]  ram_percent   — from system tool or psutil
          [9]  motion_score  — from camera motion detection, else 0.5
          [10] brightness    — from camera brightness, else 0.5
          [11] obstacle_dist — 0.5 default (no proximity sensor)
          [12] target_dist   — 0.5 default
          [13] target_bearing— 0.5 default
          [14] mission_progress — step_count / max_steps
        """
        self._step_count += 1

        # Start with safe defaults
        obs = np.full(OBS_DIM, 0.5, dtype=float)

        # [0] battery — try ToolBuilder system tool, then internal
        sys_tool = self._tools.get("system")
        if sys_tool:
            batt = sys_tool.get_battery()
            if batt["success"] and batt["result"].get("percent") is not None:
                obs[0] = batt["result"]["percent"] / 100.0
            else:
                obs[0] = 1.0  # desktop without battery
        else:
            obs[0] = self._system.get_battery()

        # [7] cpu_percent — try ToolBuilder, then internal
        if sys_tool:
            cpu = sys_tool.get_cpu_percent()
            if cpu["success"]:
                obs[7] = cpu["result"]["cpu_percent"] / 100.0
        else:
            obs[7] = self._system.get_cpu_percent()

        # [8] ram_percent — try ToolBuilder, then internal
        if sys_tool:
            ram = sys_tool.get_ram_percent()
            if ram["success"]:
                obs[8] = ram["result"]["ram_percent"] / 100.0
        else:
            obs[8] = self._system.get_ram_percent()

        # [3-6] IMU — from internal IMU sensor if available
        if self._imu.available:
            accel = self._imu.read_accelerometer()
            if accel.get("success"):
                obs[3] = np.clip((accel["x"] + 2.0) / 4.0, 0, 1)
                obs[4] = np.clip((accel["y"] + 2.0) / 4.0, 0, 1)
                obs[5] = np.clip((accel["z"] + 2.0) / 4.0, 0, 1)
                mag = math.sqrt(accel["x"]**2 + accel["y"]**2)
                obs[6] = np.clip(mag / 2.0, 0, 1)

        # [9] motion_score — try ToolBuilder camera, then internal
        cam_tool = self._tools.get("camera")
        if cam_tool:
            motion = cam_tool.detect_motion()
            if motion["success"]:
                obs[9] = motion["result"]["motion_score"]
        elif self._camera:
            detected, _ = self._camera.get_motion_bearing()
            obs[9] = 1.0 if detected else 0.0

        # [10] brightness — try ToolBuilder camera, then internal
        if cam_tool:
            bright = cam_tool.measure_brightness()
            if bright["success"]:
                obs[10] = bright["result"]["brightness"]
        elif self._camera:
            bright_result = self._camera.measure_brightness()
            if bright_result.get("success"):
                obs[10] = bright_result["brightness"]

        # [14] mission_progress
        obs[14] = min(self._step_count / max(self._max_steps, 1), 1.0)

        return np.clip(obs, 0.0, 1.0)

    def reset(self, max_steps: int = 300) -> np.ndarray:
        """Reset step counter and return initial observation."""
        self._step_count = 0
        self._max_steps = max_steps
        if self._camera:
            self._camera._prev_gray = None
        return self.observe()

    def close(self) -> None:
        """Release all hardware resources."""
        if self._camera:
            self._camera.close()
        if self._motors:
            self._motors.cleanup()
        if self._flight:
            self._flight.close()

    def print_summary(self) -> None:
        """Print navigation capability summary."""
        actions = self.available_actions()
        print(f"  Navigation Level: {self._level} ({self.level_name})")
        print(f"  Available Actions: {len(actions)}")
        for a in actions:
            print(f"    - {a}")
