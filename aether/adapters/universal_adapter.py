"""
UniversalAdapter: hardware-agnostic action executor that routes commands
to the correct hardware backend based on the ToolDiscovery capability manifest.

Always-available actions: stop, emergency_stop, report_state, safe_mode,
scan_environment.  Additional actions enabled per detected hardware.
Every call is logged to logs/hardware_calls.json.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .base_adapter import HardwareAdapter

_LOG_PATH = os.path.join("logs", "hardware_calls.json")


def _log_call(action: str, params: Dict, result: Dict) -> None:
    """Append call record to logs/hardware_calls.json."""
    entry = {
        "timestamp": time.time(),
        "action": action,
        "params": {k: str(v)[:200] for k, v in params.items()},
        "success": result.get("success", False),
    }
    os.makedirs(os.path.dirname(_LOG_PATH) or ".", exist_ok=True)
    existing = []
    if os.path.exists(_LOG_PATH):
        try:
            with open(_LOG_PATH) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append(entry)
    if len(existing) > 2000:
        existing = existing[-2000:]
    try:
        with open(_LOG_PATH, "w") as f:
            json.dump(existing, f, indent=2)
    except OSError:
        pass


def _ok(result: Any = None) -> Dict:
    return {"success": True, "result": result, "error": ""}


def _err(msg: str) -> Dict:
    return {"success": False, "result": None, "error": msg}


# ── Hardware requirement descriptions for error messages ──────────────

_ACTION_REQUIREMENTS: Dict[str, str] = {
    "capture_image":    "camera (cv2 + VideoCapture)",
    "detect_motion":    "camera (cv2 + VideoCapture)",
    "gpio_control":     "GPIO (RPi.GPIO or gpiozero)",
    "motor_forward":    "GPIO motor pins",
    "motor_turn":       "GPIO motor pins",
    "mavlink_command":  "MAVLink flight controller (serial or dronekit)",
    "takeoff":          "MAVLink flight controller",
    "land":             "MAVLink flight controller",
    "return_to_launch": "MAVLink flight controller",
}


class UniversalAdapter(HardwareAdapter):
    """
    Routes actions to the correct hardware backend based on the capability
    manifest from ToolDiscovery. Implements the HardwareAdapter ABC.

    Usage:
        from aether.core.tool_discovery import ToolDiscovery
        from aether.adapters.universal_adapter import UniversalAdapter

        manifest = ToolDiscovery().discover()
        adapter = UniversalAdapter(manifest)
        result = adapter.execute("scan_environment", {})
    """

    def __init__(self, manifest: Dict):
        self._manifest = manifest
        self._hw = manifest.get("hardware", {})
        self._sw = manifest.get("software", {})
        self._net = manifest.get("network", {})

        self._has_camera = self._hw.get("camera", {}).get("available", False)
        self._has_gpio = self._hw.get("gpio", {}).get("available", False)
        self._has_mavlink = self._hw.get("mavlink", {}).get("available", False)
        self._has_imu = self._hw.get("imu", {}).get("available", False)
        self._has_audio = self._hw.get("audio", {}).get("available", False)
        self._has_gpu = self._hw.get("gpu", {}).get("available", False)

        self._safe_mode = False
        self._stopped = False

        # Lazy-init hardware handles
        self._cv2 = None
        self._cap = None
        self._gpio = None
        self._pwm_instances: Dict[int, Any] = {}
        self._gpio_pins: set = set()
        self._vehicle = None  # dronekit vehicle

    # ── HardwareAdapter ABC implementation ────────────────────────

    def execute(self, action: str, state: Optional[Dict] = None) -> Tuple[Dict, bool]:
        """Execute action. Returns (result_dict, success). ABC compat."""
        params = state or {}
        result = self.run(action, params)
        return result, result.get("success", False)

    def is_action_available(self, action: str) -> bool:
        return action in self.available_actions()

    def get_degradation_state(self) -> Dict:
        return {
            "camera": 0.0 if self._has_camera else 1.0,
            "gpio": 0.0 if self._has_gpio else 1.0,
            "mavlink": 0.0 if self._has_mavlink else 1.0,
            "imu": 0.0 if self._has_imu else 1.0,
            "safe_mode": 1.0 if self._safe_mode else 0.0,
        }

    # ── Public API ────────────────────────────────────────────────

    def available_actions(self) -> List[str]:
        """Return list of valid action strings based on manifest."""
        actions = [
            "stop", "emergency_stop", "report_state",
            "safe_mode", "scan_environment",
        ]
        if self._has_camera:
            actions.extend(["capture_image", "detect_motion"])
        if self._has_gpio:
            actions.extend(["gpio_control", "motor_forward", "motor_turn"])
        if self._has_mavlink:
            actions.extend(["mavlink_command", "takeoff", "land",
                            "return_to_launch"])
        return sorted(actions)

    def get_hardware_state(self) -> Dict:
        """Return status of each hardware component."""
        return {
            "camera": {
                "available": self._has_camera,
                "type": self._hw.get("camera", {}).get("type", "none"),
                "active": self._cap is not None and self._cap.isOpened()
                          if self._cap else False,
            },
            "gpio": {
                "available": self._has_gpio,
                "type": self._hw.get("gpio", {}).get("type", "none"),
                "active_pins": len(self._gpio_pins),
            },
            "mavlink": {
                "available": self._has_mavlink,
                "type": self._hw.get("mavlink", {}).get("type", "none"),
                "connected": self._vehicle is not None,
            },
            "imu": {
                "available": self._has_imu,
                "type": self._hw.get("imu", {}).get("type", "none"),
            },
            "audio": {
                "available": self._has_audio,
            },
            "gpu": {
                "available": self._has_gpu,
                "type": self._hw.get("gpu", {}).get("type", "none"),
            },
            "adapter_state": {
                "safe_mode": self._safe_mode,
                "stopped": self._stopped,
            },
        }

    def run(self, action: str, params: Optional[Dict] = None) -> Dict:
        """Execute an action by name. Returns {success, result, error}."""
        params = params or {}
        valid = self.available_actions()

        if action not in valid:
            requirement = _ACTION_REQUIREMENTS.get(action, "unknown hardware")
            result = _err(
                f"action '{action}' not available — requires {requirement} "
                f"which was not detected in the capability manifest"
            )
            _log_call(action, params, result)
            return result

        if self._safe_mode and action not in (
            "stop", "emergency_stop", "report_state", "safe_mode"
        ):
            result = _err(f"safe_mode active — only stop/emergency_stop/"
                          f"report_state/safe_mode allowed")
            _log_call(action, params, result)
            return result

        try:
            result = self._dispatch(action, params)
        except Exception as e:
            result = _err(f"execution error: {e}")

        _log_call(action, params, result)
        return result

    # ── Dispatch ──────────────────────────────────────────────────

    def _dispatch(self, action: str, params: Dict) -> Dict:
        """Route action to the correct handler."""
        dispatch = {
            "stop":             self._do_stop,
            "emergency_stop":   self._do_emergency_stop,
            "report_state":     self._do_report_state,
            "safe_mode":        self._do_safe_mode,
            "scan_environment": self._do_scan_environment,
            "capture_image":    self._do_capture_image,
            "detect_motion":    self._do_detect_motion,
            "gpio_control":     self._do_gpio_control,
            "motor_forward":    self._do_motor_forward,
            "motor_turn":       self._do_motor_turn,
            "mavlink_command":  self._do_mavlink_command,
            "takeoff":          self._do_takeoff,
            "land":             self._do_land,
            "return_to_launch": self._do_return_to_launch,
        }
        handler = dispatch.get(action)
        if handler is None:
            return _err(f"no handler for action: {action}")
        return handler(params)

    # ── Always-available actions ──────────────────────────────────

    def _do_stop(self, params: Dict) -> Dict:
        self._stopped = True
        self._stop_all_motors()
        return _ok({"action": "stop", "all_motors_stopped": True})

    def _do_emergency_stop(self, params: Dict) -> Dict:
        self._stopped = True
        self._safe_mode = True
        self._stop_all_motors()
        if self._vehicle is not None:
            try:
                import dronekit
                self._vehicle.mode = dronekit.VehicleMode("LAND")
            except Exception:
                pass
        return _ok({"action": "emergency_stop", "safe_mode": True,
                     "motors_stopped": True})

    def _do_report_state(self, params: Dict) -> Dict:
        return _ok(self.get_hardware_state())

    def _do_safe_mode(self, params: Dict) -> Dict:
        enable = params.get("enable", True)
        self._safe_mode = bool(enable)
        if self._safe_mode:
            self._stop_all_motors()
        return _ok({"safe_mode": self._safe_mode})

    def _do_scan_environment(self, params: Dict) -> Dict:
        """Capture a frame via cv2 and return basic stats."""
        cv2 = self._ensure_cv2()
        if cv2 is None:
            # No camera — return system-only scan
            return _ok({
                "camera": False,
                "detail": "no camera available, returning system-only scan",
            })

        cap = self._ensure_capture()
        if cap is None:
            return _ok({"camera": False, "detail": "camera not opened"})

        ret, frame = cap.read()
        if not ret:
            return _ok({"camera": False, "detail": "failed to read frame"})

        import numpy as np
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        brightness = float(np.mean(gray)) / 255.0
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / (h * w)

        # Simple color distribution
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_mean = float(np.mean(hsv[:, :, 0])) / 180.0
        sat_mean = float(np.mean(hsv[:, :, 1])) / 255.0

        return _ok({
            "camera": True,
            "resolution": f"{w}x{h}",
            "brightness": round(brightness, 4),
            "edge_density": round(edge_density, 4),
            "hue_mean": round(hue_mean, 4),
            "saturation_mean": round(sat_mean, 4),
        })

    # ── Camera actions ────────────────────────────────────────────

    def _do_capture_image(self, params: Dict) -> Dict:
        cv2 = self._ensure_cv2()
        if cv2 is None:
            return _err("cv2 not available")

        cap = self._ensure_capture()
        if cap is None:
            return _err("camera not opened")

        ret, frame = cap.read()
        if not ret:
            return _err("failed to read frame")

        out_dir = os.path.join("logs", "captures")
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        path = os.path.join(out_dir, f"frame_{ts}.jpg")
        cv2.imwrite(path, frame)
        h, w = frame.shape[:2]
        return _ok({"filepath": path, "resolution": f"{w}x{h}"})

    def _do_detect_motion(self, params: Dict) -> Dict:
        cv2 = self._ensure_cv2()
        if cv2 is None:
            return _err("cv2 not available")

        cap = self._ensure_capture()
        if cap is None:
            return _err("camera not opened")

        import numpy as np
        prev_gray = None
        scores = []

        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if prev_gray is not None:
                delta = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                motion_px = np.count_nonzero(thresh)
                total_px = thresh.shape[0] * thresh.shape[1]
                scores.append(motion_px / total_px)
            prev_gray = gray
            time.sleep(0.08)

        if not scores:
            return _err("no frames captured for motion detection")

        avg = sum(scores) / len(scores)
        return _ok({"motion_score": round(avg, 4),
                     "frame_scores": [round(s, 4) for s in scores],
                     "motion_detected": avg > 0.01})

    # ── GPIO actions ──────────────────────────────────────────────

    def _do_gpio_control(self, params: Dict) -> Dict:
        gpio = self._ensure_gpio()
        if gpio is None:
            return _err("GPIO not available")

        pin = params.get("pin")
        if pin is None:
            return _err("pin parameter required")
        pin = int(pin)

        mode = params.get("mode", "output")  # "output" or "input"
        value = params.get("value")  # True/False for output

        if mode == "input":
            if pin not in self._gpio_pins:
                gpio.setup(pin, gpio.IN)
                self._gpio_pins.add(pin)
            val = gpio.input(pin)
            return _ok({"pin": pin, "mode": "input", "value": bool(val)})
        else:
            if pin not in self._gpio_pins:
                gpio.setup(pin, gpio.OUT)
                self._gpio_pins.add(pin)
            if value is not None:
                gpio.output(pin, gpio.HIGH if value else gpio.LOW)
            return _ok({"pin": pin, "mode": "output",
                         "value": bool(value) if value is not None else None})

    def _do_motor_forward(self, params: Dict) -> Dict:
        gpio = self._ensure_gpio()
        if gpio is None:
            return _err("GPIO not available")

        left_pin = int(params.get("left_pin", 17))
        right_pin = int(params.get("right_pin", 18))
        speed = float(params.get("speed", 100))
        duration = float(params.get("duration", 0))

        self._pwm_pin(left_pin, 1000, speed)
        self._pwm_pin(right_pin, 1000, speed)

        if duration > 0:
            time.sleep(min(duration, 10))
            self._pwm_pin(left_pin, 1000, 0)
            self._pwm_pin(right_pin, 1000, 0)

        return _ok({"action": "motor_forward", "speed": speed,
                     "duration": duration})

    def _do_motor_turn(self, params: Dict) -> Dict:
        gpio = self._ensure_gpio()
        if gpio is None:
            return _err("GPIO not available")

        direction = params.get("direction", "left")
        left_pin = int(params.get("left_pin", 17))
        right_pin = int(params.get("right_pin", 18))
        speed = float(params.get("speed", 70))
        duration = float(params.get("duration", 0))

        if direction == "left":
            self._pwm_pin(left_pin, 1000, 0)
            self._pwm_pin(right_pin, 1000, speed)
        else:
            self._pwm_pin(left_pin, 1000, speed)
            self._pwm_pin(right_pin, 1000, 0)

        if duration > 0:
            time.sleep(min(duration, 10))
            self._pwm_pin(left_pin, 1000, 0)
            self._pwm_pin(right_pin, 1000, 0)

        return _ok({"action": "motor_turn", "direction": direction,
                     "speed": speed, "duration": duration})

    # ── MAVLink actions ───────────────────────────────────────────

    def _do_mavlink_command(self, params: Dict) -> Dict:
        vehicle = self._ensure_vehicle(params)
        if vehicle is None:
            return _err("MAVLink vehicle not connected")
        command = params.get("command", "")
        if not command:
            return _err("command parameter required")
        # Pass-through raw MAVLink mode string
        try:
            import dronekit
            vehicle.mode = dronekit.VehicleMode(command)
            return _ok({"command": command, "mode_set": True})
        except Exception as e:
            return _err(f"mavlink command failed: {e}")

    def _do_takeoff(self, params: Dict) -> Dict:
        vehicle = self._ensure_vehicle(params)
        if vehicle is None:
            return _err("MAVLink vehicle not connected")
        try:
            import dronekit
            altitude = float(params.get("altitude", 5.0))
            vehicle.mode = dronekit.VehicleMode("GUIDED")
            vehicle.armed = True
            deadline = time.time() + 15
            while not vehicle.armed and time.time() < deadline:
                time.sleep(1)
            if not vehicle.armed:
                return _err("arming timed out")
            vehicle.simple_takeoff(altitude)
            return _ok({"action": "takeoff", "altitude": altitude})
        except Exception as e:
            return _err(f"takeoff failed: {e}")

    def _do_land(self, params: Dict) -> Dict:
        vehicle = self._ensure_vehicle(params)
        if vehicle is None:
            return _err("MAVLink vehicle not connected")
        try:
            import dronekit
            vehicle.mode = dronekit.VehicleMode("LAND")
            return _ok({"action": "land"})
        except Exception as e:
            return _err(f"land failed: {e}")

    def _do_return_to_launch(self, params: Dict) -> Dict:
        vehicle = self._ensure_vehicle(params)
        if vehicle is None:
            return _err("MAVLink vehicle not connected")
        try:
            import dronekit
            vehicle.mode = dronekit.VehicleMode("RTL")
            return _ok({"action": "return_to_launch"})
        except Exception as e:
            return _err(f"RTL failed: {e}")

    # ── Hardware init helpers ─────────────────────────────────────

    def _ensure_cv2(self):
        if self._cv2 is not None:
            return self._cv2
        try:
            import cv2
            self._cv2 = cv2
            return cv2
        except ImportError:
            return None

    def _ensure_capture(self):
        if self._cap is not None and self._cap.isOpened():
            return self._cap
        cv2 = self._ensure_cv2()
        if cv2 is None:
            return None
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._cap = None
            return None
        return self._cap

    def _ensure_gpio(self):
        if self._gpio is not None:
            return self._gpio
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            self._gpio = GPIO
            return GPIO
        except (ImportError, RuntimeError):
            return None

    def _ensure_vehicle(self, params: Dict):
        if self._vehicle is not None:
            return self._vehicle
        try:
            import dronekit
            port = params.get("port", "/dev/ttyUSB0")
            baud = int(params.get("baud", 57600))
            self._vehicle = dronekit.connect(port, baud=baud,
                                              wait_ready=True, timeout=30)
            return self._vehicle
        except Exception:
            return None

    def _pwm_pin(self, pin: int, freq: float, duty: float):
        """Set PWM on a pin. Creates PWM instance if needed."""
        gpio = self._gpio
        if gpio is None:
            return
        duty = max(0.0, min(100.0, duty))
        if pin not in self._gpio_pins:
            gpio.setup(pin, gpio.OUT)
            self._gpio_pins.add(pin)
        if pin in self._pwm_instances:
            self._pwm_instances[pin].ChangeDutyCycle(duty)
        else:
            pwm = gpio.PWM(pin, freq)
            pwm.start(duty)
            self._pwm_instances[pin] = pwm

    def _stop_all_motors(self):
        """Stop all PWM outputs."""
        for pwm in self._pwm_instances.values():
            try:
                pwm.ChangeDutyCycle(0)
            except Exception:
                pass

    # ── Cleanup ───────────────────────────────────────────────────

    def close(self):
        """Release all hardware resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
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
        if self._vehicle is not None:
            try:
                self._vehicle.close()
            except Exception:
                pass
            self._vehicle = None
