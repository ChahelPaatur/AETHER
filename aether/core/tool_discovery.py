"""
ToolDiscovery: universal startup capability detection engine for AETHER.

Probes every hardware interface, software package, network endpoint, and
environment variable available on the host system. Generates a structured
capability manifest consumed by the Planner so it never plans with
unavailable tools.

Works on Raspberry Pi, Jetson, Linux, macOS, and Windows.
"""
import glob
import importlib
import os
import platform
import shutil
import socket
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple


# ── Categories ────────────────────────────────────────────────────────

CAT_HARDWARE = "hardware"
CAT_SOFTWARE = "software"
CAT_NETWORK = "network"
CAT_AI = "ai"
CAT_ENVIRONMENT = "environment"


# ── Hardware Probes ───────────────────────────────────────────────────

def _probe_camera_cv2() -> Tuple[bool, Dict]:
    """Try cv2.VideoCapture(0) and read a single frame."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if ret:
                return True, {"type": "cv2", "version": cv2.__version__,
                              "resolution": f"{w}x{h}"}
            cap.release()
        else:
            try:
                cap.release()
            except Exception:
                pass
        # cv2 importable but no camera connected
        return False, {"type": "cv2", "version": cv2.__version__,
                       "detail": "imported but no camera"}
    except ImportError:
        return False, {"detail": "opencv-python not installed"}


def _probe_camera_picamera2() -> Tuple[bool, Dict]:
    """Try picamera2 (Raspberry Pi camera stack v2)."""
    try:
        from picamera2 import Picamera2  # noqa: F401
        return True, {"type": "picamera2"}
    except (ImportError, RuntimeError):
        return False, {"detail": "picamera2 not available"}


def _probe_camera_libcamera() -> Tuple[bool, Dict]:
    """Check for libcamera binary."""
    try:
        import subprocess
        result = subprocess.run(["libcamera-hello", "--list-cameras"],
                                capture_output=True, timeout=5)
        if result.returncode == 0 and b"camera" in result.stdout.lower():
            return True, {"type": "libcamera"}
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return False, {"detail": "libcamera not found"}


def _probe_gpio_rpi() -> Tuple[bool, Dict]:
    """Try RPi.GPIO."""
    try:
        import RPi.GPIO as GPIO  # noqa: F401
        return True, {"type": "RPi.GPIO", "pins": 40}
    except (ImportError, RuntimeError):
        return False, {"detail": "RPi.GPIO not available"}


def _probe_gpio_gpiozero() -> Tuple[bool, Dict]:
    """Try gpiozero."""
    try:
        import gpiozero  # noqa: F401
        return True, {"type": "gpiozero", "version": gpiozero.__version__}
    except (ImportError, RuntimeError):
        return False, {"detail": "gpiozero not available"}


def _probe_i2c() -> Tuple[bool, Dict]:
    """Try board + busio for I2C (Adafruit Blinka)."""
    try:
        import board  # noqa: F401
        import busio  # noqa: F401
        return True, {"type": "blinka_i2c"}
    except (ImportError, RuntimeError, NotImplementedError):
        return False, {"detail": "board/busio not available"}


def _probe_mavlink_serial() -> Tuple[bool, Dict]:
    """Scan serial ports for flight controller at 57600 baud."""
    candidates = []
    if sys.platform == "win32":
        candidates = [f"COM{i}" for i in range(1, 20)]
    else:
        candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        if os.path.exists("/dev/serial0"):
            candidates.append("/dev/serial0")

    for port in candidates:
        try:
            import serial
            conn = serial.Serial(port, 57600, timeout=2)
            # Read a few bytes to see if anything responds
            data = conn.read(8)
            conn.close()
            if len(data) > 0:
                return True, {"type": "serial_mavlink", "port": port,
                              "baud": 57600}
        except (ImportError, OSError, serial.SerialException):
            continue
    return False, {"detail": "no serial flight controller found"}


def _probe_dronekit() -> Tuple[bool, Dict]:
    """Try dronekit import."""
    try:
        import dronekit  # noqa: F401
        return True, {"type": "dronekit", "version": dronekit.__version__}
    except ImportError:
        return False, {"detail": "dronekit not installed"}


def _probe_imu_mpu6050() -> Tuple[bool, Dict]:
    """Try smbus2 read at I2C address 0x68 (MPU6050)."""
    try:
        import smbus2
        bus = smbus2.SMBus(1)
        who_am_i = bus.read_byte_data(0x68, 0x75)
        bus.close()
        return True, {"type": "mpu6050", "address": "0x68",
                      "who_am_i": hex(who_am_i)}
    except (ImportError, OSError, IOError):
        return False, {"detail": "smbus2/MPU6050 not available"}


def _probe_imu_i2c_dev() -> Tuple[bool, Dict]:
    """Check for /dev/i2c-1 existence."""
    if os.path.exists("/dev/i2c-1"):
        return True, {"type": "i2c_device", "path": "/dev/i2c-1"}
    return False, {"detail": "/dev/i2c-1 not found"}


# ── Motor Controller Detection ────────────────────────────────────────
#
# Priority order: MAVLink FC → PCA9685 I2C → GPIO direct → ROS → USB

# MAVLink firmware type IDs
_MAV_AUTOPILOT_NAMES = {
    3: "ArduPilot", 12: "Betaflight", 18: "iNAV",
}
# MAVLink vehicle type IDs
_MAV_TYPE_NAMES = {
    1: "plane", 2: "copter", 10: "rover", 13: "hexarotor",
    14: "octorotor", 20: "vtol",
}
# Known USB motor controller VID:PID
_USB_MOTOR_CONTROLLERS = {
    (0x1209, 0x0D32): "ODrive",
    (0x0483, 0x5740): "VESC",
    (0x03EB, 0x2404): "Roboclaw",
}


def _detect_motor_controllers(serial_ports: List[str]) -> List[Dict]:
    """Run all 5 motor detection probes in priority order.

    Returns a list of detected motor controller dicts, each with:
      controller_type, connection_method, port/address,
      detected_capabilities, suggested_actions, and probe details.
    """
    detected = []
    print("\n  Motor Controller Detection:")

    # ── Detection 1: MAVLink FC on serial ports ───────────────────
    print("    Scanning serial ports...", end="")
    mav_ports = serial_ports[:]
    if not mav_ports:
        # Also try common port patterns
        mav_ports = (glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
                     + glob.glob("/dev/tty.usb*"))
        if sys.platform == "win32":
            mav_ports = [f"COM{i}" for i in range(1, 20)]

    if mav_ports:
        print(f" found {', '.join(mav_ports[:5])}")
    else:
        print(" none found")

    for port in mav_ports:
        mav_result = _probe_mavlink_heartbeat(port)
        if mav_result:
            detected.append(mav_result)
            fw = mav_result.get("firmware", "unknown")
            vtype = mav_result.get("vehicle_type", "unknown")
            ver = mav_result.get("firmware_version", "")
            baud = mav_result.get("baud", "?")
            print(f"    [+] {fw} {vtype} detected"
                  + (f" v{ver}" if ver else ""))
            print(f"        Port: {port} | Baud: {baud}")
            print(f"        Capabilities: {', '.join(mav_result['detected_capabilities'][:6])}")
            break  # Use first detected FC

    # ── Detection 2: PCA9685 servo/motor driver via I2C ───────────
    if not detected:
        pca_result = _probe_pca9685()
        if pca_result:
            detected.append(pca_result)
            print(f"    [+] PCA9685 servo driver detected at I2C {pca_result['address']}")
            print(f"        Channels: {pca_result['channels']} | PWM capable")

    # ── Detection 3: L298N or direct GPIO motors ──────────────────
    if not detected:
        gpio_result = _probe_gpio_motors()
        if gpio_result:
            detected.append(gpio_result)
            print(f"    [+] GPIO motor control available ({gpio_result.get('type', 'RPi.GPIO')})")
            print(f"        PWM pins available for motor control")

    # ── Detection 4: ROS motor controller ─────────────────────────
    ros_result = _probe_ros_motors()
    if ros_result:
        detected.append(ros_result)
        print(f"    [+] ROS navigation stack detected")
        print(f"        Master: {ros_result.get('master_uri', '?')}")

    # ── Detection 5: USB motor controllers (ODrive, VESC, etc.) ───
    if not detected:
        usb_result = _probe_usb_motors()
        if usb_result:
            detected.append(usb_result)
            print(f"    [+] {usb_result['controller_type']} USB motor controller detected")

    if not detected:
        print("    Scanning I2C bus... no motor controllers")
        print("    Scanning USB... no known controllers")
        print("    [-] No motor controller detected")
        print("        To enable motors: connect SpeedyBee via USB, L298N to GPIO,")
        print("        or PCA9685 via I2C")

    return detected


def _probe_mavlink_heartbeat(port: str) -> Optional[Dict]:
    """Attempt MAVLink heartbeat on a serial port at multiple baud rates."""
    try:
        from pymavlink import mavutil
    except ImportError:
        try:
            import serial as _serial
        except ImportError:
            return None
        # Fallback: just test if serial port responds at 115200
        for baud in [115200, 57600, 38400]:
            try:
                conn = _serial.Serial(port, baud, timeout=2)
                data = conn.read(16)
                conn.close()
                if len(data) > 0:
                    return {
                        "controller_type": "mavlink_fc",
                        "connection_method": "serial",
                        "port": port,
                        "baud": baud,
                        "firmware": "unknown (pymavlink not installed)",
                        "vehicle_type": "unknown",
                        "detected_capabilities": [
                            "arm", "disarm", "fly", "hover", "navigate", "RTL"],
                        "suggested_actions": [
                            "takeoff", "land", "return_to_launch",
                            "navigate_waypoint", "set_mode"],
                    }
            except Exception:
                continue
        return None

    # pymavlink available — do proper heartbeat detection
    for baud in [115200, 57600, 38400]:
        try:
            print(f"    Testing MAVLink at {baud} baud on {port}...", end="")
            conn = mavutil.mavlink_connection(port, baud=baud, timeout=3)
            msg = conn.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
            if msg:
                autopilot = _MAV_AUTOPILOT_NAMES.get(msg.autopilot, f"autopilot_{msg.autopilot}")
                vehicle = _MAV_TYPE_NAMES.get(msg.type, f"type_{msg.type}")

                # Try to get firmware version
                fw_version = ""
                conn.mav.request_data_stream_send(
                    conn.target_system, conn.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL, 1, 1)
                ver_msg = conn.recv_match(type="AUTOPILOT_VERSION",
                                          blocking=True, timeout=2)
                if ver_msg:
                    major = (ver_msg.flight_sw_version >> 24) & 0xFF
                    minor = (ver_msg.flight_sw_version >> 16) & 0xFF
                    patch = (ver_msg.flight_sw_version >> 8) & 0xFF
                    fw_version = f"{major}.{minor}.{patch}"

                # Capabilities bitmask
                cap_bits = getattr(msg, "capabilities", 0) if hasattr(msg, "capabilities") else 0

                conn.close()
                print(" HEARTBEAT received")

                capabilities = ["arm", "disarm"]
                if vehicle in ("copter", "hexarotor", "octorotor", "vtol"):
                    capabilities.extend(["fly", "hover", "navigate", "RTL"])
                elif vehicle == "rover":
                    capabilities.extend(["drive", "navigate", "RTL"])
                elif vehicle == "plane":
                    capabilities.extend(["fly", "navigate", "RTL"])

                return {
                    "controller_type": "mavlink_fc",
                    "connection_method": "serial",
                    "port": port,
                    "baud": baud,
                    "firmware": autopilot,
                    "firmware_version": fw_version,
                    "vehicle_type": vehicle,
                    "capabilities_bitmask": cap_bits,
                    "detected_capabilities": capabilities,
                    "suggested_actions": [
                        "takeoff", "land", "return_to_launch",
                        "navigate_waypoint", "set_mode", "arm", "disarm",
                    ],
                }
            conn.close()
            print(" no response")
        except Exception:
            print(" failed")
            continue
    return None


def _probe_pca9685() -> Optional[Dict]:
    """Scan I2C bus for PCA9685 servo driver at address 0x40."""
    try:
        import smbus2
        bus = smbus2.SMBus(1)
        # PCA9685 MODE1 register at address 0x40
        mode1 = bus.read_byte_data(0x40, 0x00)
        bus.close()
        return {
            "controller_type": "pca9685_driver",
            "connection_method": "i2c",
            "address": "0x40",
            "channels": 16,
            "mode1_register": hex(mode1),
            "detected_capabilities": [
                "servo_control", "motor_pwm", "16_channels"],
            "suggested_actions": [
                "motor_forward", "motor_turn", "motor_stop",
                "set_servo", "set_pwm"],
        }
    except (ImportError, OSError, IOError):
        return None


def _probe_gpio_motors() -> Optional[Dict]:
    """Check if GPIO is available for direct motor control."""
    # Try RPi.GPIO
    try:
        import RPi.GPIO as GPIO  # noqa: F401
        return {
            "controller_type": "gpio_direct_motors",
            "connection_method": "gpio",
            "type": "RPi.GPIO",
            "detected_capabilities": ["motor_pwm", "direction_control"],
            "suggested_actions": [
                "motor_forward", "motor_turn", "motor_stop",
                "gpio_control"],
        }
    except (ImportError, RuntimeError):
        pass
    # Try gpiozero
    try:
        import gpiozero  # noqa: F401
        return {
            "controller_type": "gpio_direct_motors",
            "connection_method": "gpio",
            "type": "gpiozero",
            "detected_capabilities": ["motor_pwm", "direction_control"],
            "suggested_actions": [
                "motor_forward", "motor_turn", "motor_stop"],
        }
    except (ImportError, RuntimeError):
        pass
    return None


def _probe_ros_motors() -> Optional[Dict]:
    """Check if ROS motor control is available via /cmd_vel topic."""
    master_uri = os.environ.get("ROS_MASTER_URI", "")
    if not master_uri:
        return None
    try:
        import rospy
        # Don't actually init the node, just check if rospy is importable
        # and ROS_MASTER_URI is set
        return {
            "controller_type": "ros_motors",
            "connection_method": "ros_topic",
            "master_uri": master_uri,
            "topic": "/cmd_vel",
            "detected_capabilities": [
                "velocity_control", "ros_navigation", "tf_transforms"],
            "suggested_actions": [
                "motor_forward", "motor_turn", "motor_stop",
                "navigate_waypoint", "set_velocity"],
        }
    except ImportError:
        return None


def _probe_usb_motors() -> Optional[Dict]:
    """Scan USB devices for known motor controller VID/PID."""
    # Method 1: pyusb
    try:
        import usb.core
        for (vid, pid), name in _USB_MOTOR_CONTROLLERS.items():
            dev = usb.core.find(idVendor=vid, idProduct=pid)
            if dev is not None:
                return {
                    "controller_type": name.lower(),
                    "connection_method": "usb",
                    "vid": hex(vid),
                    "pid": hex(pid),
                    "detected_capabilities": ["motor_pwm", "velocity_control",
                                              "encoder_feedback"],
                    "suggested_actions": [
                        "motor_forward", "motor_turn", "motor_stop",
                        "set_velocity"],
                }
    except (ImportError, Exception):
        pass

    # Method 2: lsusb on Linux/macOS
    if sys.platform != "win32":
        try:
            import subprocess
            result = subprocess.run(["lsusb"], capture_output=True,
                                    text=True, timeout=5)
            if result.returncode == 0:
                for (vid, pid), name in _USB_MOTOR_CONTROLLERS.items():
                    vid_str = f"{vid:04x}"
                    pid_str = f"{pid:04x}"
                    if f"{vid_str}:{pid_str}" in result.stdout.lower():
                        return {
                            "controller_type": name.lower(),
                            "connection_method": "usb",
                            "vid": hex(vid),
                            "pid": hex(pid),
                            "detected_capabilities": [
                                "motor_pwm", "velocity_control"],
                            "suggested_actions": [
                                "motor_forward", "motor_turn", "motor_stop"],
                        }
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

    return None


def _save_motor_config(controllers: List[Dict]) -> Optional[str]:
    """Save detected motor config to configs/auto_detected_motors.json."""
    import json
    import time as _time
    config_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "configs")
    os.makedirs(config_dir, exist_ok=True)
    path = os.path.join(config_dir, "auto_detected_motors.json")
    data = {
        "detected_at": _time.time(),
        "controllers": controllers,
        "primary": controllers[0] if controllers else None,
    }
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path
    except OSError:
        return None


def _probe_audio() -> Tuple[bool, Dict]:
    """Try pyaudio import."""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        input_count = pa.get_device_count()
        pa.terminate()
        return True, {"type": "pyaudio", "devices": input_count}
    except (ImportError, OSError):
        return False, {"detail": "pyaudio not available"}


def _probe_display() -> Tuple[bool, Dict]:
    """Try tkinter (headless-safe)."""
    try:
        import tkinter
        # Attempt to create root — will fail if no display
        root = tkinter.Tk()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return True, {"type": "tkinter", "resolution": f"{w}x{h}"}
    except Exception:
        return False, {"detail": "no display / tkinter unavailable"}


def _probe_storage() -> Tuple[bool, Dict]:
    """Check disk space via shutil."""
    try:
        usage = shutil.disk_usage("/")
        free_gb = round(usage.free / (1024 ** 3), 1)
        total_gb = round(usage.total / (1024 ** 3), 1)
        return True, {"free_gb": free_gb, "total_gb": total_gb}
    except OSError:
        return False, {"detail": "cannot read disk usage"}


def _probe_gpu() -> Tuple[bool, Dict]:
    """Try torch.cuda or torch.backends.mps (Apple Silicon)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, {"type": "cuda", "device": name,
                          "torch_version": torch.__version__}
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, {"type": "mps", "device": "Apple Silicon",
                          "torch_version": torch.__version__}
        # Torch present but no GPU acceleration
        return False, {"detail": f"torch {torch.__version__} (CPU only)"}
    except ImportError:
        return False, {"detail": "torch not installed"}


# ── Network Probes ────────────────────────────────────────────────────

def _probe_internet() -> Tuple[bool, Dict]:
    """Ping 8.8.8.8 on port 53 (DNS)."""
    try:
        sock = socket.create_connection(("8.8.8.8", 53), timeout=3)
        sock.close()
        return True, {"target": "8.8.8.8:53"}
    except (OSError, socket.timeout):
        return False, {"detail": "no internet connectivity"}


def _probe_local_network() -> Tuple[bool, Dict]:
    """Check localhost is reachable."""
    try:
        sock = socket.create_connection(("127.0.0.1", 80), timeout=1)
        sock.close()
        return True, {"detail": "localhost:80 open"}
    except (OSError, socket.timeout):
        # localhost exists even if port 80 isn't open
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            return True, {"hostname": hostname, "ip": ip}
        except socket.error:
            return False, {"detail": "no local network"}


# ── Software Probes ───────────────────────────────────────────────────

_SOFTWARE_PACKAGES = [
    # (import_name, display_name, weight)
    ("numpy",      "NumPy",         15),
    ("scipy",      "SciPy",          5),
    ("sklearn",    "scikit-learn",   5),
    ("cv2",        "OpenCV",        10),
    ("torch",      "PyTorch",       10),
    ("anthropic",  "Anthropic SDK", 12),
    ("dronekit",   "DroneKit",       5),
    ("pymavlink",  "PyMAVLink",      5),
    ("RPi",        "RPi.GPIO",       5),
    ("gpiozero",   "gpiozero",       3),
    ("picamera2",  "picamera2",      3),
    ("psutil",     "psutil",        10),
    ("pyaudio",    "PyAudio",        3),
    ("flask",      "Flask",          3),
    ("requests",   "Requests",       5),
    ("sqlite3",    "SQLite3",        5),
    ("smbus2",     "smbus2",         3),
    ("serial",     "pyserial",       3),
    ("ultralytics","Ultralytics",    5),
]


def _probe_software(import_name: str) -> Tuple[bool, str]:
    """Try importing a package, return (available, version_or_detail)."""
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "OK")
        return True, str(version)
    except (ImportError, ModuleNotFoundError):
        return False, "not installed"


# ── Environment Probes ────────────────────────────────────────────────

_ENV_KEYS = [
    ("ANTHROPIC_API_KEY", "Anthropic API"),
    ("DATABASE_URL",      "Database"),
    ("MQTT_BROKER",       "MQTT Broker"),
    ("ROS_MASTER_URI",    "ROS Master"),
    ("OPENAI_API_KEY",    "OpenAI API"),
]


def _detect_platform() -> Dict:
    """Detect host platform: Pi, Jetson, Linux, macOS, Windows."""
    info: Dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }

    # Raspberry Pi
    model_path = "/proc/device-tree/model"
    if os.path.exists(model_path):
        try:
            with open(model_path, "r") as f:
                model = f.read().strip().rstrip("\x00")
            if "raspberry" in model.lower():
                info["platform"] = "raspberry_pi"
                info["model"] = model
                return info
        except OSError:
            pass

    # Jetson (NVIDIA Tegra)
    if os.path.exists("/etc/nv_tegra_release"):
        try:
            with open("/etc/nv_tegra_release", "r") as f:
                tegra = f.readline().strip()
            info["platform"] = "jetson"
            info["tegra"] = tegra
            return info
        except OSError:
            pass
    # Also check /proc/device-tree/compatible for tegra
    dt_compat = "/proc/device-tree/compatible"
    if os.path.exists(dt_compat):
        try:
            with open(dt_compat, "rb") as f:
                compat = f.read().decode("utf-8", errors="ignore")
            if "tegra" in compat.lower():
                info["platform"] = "jetson"
                return info
        except OSError:
            pass

    system = platform.system().lower()
    if system == "darwin":
        info["platform"] = "macos"
        info["version"] = platform.mac_ver()[0]
    elif system == "windows":
        info["platform"] = "windows"
        info["version"] = platform.version()
    else:
        info["platform"] = "linux"
        # Try to get distro info
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        info["distro"] = line.split("=", 1)[1].strip().strip('"')
                        break
        except OSError:
            pass

    return info


def _detect_serial_ports() -> List[str]:
    """Scan for available serial ports."""
    ports = []
    if sys.platform == "win32":
        for i in range(1, 33):
            port = f"COM{i}"
            try:
                import serial
                s = serial.Serial(port)
                s.close()
                ports.append(port)
            except (ImportError, OSError):
                pass
    else:
        # Linux / macOS
        patterns = ["/dev/ttyUSB*", "/dev/ttyACM*", "/dev/tty.usb*",
                    "/dev/cu.usb*"]
        for pat in patterns:
            ports.extend(glob.glob(pat))
        if os.path.exists("/dev/serial0"):
            ports.append("/dev/serial0")
    return sorted(set(ports))


def _detect_env_keys() -> Dict[str, bool]:
    """Check for known environment variables (value masked)."""
    found = {}
    for key, label in _ENV_KEYS:
        found[key] = bool(os.environ.get(key))
    return found


# ── Capability → Action Mapping ──────────────────────────────────────

_CAPABILITY_ACTION_MAP: Dict[str, List[str]] = {
    # Hardware → actions
    "camera":           ["capture_image", "detect_motion", "visual_inspection"],
    "pi_camera":        ["capture_image", "detect_motion"],
    "libcamera":        ["capture_image"],
    "gpio":             ["hardware_control", "sensor_read"],
    "gpiozero":         ["hardware_control", "sensor_read"],
    "i2c":              ["i2c_read", "i2c_write", "sensor_read"],
    "mavlink":          ["mavlink_command", "drone_telemetry"],
    "dronekit":         ["mavlink_command", "drone_telemetry"],
    "imu_mpu6050":      ["imu_read", "sensor_read"],
    "i2c_imu":          ["imu_read"],
    "microphone":       ["audio_capture"],
    "speaker":          ["audio_playback"],
    "display":          ["display_render"],
    "gpu":              ["drl_inference", "gpu_compute"],
    "storage":          ["file_ops"],
    # Software → actions
    "anthropic":        ["web_search_ai", "summarize_text_ai"],
    "psutil":           ["system_monitor", "resource_monitor", "system_metrics"],
    "numpy":            ["run_simulation"],
    "torch":            ["drl_inference"],
    "ultralytics":      ["yolo_detect"],
    "flask":            ["serve_api"],
    "requests":         ["call_api"],
    "sqlite3":          ["database_query"],
    # Network → actions
    "internet":         ["call_api", "web_search"],
    "local_network":    ["local_api"],
    # Environment → actions
    "ANTHROPIC_API_KEY": ["web_search_ai", "summarize_text_ai"],
}

# Actions always available (stdlib only, or have local fallbacks)
_ALWAYS_AVAILABLE = [
    "read_file", "write_file", "execute_shell",
    "run_simulation", "web_search", "summarize_text",
]


# ── Weight table for scoring ─────────────────────────────────────────

_PROBE_WEIGHTS: Dict[str, int] = {
    "camera": 10, "pi_camera": 5, "libcamera": 3,
    "gpio": 8, "gpiozero": 5, "i2c": 5,
    "mavlink": 8, "dronekit": 5,
    "imu_mpu6050": 6, "i2c_imu": 3,
    "microphone": 4, "speaker": 3, "display": 3,
    "storage": 5, "gpu": 8,
    "internet": 10, "local_network": 3,
    "numpy": 12, "anthropic": 12, "psutil": 8, "torch": 8,
    "scipy": 3, "sklearn": 3, "cv2": 5, "ultralytics": 5,
    "requests": 4, "flask": 2, "sqlite3": 4,
    "ANTHROPIC_API_KEY": 10,
}


# ── Main Discovery Class ─────────────────────────────────────────────

class ToolDiscovery:
    """
    Universal capability discovery engine.
    Runs at startup to detect all available hardware, software, network,
    and environment capabilities. Produces a structured manifest dict
    consumed by the Planner.

    Public API (preserved for backward compatibility):
      discover()                → Dict (manifest)
      manifest                  → Dict property
      score                     → int property
      print_summary()           → None
      is_available(cap)         → bool
      get_available_tool_names()→ List[str]
      filter_planner_actions()  → List[str]
    """

    def __init__(self):
        self._platform: Dict = {}
        self._hardware: Dict[str, Dict] = {}
        self._software: Dict[str, Dict] = {}
        self._network: Dict[str, Dict] = {}
        self._environment: Dict[str, Dict] = {}
        self._serial_ports: List[str] = []
        self._manifest: Dict = {}
        self._score: int = 0
        self._all_caps: Dict[str, Dict] = {}  # key → {available, detail, category, weight}

    def discover(self) -> Dict:
        """Run all probes and build the full capability manifest."""
        self._platform = _detect_platform()
        self._serial_ports = _detect_serial_ports()

        self._probe_hardware()
        self._probe_software()
        self._probe_network()
        self._probe_environment()

        self._compute_score()
        self._manifest = self._build_manifest()
        return self._manifest

    # ── Hardware ──────────────────────────────────────────────────

    def _probe_hardware(self) -> None:
        hw = {}

        # Camera (try multiple backends, take first that works)
        cam_ok, cam_info = _probe_camera_cv2()
        if cam_ok:
            hw["camera"] = {"available": True, **cam_info}
            self._record("camera", True, cam_info, CAT_HARDWARE)
        else:
            # Try picamera2
            pic_ok, pic_info = _probe_camera_picamera2()
            if pic_ok:
                hw["camera"] = {"available": True, **pic_info}
                self._record("pi_camera", True, pic_info, CAT_HARDWARE)
            else:
                # Try libcamera
                lib_ok, lib_info = _probe_camera_libcamera()
                if lib_ok:
                    hw["camera"] = {"available": True, **lib_info}
                    self._record("libcamera", True, lib_info, CAT_HARDWARE)
                else:
                    hw["camera"] = {"available": False,
                                    "detail": "no camera backend"}
                    self._record("camera", False, cam_info, CAT_HARDWARE)

            # Still record cv2 import status even if camera not found
            if not cam_ok and cam_info.get("version"):
                self._record("cv2", True,
                             {"version": cam_info["version"]}, CAT_SOFTWARE)

        # GPIO (try RPi.GPIO, then gpiozero)
        gpio_ok, gpio_info = _probe_gpio_rpi()
        if gpio_ok:
            hw["gpio"] = {"available": True, **gpio_info}
            self._record("gpio", True, gpio_info, CAT_HARDWARE)
        else:
            gz_ok, gz_info = _probe_gpio_gpiozero()
            if gz_ok:
                hw["gpio"] = {"available": True, **gz_info}
                self._record("gpiozero", True, gz_info, CAT_HARDWARE)
            else:
                hw["gpio"] = {"available": False}
                self._record("gpio", False, gpio_info, CAT_HARDWARE)

        # I2C (board + busio)
        i2c_ok, i2c_info = _probe_i2c()
        if i2c_ok:
            hw["i2c"] = {"available": True, **i2c_info}
            self._record("i2c", True, i2c_info, CAT_HARDWARE)
        else:
            hw["i2c"] = {"available": False}
            self._record("i2c", False, i2c_info, CAT_HARDWARE)

        # IMU (try smbus2 MPU6050, then raw /dev/i2c-1)
        imu_ok, imu_info = _probe_imu_mpu6050()
        if imu_ok:
            hw["imu"] = {"available": True, **imu_info}
            self._record("imu_mpu6050", True, imu_info, CAT_HARDWARE)
        else:
            dev_ok, dev_info = _probe_imu_i2c_dev()
            if dev_ok:
                hw["imu"] = {"available": True, **dev_info}
                self._record("i2c_imu", True, dev_info, CAT_HARDWARE)
            else:
                hw["imu"] = {"available": False}
                self._record("imu_mpu6050", False, imu_info, CAT_HARDWARE)

        # Motor controller auto-detection (replaces simple mavlink probe)
        motor_controllers = _detect_motor_controllers(self._serial_ports)
        self._motor_controllers = motor_controllers

        # Determine mavlink status from motor detection results
        mavlink_ctrl = next((c for c in motor_controllers
                             if c["controller_type"] == "mavlink_fc"), None)
        if mavlink_ctrl:
            hw["mavlink"] = {"available": True, **mavlink_ctrl}
            self._record("mavlink", True, mavlink_ctrl, CAT_HARDWARE)
        else:
            # Fallback: try dronekit import
            dk_ok, dk_info = _probe_dronekit()
            if dk_ok:
                hw["mavlink"] = {"available": True, **dk_info}
                self._record("dronekit", True, dk_info, CAT_HARDWARE)
            else:
                hw["mavlink"] = {"available": False}
                self._record("mavlink", False,
                             {"detail": "no flight controller detected"},
                             CAT_HARDWARE)

        # Record motor controllers detected (may include non-mavlink types)
        motor_ctrl = next((c for c in motor_controllers
                           if c["controller_type"] != "mavlink_fc"), None)
        if motor_ctrl:
            hw["motor_controller"] = {"available": True, **motor_ctrl}
            self._record("motor_controller", True, motor_ctrl, CAT_HARDWARE)
        elif not mavlink_ctrl:
            hw["motor_controller"] = {"available": False}
            self._record("motor_controller", False,
                         {"detail": "no motor controller detected"},
                         CAT_HARDWARE)

        # Save motor config for ToolBuilder to read
        if motor_controllers:
            config_path = _save_motor_config(motor_controllers)
            if config_path:
                print(f"    Motor config saved to {config_path}")

        # Audio
        aud_ok, aud_info = _probe_audio()
        hw["audio"] = {"available": aud_ok, **(aud_info if aud_ok else {})}
        if aud_ok:
            self._record("microphone", True, aud_info, CAT_HARDWARE)
            self._record("speaker", True, aud_info, CAT_HARDWARE)
        else:
            self._record("microphone", False, aud_info, CAT_HARDWARE)

        # Display
        disp_ok, disp_info = _probe_display()
        hw["display"] = {"available": disp_ok, **(disp_info if disp_ok else {})}
        self._record("display", disp_ok, disp_info, CAT_HARDWARE)

        # Storage
        stor_ok, stor_info = _probe_storage()
        hw["storage"] = {"available": stor_ok, **(stor_info if stor_ok else {})}
        self._record("storage", stor_ok, stor_info, CAT_HARDWARE)

        # GPU
        gpu_ok, gpu_info = _probe_gpu()
        hw["gpu"] = {"available": gpu_ok, **(gpu_info if gpu_ok else {})}
        self._record("gpu", gpu_ok, gpu_info, CAT_HARDWARE)

        self._hardware = hw

    # ── Software ──────────────────────────────────────────────────

    def _probe_software(self) -> None:
        sw = {}
        for import_name, display_name, weight in _SOFTWARE_PACKAGES:
            ok, detail = _probe_software(import_name)
            sw[import_name] = ok
            # Only record if not already recorded by hardware probe
            if import_name not in self._all_caps:
                self._record(import_name, ok,
                             {"version": detail} if ok else {"detail": detail},
                             CAT_SOFTWARE, weight=weight)
        self._software = sw

    # ── Network ───────────────────────────────────────────────────

    def _probe_network(self) -> None:
        net = {}

        inet_ok, inet_info = _probe_internet()
        net["internet"] = inet_ok
        self._record("internet", inet_ok, inet_info, CAT_NETWORK)

        local_ok, local_info = _probe_local_network()
        net["local"] = local_ok
        self._record("local_network", local_ok, local_info, CAT_NETWORK)

        self._network = net

    # ── Environment ───────────────────────────────────────────────

    def _probe_environment(self) -> None:
        env_keys = _detect_env_keys()
        self._environment = {}
        for key, present in env_keys.items():
            self._environment[key] = present
            # Special: ANTHROPIC_API_KEY gates AI tools
            self._record(key, present,
                         {"set": True} if present else {"detail": "not set"},
                         CAT_ENVIRONMENT)

    # ── Internal helpers ──────────────────────────────────────────

    def _record(self, key: str, available: bool, info: Dict,
                category: str, weight: int = 0) -> None:
        """Record a probe result into the unified capabilities dict."""
        if not weight:
            weight = _PROBE_WEIGHTS.get(key, 3)
        self._all_caps[key] = {
            "available": available,
            "info": info,
            "category": category,
            "weight": weight,
        }

    def _compute_score(self) -> None:
        """Weighted capability score 0-100."""
        total = 0
        earned = 0
        for cap in self._all_caps.values():
            total += cap["weight"]
            if cap["available"]:
                earned += cap["weight"]
        self._score = round(earned / total * 100) if total else 0

    def _build_manifest(self) -> Dict:
        """Build the structured manifest dict."""
        available_caps = [k for k, v in self._all_caps.items() if v["available"]]
        missing_caps = [k for k, v in self._all_caps.items() if not v["available"]]

        # Resolve available actions from capabilities
        actions = set(_ALWAYS_AVAILABLE)
        for cap in available_caps:
            for action in _CAPABILITY_ACTION_MAP.get(cap, []):
                actions.add(action)

        # Resolve unavailable actions (have a mapping but no capability)
        all_mapped_actions = set()
        for action_list in _CAPABILITY_ACTION_MAP.values():
            all_mapped_actions.update(action_list)
        unavailable_actions = sorted(all_mapped_actions - actions)

        # Add motor controller actions to available tools
        motor_ctrls = getattr(self, "_motor_controllers", [])
        for ctrl in motor_ctrls:
            for act in ctrl.get("suggested_actions", []):
                actions.add(act)

        return {
            "platform": self._platform.get("platform", "unknown"),
            "platform_info": self._platform,
            "hardware": self._hardware,
            "software": self._software,
            "network": self._network,
            "environment": self._environment,
            "serial_ports": self._serial_ports,
            "motor_controllers": motor_ctrls,
            "available_capabilities": available_caps,
            "missing_capabilities": missing_caps,
            "available_tools": sorted(actions),
            "unavailable_tools": unavailable_actions,
            "capability_score": self._score,
            "hardware_detected": [
                k for k, v in self._all_caps.items()
                if v["category"] == CAT_HARDWARE and v["available"]
            ],
            "hardware_missing": [
                k for k, v in self._all_caps.items()
                if v["category"] == CAT_HARDWARE and not v["available"]
            ],
        }

    # ── Public API ────────────────────────────────────────────────

    @property
    def manifest(self) -> Dict:
        if not self._manifest:
            self.discover()
        return self._manifest

    @property
    def score(self) -> int:
        if not self._manifest:
            self.discover()
        return self._score

    def get_manifest(self) -> Dict:
        """Return the full capability manifest (runs discovery if needed)."""
        return self.manifest

    def get_available_tool_names(self) -> List[str]:
        """Return list of action names that have all dependencies met."""
        return self.manifest.get("available_tools", [])

    def is_available(self, capability: str) -> bool:
        """Check if a specific capability is available."""
        cap = self._all_caps.get(capability)
        return cap["available"] if cap else False

    def filter_planner_actions(self, actions: List[str]) -> List[str]:
        """Remove actions that depend on unavailable capabilities."""
        unavail = set(self.manifest.get("unavailable_tools", []))
        return [a for a in actions if a not in unavail]

    def print_summary(self) -> None:
        """Print a clean startup capability summary."""
        if not self._manifest:
            self.discover()

        plat = self._platform
        plat_label = plat.get("platform", "unknown")
        model = plat.get("model", plat.get("distro", plat.get("version", "")))

        print("=" * 60)
        print("AETHER v3 — Universal Capability Discovery")
        print("=" * 60)
        print(f"  Platform: {plat_label}"
              + (f" ({model})" if model else ""))
        print(f"  Python:   {plat.get('python', '?')} | "
              f"Arch: {plat.get('machine', '?')}")

        # Hardware
        print(f"\n  Hardware:")
        hw_order = ["camera", "gpio", "i2c", "imu", "mavlink",
                    "audio", "display", "storage", "gpu"]
        for key in hw_order:
            info = self._hardware.get(key, {})
            avail = info.get("available", False)
            icon = "+" if avail else "-"
            status = "OK" if avail else "---"
            detail = ""
            if avail:
                hw_type = info.get("type", "")
                resolution = info.get("resolution", "")
                if hw_type:
                    detail = hw_type
                if resolution:
                    detail += f" {resolution}"
                if key == "storage":
                    detail = f"{info.get('free_gb', '?')}GB free / {info.get('total_gb', '?')}GB"
                if key == "gpu":
                    detail = info.get("device", info.get("type", ""))
            else:
                detail = info.get("detail", "not detected")
            print(f"    [{icon}] {key:<14} {status:<6} {detail}")

        # Software
        print(f"\n  Software:")
        for import_name, display_name, _ in _SOFTWARE_PACKAGES:
            avail = self._software.get(import_name, False)
            icon = "+" if avail else "-"
            status = "OK" if avail else "---"
            cap = self._all_caps.get(import_name, {})
            ver = cap.get("info", {}).get("version", "") if avail else ""
            print(f"    [{icon}] {display_name:<16} {status:<6} {ver}")

        # Network
        print(f"\n  Network:")
        inet = self._network.get("internet", False)
        local = self._network.get("local", False)
        print(f"    [{'+' if inet else '-'}] {'Internet':<16} "
              f"{'OK' if inet else '---':<6}")
        print(f"    [{'+' if local else '-'}] {'Local Network':<16} "
              f"{'OK' if local else '---':<6}")

        # Environment
        env_set = [k for k, v in self._environment.items() if v]
        env_missing = [k for k, v in self._environment.items() if not v]
        if env_set or env_missing:
            print(f"\n  Environment:")
            for key in env_set:
                label = dict(_ENV_KEYS).get(key, key)
                print(f"    [+] {label:<20} set")
            for key in env_missing:
                label = dict(_ENV_KEYS).get(key, key)
                print(f"    [-] {label:<20} not set")

        # Serial ports
        if self._serial_ports:
            print(f"\n  Serial Ports: {', '.join(self._serial_ports)}")

        # Available actions
        avail = self._manifest.get("available_tools", [])
        unavail = self._manifest.get("unavailable_tools", [])
        print(f"\n  Actions: {len(avail)} available, {len(unavail)} unavailable")
        if avail:
            # Wrap action names at 60 chars
            line = "    "
            for a in avail:
                if len(line) + len(a) + 2 > 60:
                    print(line)
                    line = "    "
                line += a + ", "
            if line.strip():
                print(line.rstrip(", "))
        if unavail:
            print(f"    Unavailable: {', '.join(unavail[:10])}"
                  + ("..." if len(unavail) > 10 else ""))

        # Score bar
        score = self._score
        bar_len = 25
        filled = round(score / 100 * bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\n  Capability Score: [{bar}] {score}/100")
        print("=" * 60)
