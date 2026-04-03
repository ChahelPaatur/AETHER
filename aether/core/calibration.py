"""
CalibrationWizard: interactive hardware calibration for AETHER.

Runs when new hardware is detected that hasn't been calibrated before.
Determines what every connected component physically does through a
combination of user prompts, low-power test movements, and optional
camera-based visual feedback.

Phases:
  1. Component Identification — ask user what type of robot
  2. Motor Mapping — test each channel and record what moved
  3. Camera-Assisted Calibration — verify with optical flow
  4. Safety Limits — set per-component limits
  5. Capability Generation — save robot profile JSON
  6. Action Generation — build correct action set for robot type
  7. Re-calibration & Learning — runtime drift detection

CLI flags:
  --calibrate         run full calibration wizard
  --recalibrate       force re-calibration even if profile exists
  --calibrate --auto  auto-calibrate using camera only, no questions
"""
import json
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")

# Robot type constants
GROUND_VEHICLE = "ground_vehicle"
AERIAL_VEHICLE = "aerial_vehicle"
ROBOTIC_ARM = "robotic_arm"
AQUATIC_VEHICLE = "aquatic_vehicle"
CUSTOM = "custom"

# Robot type map from menu choice
_ROBOT_TYPE_MAP = {
    1: GROUND_VEHICLE,
    2: AERIAL_VEHICLE,
    3: ROBOTIC_ARM,
    4: AQUATIC_VEHICLE,
    5: CUSTOM,
}

# Capabilities generated per robot type
_TYPE_CAPABILITIES = {
    GROUND_VEHICLE: [
        "move_forward", "move_backward", "turn_left", "turn_right",
        "navigate_to_color", "avoid_obstacle", "follow_object", "stop",
    ],
    AERIAL_VEHICLE: [
        "takeoff", "land", "hover", "navigate_waypoint",
        "return_to_launch", "visual_scan", "stop",
    ],
    ROBOTIC_ARM: [
        "move_joint", "grip", "release", "point_to",
        "home_position", "pick_and_place", "stop",
    ],
    AQUATIC_VEHICLE: [
        "move_forward", "move_backward", "turn_left", "turn_right",
        "dive", "surface", "stop",
    ],
    CUSTOM: ["stop"],
}

# Frame types for drones
_DRONE_FRAMES = {
    1: "quadcopter",
    2: "hexacopter",
    3: "octocopter",
    4: "fixed_wing",
}

# Ground vehicle drive types
_DRIVE_TYPES = {
    1: ("differential", 2),
    2: ("skid_steer", 4),
    3: ("other", None),
}


def _input_int(prompt: str, valid_range: Optional[range] = None) -> int:
    """Prompt user for an integer, retry until valid."""
    while True:
        try:
            val = int(input(prompt))
            if valid_range is not None and val not in valid_range:
                print(f"  Please enter a value between {valid_range.start} "
                      f"and {valid_range.stop - 1}.")
                continue
            return val
        except (ValueError, EOFError):
            print("  Please enter a valid integer.")


def _input_float(prompt: str, lo: float = 0.0, hi: float = 1.0) -> float:
    """Prompt user for a float within [lo, hi]."""
    while True:
        try:
            val = float(input(prompt))
            if val < lo or val > hi:
                print(f"  Please enter a value between {lo} and {hi}.")
                continue
            return val
        except (ValueError, EOFError):
            print("  Please enter a valid number.")


def _input_yn(prompt: str, default: bool = True) -> bool:
    """Prompt user for Y/n."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        ans = input(f"{prompt} {suffix}: ").strip().lower()
    except EOFError:
        return default
    if ans == "":
        return default
    return ans in ("y", "yes")


def _find_existing_profile(configs_dir: str) -> Optional[str]:
    """Return path of the most recent calibration profile, or None."""
    os.makedirs(configs_dir, exist_ok=True)
    profiles = [
        f for f in os.listdir(configs_dir)
        if f.startswith("calibrated_") and f.endswith(".json")
    ]
    if not profiles:
        return None
    profiles.sort()
    return os.path.join(configs_dir, profiles[-1])


def load_calibration(configs_dir: Optional[str] = None) -> Optional[Dict]:
    """Load the most recent calibration profile if it exists."""
    configs_dir = configs_dir or os.path.abspath(_CONFIGS_DIR)
    path = _find_existing_profile(configs_dir)
    if path is None:
        return None
    try:
        with open(path) as f:
            profile = json.load(f)
        if profile.get("calibrated"):
            return profile
    except (json.JSONDecodeError, OSError, KeyError):
        pass
    return None


class CalibrationWizard:
    """Interactive hardware calibration wizard for AETHER.

    Determines what every connected component physically does through
    a combination of user interaction, low-power motor tests, and
    optional camera-based optical flow verification.
    """

    def __init__(
        self,
        manifest: Dict,
        configs_dir: Optional[str] = None,
        auto_mode: bool = False,
        adapter: Optional[Any] = None,
    ):
        self.manifest = manifest
        self.configs_dir = os.path.abspath(configs_dir or _CONFIGS_DIR)
        self.auto_mode = auto_mode
        self.adapter = adapter

        self._hw = manifest.get("hardware", {})
        self._has_camera = self._hw.get("camera", {}).get("available", False)
        self._has_gpio = self._hw.get("gpio", {}).get("available", False)
        self._has_mavlink = self._hw.get("mavlink", {}).get("available", False)

        self.robot_type: Optional[str] = None
        self.robot_name: str = "unnamed"
        self.robot_purpose: str = ""
        self.motor_channels: List[str] = []
        self.motor_map: Dict[str, Dict] = {}
        self.safety_limits: Dict[str, Any] = {}
        self.components: Dict[str, Dict] = {}
        self.capabilities: List[str] = []
        self.camera_verified: bool = False
        self.profile: Dict = {}

    # ── Public API ────────────────────────────────────────────────

    def needs_calibration(self, force: bool = False) -> bool:
        """Return True if calibration is needed."""
        if force:
            return True
        existing = load_calibration(self.configs_dir)
        if existing is not None:
            return False
        # New hardware detected with no profile
        return self._has_gpio or self._has_mavlink

    def run(self) -> Dict:
        """Run the full calibration wizard. Returns the saved profile."""
        self._detect_channels()

        if not self.motor_channels:
            print("[Calibration] No motor/actuator channels detected.")
            print("[Calibration] Generating camera-only profile.")
            self.robot_type = CUSTOM
            self.robot_name = "camera_only"
            self.camera_verified = self._has_camera
            self.capabilities = ["visual_scan", "detect_color",
                                 "report_surroundings", "stop"]
            return self._save_profile()

        n = len(self.motor_channels)
        print(f"\n[Calibration] New hardware detected: "
              f"{n} motor channel(s) via "
              f"{'GPIO' if self._has_gpio else 'MAVLink'}")

        # Phase 1
        self._phase1_identification()
        # Phase 2
        self._phase2_motor_mapping()
        # Phase 3
        if self._has_camera:
            self._phase3_camera_verification()
            # Environment mapping: build occupancy grid after motor mapping
            self._run_environment_mapping()
        # Phase 4
        self._phase4_safety_limits()
        # Semantic understanding
        self._ask_purpose()
        # Phase 5
        profile = self._save_profile()
        # Phase 6 — action generation happens on load
        print(f"\n[Calibration] Complete! Profile saved.")
        print(f"[Calibration] Robot: {self.robot_name} ({self.robot_type})")
        print(f"[Calibration] Capabilities: {', '.join(self.capabilities)}")
        return profile

    # ── Phase 1: Component Identification ─────────────────────────

    def _phase1_identification(self) -> None:
        print("\n" + "=" * 50)
        print("PHASE 1 — Component Identification")
        print("=" * 50)

        if self.auto_mode:
            self.robot_type = self._auto_detect_type()
            print(f"[Calibration] Auto-detected type: {self.robot_type}")
        else:
            print("What type of robot is this?")
            print("  1. Ground vehicle (wheels/tracks)")
            print("  2. Aerial vehicle (drone/quadcopter)")
            print("  3. Robotic arm / manipulator")
            print("  4. Boat / aquatic vehicle")
            print("  5. Custom / unknown")
            choice = _input_int("Enter choice (1-5): ", range(1, 6))
            self.robot_type = _ROBOT_TYPE_MAP[choice]

        # Follow-up questions based on type (skip in auto mode)
        if not self.auto_mode:
            if self.robot_type == GROUND_VEHICLE:
                self._ask_ground_details()
            elif self.robot_type == AERIAL_VEHICLE:
                self._ask_aerial_details()
            elif self.robot_type == ROBOTIC_ARM:
                self._ask_arm_details()
            elif self.robot_type == AQUATIC_VEHICLE:
                self._ask_aquatic_details()
        else:
            # Set sensible defaults for auto mode
            if self.robot_type == GROUND_VEHICLE:
                self.profile["drive_type"] = "differential"
            elif self.robot_type == AERIAL_VEHICLE:
                self.profile["frame_type"] = "quadcopter"
            elif self.robot_type == ROBOTIC_ARM:
                n = len(self.motor_channels)
                self.profile["joint_count"] = n
                self.profile["joint_ranges"] = {
                    f"joint_{i+1}": {"min": -90, "max": 90}
                    for i in range(n)
                }
            elif self.robot_type == AQUATIC_VEHICLE:
                self.profile["thruster_count"] = len(self.motor_channels)

        name_default = f"{self.robot_type.replace('_', '')}_v1"
        if self.auto_mode:
            self.robot_name = name_default
        else:
            try:
                name = input(f"Robot name [{name_default}]: ").strip()
            except EOFError:
                name = ""
            self.robot_name = name or name_default

    def _ask_ground_details(self) -> None:
        print("\nHow many drive motors?")
        print("  1. Two motors (differential drive)")
        print("  2. Four motors (skid steer)")
        print("  3. Other")
        choice = _input_int("Enter choice (1-3): ", range(1, 4))
        drive_type, expected_count = _DRIVE_TYPES[choice]
        self.profile["drive_type"] = drive_type
        if expected_count:
            self.profile["expected_motors"] = expected_count

    def _ask_aerial_details(self) -> None:
        print("\nWhat frame type?")
        print("  1. Quadcopter (X or +)")
        print("  2. Hexacopter")
        print("  3. Octocopter")
        print("  4. Fixed wing")
        choice = _input_int("Enter choice (1-4): ", range(1, 5))
        self.profile["frame_type"] = _DRONE_FRAMES[choice]

    def _ask_arm_details(self) -> None:
        n_joints = _input_int("\nHow many joints/servos? Enter number: ",
                              range(1, 20))
        self.profile["joint_count"] = n_joints
        self.profile["joint_ranges"] = {}
        for j in range(1, n_joints + 1):
            print(f"\n  Joint {j} range of motion:")
            min_deg = _input_int(f"    Min degrees: ")
            max_deg = _input_int(f"    Max degrees: ")
            self.profile["joint_ranges"][f"joint_{j}"] = {
                "min": min_deg, "max": max_deg,
            }

    def _ask_aquatic_details(self) -> None:
        n_thrusters = _input_int("\nHow many thrusters? Enter number: ",
                                 range(1, 12))
        self.profile["thruster_count"] = n_thrusters

    # ── Phase 2: Motor Mapping ────────────────────────────────────

    def _phase2_motor_mapping(self) -> None:
        print("\n" + "=" * 50)
        print("PHASE 2 — Motor Mapping")
        print("=" * 50)

        if self.robot_type == ROBOTIC_ARM:
            self._map_arm_motors()
        elif self.robot_type == AERIAL_VEHICLE:
            self._map_aerial_motors()
        else:
            self._map_generic_motors()

    def _map_generic_motors(self) -> None:
        """Map motors for ground/aquatic/custom vehicles."""
        wheel_options = [
            "Left wheel forward",
            "Right wheel forward",
            "Left wheel backward",
            "Right wheel backward",
            "Nothing moved",
            "Something else moved (describe)",
        ]
        for ch in self.motor_channels:
            print(f"\n[Calibration] Testing {ch} at 10% power "
                  f"for 0.5 seconds...")
            self._test_channel(ch, power=0.1, duration=0.5)

            if self.auto_mode:
                mapping = self._auto_map_channel(ch)
            else:
                print("\nWhat moved?")
                for i, opt in enumerate(wheel_options, 1):
                    print(f"  {i}. {opt}")
                choice = _input_int("Enter choice (1-6): ", range(1, 7))

                if choice == 6:
                    try:
                        desc = input("  Describe what moved: ").strip()
                    except EOFError:
                        desc = "unknown"
                    mapping = {"function": desc, "direction": "unknown",
                               "confirmed": True}
                elif choice == 5:
                    mapping = {"function": "disconnected",
                               "direction": "none", "confirmed": False}
                else:
                    funcs = [
                        ("left_wheel", "forward"),
                        ("right_wheel", "forward"),
                        ("left_wheel", "backward"),
                        ("right_wheel", "backward"),
                    ]
                    func, direction = funcs[choice - 1]
                    mapping = {"function": func, "direction": direction,
                               "confirmed": True}

            self.motor_map[ch] = mapping
            print(f"  Mapped: {ch} → {mapping['function']} "
                  f"({mapping.get('direction', '')})")

    def _map_arm_motors(self) -> None:
        """Map servos for robotic arm."""
        joint_names = [
            "base_rotation", "shoulder", "elbow", "wrist_pitch",
            "wrist_roll", "gripper",
        ]
        joint_ranges = self.profile.get("joint_ranges", {})

        for i, ch in enumerate(self.motor_channels):
            default_name = (joint_names[i]
                            if i < len(joint_names) else f"joint_{i+1}")
            jr_key = f"joint_{i+1}"
            jr = joint_ranges.get(jr_key, {"min": -90, "max": 90})

            print(f"\n[Calibration] Testing {ch} — small movement test...")
            self._test_channel(ch, power=0.05, duration=0.3)

            if self.auto_mode:
                name = default_name
            else:
                try:
                    name = input(f"  Joint name [{default_name}]: ").strip()
                except EOFError:
                    name = ""
                name = name or default_name

                if jr_key not in joint_ranges:
                    jr["min"] = _input_int(f"  Min angle (degrees) [{jr['min']}]: ")
                    jr["max"] = _input_int(f"  Max angle (degrees) [{jr['max']}]: ")

            self.motor_map[ch] = {
                "function": name,
                "min_angle": jr["min"],
                "max_angle": jr["max"],
            }
            self.components[name] = {
                "channel": i + 1,
                "type": "servo",
                "min": jr["min"],
                "max": jr["max"],
            }
            print(f"  Mapped: {ch} → {name} [{jr['min']}°..{jr['max']}°]")

    def _map_aerial_motors(self) -> None:
        """Map motors for aerial vehicle (auto-assign by position)."""
        positions = ["front_left", "front_right",
                     "rear_left", "rear_right",
                     "mid_left", "mid_right",
                     "extra_1", "extra_2"]
        for i, ch in enumerate(self.motor_channels):
            pos = positions[i] if i < len(positions) else f"motor_{i+1}"

            if not self.auto_mode:
                print(f"\n[Calibration] Testing {ch} — brief spin...")
                self._test_channel(ch, power=0.05, duration=0.3)
                try:
                    custom = input(f"  Motor position [{pos}]: ").strip()
                except EOFError:
                    custom = ""
                pos = custom or pos

            self.motor_map[ch] = {
                "function": pos,
                "type": "brushless",
                "confirmed": True,
            }
            self.components[pos] = {
                "channel": i + 1,
                "type": "brushless",
            }
            print(f"  Mapped: {ch} → {pos}")

    # ── Phase 3: Camera-Assisted Calibration ──────────────────────

    def _phase3_camera_verification(self) -> None:
        print("\n" + "=" * 50)
        print("PHASE 3 — Camera-Assisted Verification")
        print("=" * 50)

        cv2 = self._get_cv2()
        if cv2 is None:
            print("[Calibration] Camera library not available, skipping.")
            return

        try:
            from aether.core.tool_builder import _ON_PI
        except ImportError:
            _ON_PI = False

        if _ON_PI:
            from aether.core.tool_builder import _capture_frame_any
            # Use Pi singleton wrapper instead of cv2.VideoCapture
            class _PiCapWrap:
                def isOpened(self):
                    return True
                def read(self):
                    frame, backend = _capture_frame_any()
                    return (frame is not None), frame
                def release(self):
                    pass
            cap = _PiCapWrap()
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[Calibration] Camera could not be opened, skipping.")
                return

        verified_count = 0
        for ch, mapping in self.motor_map.items():
            if mapping.get("function") == "disconnected":
                continue

            print(f"\n[Calibration] Testing {ch} at 10% power...")

            # Capture before frame
            ret, before = cap.read()
            if not ret:
                print("  Could not capture before-frame, skipping.")
                continue
            before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
            before_gray = cv2.GaussianBlur(before_gray, (21, 21), 0)

            # Run the motor
            self._test_channel(ch, power=0.1, duration=0.5)

            # Capture after frame
            ret, after = cap.read()
            if not ret:
                print("  Could not capture after-frame, skipping.")
                continue
            after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
            after_gray = cv2.GaussianBlur(after_gray, (21, 21), 0)

            # Frame differencing
            delta = cv2.absdiff(before_gray, after_gray)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_px = np.count_nonzero(thresh)
            total_px = thresh.shape[0] * thresh.shape[1]
            motion_score = motion_px / total_px if total_px > 0 else 0.0

            # Determine motion region
            h, w = thresh.shape
            left_motion = np.count_nonzero(thresh[:, :w // 2])
            right_motion = np.count_nonzero(thresh[:, w // 2:])
            region = ("left side" if left_motion > right_motion * 1.5
                      else "right side" if right_motion > left_motion * 1.5
                      else "center/full")

            detected = motion_score > 0.005
            status = "YES" if detected else "NO"
            print(f"  Camera detected motion: {status} "
                  f"(motion score: {motion_score:.2f})")

            if detected:
                print(f"  Motion region: {region} of frame")
                suggested = self._suggest_mapping(mapping, region)
                print(f"  Suggested mapping: {suggested}")

                if self.auto_mode:
                    accept = True
                else:
                    accept = _input_yn("  Accept?")

                if accept:
                    mapping["camera_verified"] = True
                    mapping["motion_score"] = round(motion_score, 4)
                    mapping["motion_region"] = region
                    verified_count += 1
                else:
                    mapping["camera_verified"] = False
            else:
                print("  WARNING: No motion detected — component may be "
                      "disconnected.")
                mapping["camera_verified"] = False
                if not mapping.get("confirmed", False):
                    mapping["function"] = "disconnected"

        cap.release()
        self.camera_verified = verified_count > 0
        print(f"\n[Calibration] Camera verified {verified_count}/"
              f"{len(self.motor_map)} channels.")

    # ── Environment Mapping (post-Phase 3) ──────────────────────

    def _run_environment_mapping(self) -> None:
        """Build occupancy grid after motor mapping + camera verification."""
        if not self.motor_map:
            return
        try:
            from aether.core.mapper import EnvironmentMapper

            # Build a lightweight camera wrapper
            cv2 = self._get_cv2()
            if cv2 is None:
                return

            class _CamWrap:
                def __init__(self, cv2_mod):
                    self._cv2 = cv2_mod

                def capture_frame(self):
                    if _ON_PI:
                        frame, backend = _capture_frame_any()
                        return frame
                    cap = self._cv2.VideoCapture(0)
                    ret, frame = cap.read()
                    cap.release()
                    return frame if ret else None

            # Build a motor function from adapter or stub
            def _motor_fn(action: str, duration: float):
                if self.adapter and hasattr(self.adapter, "execute"):
                    self.adapter.execute(action, duration=duration)

            mapper = EnvironmentMapper(
                camera_tool=_CamWrap(cv2),
                motor_fn=_motor_fn,
                configs_dir=self.configs_dir,
            )
            print("\n[Calibration] Building environment map...")
            result = mapper.run()
            print(f"[Calibration] Map saved — "
                  f"{result['waypoints']} waypoints, "
                  f"{result['occupied_cells']} occupied cells")
        except Exception as e:
            print(f"[Calibration] Environment mapping skipped: {e}")

    # ── Phase 4: Safety Limits ────────────────────────────────────

    def _phase4_safety_limits(self) -> None:
        print("\n" + "=" * 50)
        print("PHASE 4 — Safety Limits")
        print("=" * 50)

        if self.robot_type == ROBOTIC_ARM:
            self._arm_safety_limits()
        elif self.robot_type == AERIAL_VEHICLE:
            self._aerial_safety_limits()
        else:
            self._generic_safety_limits()

    def _generic_safety_limits(self) -> None:
        for ch, mapping in self.motor_map.items():
            func = mapping.get("function", "unknown")
            if func == "disconnected":
                continue

            print(f"\n[Calibration] Setting safety limits for {func}")
            if self.auto_mode:
                self.safety_limits[func] = {
                    "max_speed": 0.7,
                    "max_continuous_seconds": 30,
                    "emergency_stop": "immediate_cutoff",
                }
            else:
                max_speed = _input_float(
                    "  Maximum safe speed (0.0-1.0, recommended 0.7): ",
                    0.0, 1.0)
                max_time = _input_float(
                    "  Maximum continuous run time (seconds, max 120): ",
                    0.0, 120.0)
                e_stop = _input_yn("  Emergency stop response: immediate "
                                   "cutoff?")
                self.safety_limits[func] = {
                    "max_speed": max_speed,
                    "max_continuous_seconds": max_time,
                    "emergency_stop": ("immediate_cutoff"
                                       if e_stop else "gradual_stop"),
                }

    def _arm_safety_limits(self) -> None:
        for ch, mapping in self.motor_map.items():
            func = mapping.get("function", "unknown")
            print(f"\n[Calibration] {func} limits")
            if self.auto_mode:
                self.safety_limits[func] = {
                    "min_angle": mapping.get("min_angle", -90),
                    "max_angle": mapping.get("max_angle", 90),
                    "max_speed_dps": 45,
                    "hard_stop": True,
                }
            else:
                min_a = _input_int(f"  Minimum angle (degrees) "
                                   f"[{mapping.get('min_angle', -90)}]: ")
                max_a = _input_int(f"  Maximum angle (degrees) "
                                   f"[{mapping.get('max_angle', 90)}]: ")
                max_spd = _input_float(
                    "  Maximum speed (degrees/second, max 360): ", 0.0, 360.0)
                hard = _input_yn("  Hard stop detected?")
                self.safety_limits[func] = {
                    "min_angle": min_a,
                    "max_angle": max_a,
                    "max_speed_dps": max_spd,
                    "hard_stop": hard,
                }

    def _aerial_safety_limits(self) -> None:
        print("\n[Calibration] Aerial vehicle safety limits")
        if self.auto_mode:
            self.safety_limits["flight"] = {
                "max_altitude_m": 10.0,
                "max_speed_ms": 5.0,
                "geofence_radius_m": 50.0,
                "low_battery_rtl_pct": 25,
                "failsafe": "return_to_launch",
            }
        else:
            alt = _input_float("  Maximum altitude (meters, max 120): ",
                               0.0, 120.0)
            spd = _input_float("  Maximum speed (m/s, max 30): ", 0.0, 30.0)
            fence = _input_float("  Geofence radius (meters, max 500): ",
                                 0.0, 500.0)
            bat = _input_int("  Low battery RTL threshold (%, 10-50): ",
                             range(10, 51))
            self.safety_limits["flight"] = {
                "max_altitude_m": alt,
                "max_speed_ms": spd,
                "geofence_radius_m": fence,
                "low_battery_rtl_pct": bat,
                "failsafe": "return_to_launch",
            }

    # ── Semantic Understanding ────────────────────────────────────

    def _ask_purpose(self) -> None:
        if self.auto_mode:
            self.robot_purpose = f"general purpose {self.robot_type}"
            return
        print("\n[Calibration] Semantic Understanding")
        try:
            purpose = input(
                "Describe what this robot is designed to do "
                "in one sentence:\n> "
            ).strip()
        except EOFError:
            purpose = ""
        self.robot_purpose = purpose or f"general purpose {self.robot_type}"

    # ── Phase 5: Save Profile ─────────────────────────────────────

    def _save_profile(self) -> Dict:
        if not self.capabilities:
            self.capabilities = list(_TYPE_CAPABILITIES.get(
                self.robot_type, _TYPE_CAPABILITIES[CUSTOM]))

        # Build components from motor_map if not already set
        if not self.components:
            for ch, mapping in self.motor_map.items():
                func = mapping.get("function", "unknown")
                if func == "disconnected":
                    continue
                try:
                    ch_num = self.motor_channels.index(ch) + 1
                except ValueError:
                    ch_num = int(ch.split("_")[-1]) if "_" in ch else 0
                entry = {"channel": ch_num, "type": "motor"}
                if "min_angle" in mapping:
                    entry["type"] = "servo"
                    entry["min"] = mapping["min_angle"]
                    entry["max"] = mapping["max_angle"]
                self.components[func] = entry

        # Default purpose based on robot type if empty
        if not self.robot_purpose:
            _DEFAULT_PURPOSES = {
                GROUND_VEHICLE: "autonomous ground navigation",
                AERIAL_VEHICLE: "autonomous aerial navigation",
                ROBOTIC_ARM: "manipulation and object handling",
                AQUATIC_VEHICLE: "autonomous aquatic navigation",
                CUSTOM: "visual perception and environment monitoring",
            }
            self.robot_purpose = _DEFAULT_PURPOSES.get(
                self.robot_type, "general purpose robot")

        profile = {
            "robot_name": self.robot_name,
            "robot_type": self.robot_type,
            "robot_purpose": self.robot_purpose,
            "calibrated": True,
            "calibration_date": datetime.now().strftime("%Y-%m-%d"),
            "motor_map": self.motor_map,
            "components": self.components,
            "capabilities": self.capabilities,
            "safety_limits": self.safety_limits,
            "camera_verified": self.camera_verified,
        }

        # Merge any extra profile data (drive_type, frame_type, etc.)
        for key in ("drive_type", "expected_motors", "frame_type",
                     "joint_count", "joint_ranges", "thruster_count"):
            if key in self.profile:
                profile[key] = self.profile[key]

        os.makedirs(self.configs_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibrated_{self.robot_name}_{ts}.json"
        path = os.path.join(self.configs_dir, filename)

        with open(path, "w") as f:
            json.dump(profile, f, indent=2)

        print(f"\n[Calibration] Profile saved to: {path}")
        self.profile = profile
        return profile

    # ── Phase 6: Action Generation ────────────────────────────────

    @staticmethod
    def generate_actions(profile: Dict) -> List[Dict[str, Any]]:
        """Generate the correct action set from a calibration profile.

        Called by NavigationEngine on startup to build robot-specific
        actions.  Returns a list of action descriptors.
        """
        robot_type = profile.get("robot_type", CUSTOM)
        components = profile.get("components", {})
        safety = profile.get("safety_limits", {})

        actions: List[Dict[str, Any]] = []

        if robot_type == ROBOTIC_ARM:
            for name, comp in components.items():
                actions.append({
                    "name": f"move_joint_{name}",
                    "type": "move_joint",
                    "joint": name,
                    "channel": comp.get("channel"),
                    "min_angle": comp.get("min", -90),
                    "max_angle": comp.get("max", 90),
                    "max_speed": safety.get(name, {}).get("max_speed_dps", 45),
                })
            actions.append({"name": "grip", "type": "grip"})
            actions.append({"name": "release", "type": "release"})
            actions.append({
                "name": "point_to", "type": "point_to",
                "params": ["x", "y", "z"],
            })
            actions.append({"name": "home_position", "type": "home_position"})

        elif robot_type == GROUND_VEHICLE:
            max_speed = 0.7
            for lim in safety.values():
                if isinstance(lim, dict) and "max_speed" in lim:
                    max_speed = min(max_speed, lim["max_speed"])
            actions.extend([
                {"name": "move_forward", "type": "move",
                 "max_speed": max_speed},
                {"name": "move_backward", "type": "move",
                 "max_speed": max_speed},
                {"name": "turn_left", "type": "turn"},
                {"name": "turn_right", "type": "turn"},
                {"name": "navigate_to_color", "type": "navigate",
                 "params": ["color"]},
                {"name": "avoid_obstacle", "type": "navigate"},
                {"name": "follow_object", "type": "navigate",
                 "params": ["object"]},
            ])

        elif robot_type == AERIAL_VEHICLE:
            flight_safety = safety.get("flight", {})
            actions.extend([
                {"name": "takeoff", "type": "flight",
                 "params": ["altitude"],
                 "max_altitude": flight_safety.get("max_altitude_m", 10)},
                {"name": "hover", "type": "flight"},
                {"name": "navigate_waypoint", "type": "flight",
                 "params": ["lat", "lon", "alt"]},
                {"name": "land", "type": "flight"},
                {"name": "return_to_launch", "type": "flight"},
                {"name": "visual_scan", "type": "perception"},
            ])

        elif robot_type == AQUATIC_VEHICLE:
            actions.extend([
                {"name": "move_forward", "type": "move"},
                {"name": "move_backward", "type": "move"},
                {"name": "turn_left", "type": "turn"},
                {"name": "turn_right", "type": "turn"},
                {"name": "dive", "type": "depth"},
                {"name": "surface", "type": "depth"},
            ])

        # Always include stop
        actions.append({"name": "stop", "type": "safety"})
        return actions

    # ── Phase 7: Runtime Calibration Checks ───────────────────────

    @staticmethod
    def check_movement(
        expected_action: str,
        motor_map: Dict,
        before_frame: Optional[np.ndarray],
        after_frame: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Compare expected vs observed movement using camera frames.

        Returns a dict with:
          - matches: bool — whether observation agrees with expectation
          - motion_score: float
          - motion_direction: str
          - warning: optional warning message
        """
        if before_frame is None or after_frame is None:
            return {"matches": True, "motion_score": 0.0,
                    "motion_direction": "unknown",
                    "warning": "no frames available for verification"}

        cv2 = _try_cv2()
        if cv2 is None:
            return {"matches": True, "motion_score": 0.0,
                    "motion_direction": "unknown",
                    "warning": "cv2 not available"}

        before_gray = cv2.cvtColor(before_frame, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_frame, cv2.COLOR_BGR2GRAY)
        before_gray = cv2.GaussianBlur(before_gray, (21, 21), 0)
        after_gray = cv2.GaussianBlur(after_gray, (21, 21), 0)

        delta = cv2.absdiff(before_gray, after_gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        h, w = thresh.shape
        total_px = h * w
        motion_px = np.count_nonzero(thresh)
        score = motion_px / total_px if total_px > 0 else 0.0

        # Determine direction from optical flow centroid shift
        top_motion = np.count_nonzero(thresh[:h // 2, :])
        bot_motion = np.count_nonzero(thresh[h // 2:, :])
        left_motion = np.count_nonzero(thresh[:, :w // 2])
        right_motion = np.count_nonzero(thresh[:, w // 2:])

        if left_motion > right_motion * 1.5:
            direction = "leftward"
        elif right_motion > left_motion * 1.5:
            direction = "rightward"
        elif top_motion > bot_motion * 1.5:
            direction = "upward"
        elif bot_motion > top_motion * 1.5:
            direction = "downward"
        else:
            direction = "forward" if score > 0.01 else "none"

        # Check for mismatch
        expected_dir = _expected_direction(expected_action)
        matches = True
        warning = None

        if score < 0.002 and expected_action != "stop":
            matches = False
            warning = (f"[CalibrationCheck] WARNING: {expected_action} "
                       f"commanded but camera detected no motion. "
                       f"Component may have failed.")
        elif expected_dir and expected_dir != direction and score > 0.01:
            matches = False
            warning = (f"[CalibrationCheck] WARNING: {expected_action} "
                       f"commanded but camera detected {direction} motion. "
                       f"Motor mapping may be incorrect. Re-calibrate?")

        return {
            "matches": matches,
            "motion_score": round(score, 4),
            "motion_direction": direction,
            "warning": warning,
        }

    # ── Internal Helpers ──────────────────────────────────────────

    def _detect_channels(self) -> None:
        """Detect motor/actuator channels from manifest."""
        gpio_info = self._hw.get("gpio", {})
        if gpio_info.get("available"):
            pin_list = gpio_info.get("pins", [])
            if pin_list:
                self.motor_channels = [f"channel_{i+1}"
                                       for i in range(len(pin_list))]
            else:
                # Default: assume 4 channels on standard GPIO
                self.motor_channels = [f"channel_{i+1}" for i in range(4)]
            return

        mavlink_info = self._hw.get("mavlink", {})
        if mavlink_info.get("available"):
            motor_count = mavlink_info.get("motor_count", 4)
            self.motor_channels = [f"channel_{i+1}"
                                   for i in range(motor_count)]
            return

        self.motor_channels = []

    def _test_channel(self, channel: str, power: float = 0.1,
                      duration: float = 0.5) -> None:
        """Run a low-power test on a single channel."""
        print(f"  Testing now...")
        if self.adapter is not None:
            ch_num = self.motor_channels.index(channel)
            try:
                self.adapter.run("gpio_control", {
                    "pin": ch_num,
                    "mode": "output",
                    "value": True,
                })
                time.sleep(min(duration, 2.0))
                self.adapter.run("gpio_control", {
                    "pin": ch_num,
                    "mode": "output",
                    "value": False,
                })
            except Exception as e:
                print(f"  Test error: {e}")
        else:
            # No adapter — simulate the test delay
            time.sleep(min(duration, 2.0))

    def _auto_detect_type(self) -> str:
        """Attempt to auto-detect robot type from manifest."""
        if self._has_mavlink:
            return AERIAL_VEHICLE
        n = len(self.motor_channels)
        if n <= 4 and self._has_gpio:
            return GROUND_VEHICLE
        return CUSTOM

    def _auto_map_channel(self, channel: str) -> Dict:
        """Auto-map a channel using camera feedback."""
        # Without camera, assign by position
        idx = self.motor_channels.index(channel)
        default_funcs = [
            ("left_wheel", "forward"),
            ("right_wheel", "forward"),
            ("left_wheel", "backward"),
            ("right_wheel", "backward"),
        ]
        if idx < len(default_funcs):
            func, direction = default_funcs[idx]
        else:
            func, direction = f"motor_{idx+1}", "unknown"
        return {"function": func, "direction": direction, "confirmed": False}

    def _suggest_mapping(self, mapping: Dict, region: str) -> str:
        """Suggest a motor mapping name based on camera region."""
        if "left" in region:
            return "left_wheel_forward"
        elif "right" in region:
            return "right_wheel_forward"
        return mapping.get("function", "unknown")

    def _get_cv2(self):
        """Try to import cv2."""
        try:
            import cv2
            return cv2
        except ImportError:
            return None


def _try_cv2():
    """Module-level cv2 import helper for static methods."""
    try:
        import cv2
        return cv2
    except ImportError:
        return None


def _expected_direction(action: str) -> Optional[str]:
    """Map an action name to the expected camera motion direction."""
    mapping = {
        "move_forward": "forward",
        "move_backward": "downward",
        "turn_left": "leftward",
        "turn_right": "rightward",
    }
    return mapping.get(action)
