"""
AutoInstaller: detects missing Python libraries and offers to install them.

Usage:
    from aether.core.auto_installer import AutoInstaller
    installer = AutoInstaller(manifest)
    installer.run(auto_install=False, no_install=False)
"""
import importlib
import json
import os
import platform
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


INSTALL_MAP = {
    "cv2": "opencv-python-headless",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "dronekit": "dronekit",
    "pymavlink": "pymavlink",
    "pyaudio": "pyaudio",
    "smbus2": "smbus2",
    "serial": "pyserial",
    "tflite_runtime": "tflite-runtime",
    "ultralytics": "ultralytics",
    "flask": "flask",
    "requests": "requests",
    "anthropic": "anthropic",
    "psutil": "psutil",
    "PIL": "Pillow",
}

# What each library enables
CAPABILITY_MAP = {
    "cv2": "camera vision, object detection",
    "scipy": "signal processing",
    "sklearn": "machine learning",
    "torch": "deep learning, GPU inference",
    "dronekit": "drone control",
    "pymavlink": "MAVLink communication",
    "pyaudio": "audio capture/playback",
    "smbus2": "I2C sensor communication",
    "serial": "serial port communication",
    "tflite_runtime": "TFLite object detection (Pi-friendly)",
    "ultralytics": "YOLOv8 object detection",
    "flask": "HTTP server",
    "requests": "HTTP requests",
    "anthropic": "Anthropic API (LLM planner)",
    "psutil": "system monitoring",
    "PIL": "image processing",
}


_PREFS_PATH = os.path.join("logs", "installer_prefs.json")


def _load_skip_pref() -> bool:
    """Return True if the user previously chose 'skip' permanently."""
    if not os.path.exists(_PREFS_PATH):
        return False
    try:
        with open(_PREFS_PATH) as f:
            data = json.load(f)
        return data.get("skip_install", False)
    except (json.JSONDecodeError, OSError):
        return False


def _save_skip_pref(skip: bool) -> None:
    """Persist the skip preference to disk."""
    os.makedirs(os.path.dirname(_PREFS_PATH) or ".", exist_ok=True)
    data = {}
    if os.path.exists(_PREFS_PATH):
        try:
            with open(_PREFS_PATH) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data["skip_install"] = skip
    with open(_PREFS_PATH, "w") as f:
        json.dump(data, f, indent=2)


class AutoInstaller:
    """Detects missing libraries and offers to install them."""

    def __init__(self, manifest: Optional[Dict] = None):
        self._manifest = manifest or {}
        self._skip_session = False
        self._is_linux = platform.system() == "Linux"

    def check_missing(self) -> List[Tuple[str, str, str]]:
        """Return list of (module_name, pip_package, capability) for missing libs."""
        missing = []
        for mod_name, pip_name in INSTALL_MAP.items():
            try:
                importlib.import_module(mod_name)
            except ImportError:
                capability = CAPABILITY_MAP.get(mod_name, "unknown")
                missing.append((mod_name, pip_name, capability))
        return missing

    def compute_capability_score(self) -> int:
        """Return a 0-100 score based on installed libraries."""
        total = len(INSTALL_MAP)
        installed = 0
        for mod_name in INSTALL_MAP:
            try:
                importlib.import_module(mod_name)
                installed += 1
            except ImportError:
                pass
        return int(installed / total * 100) if total > 0 else 0

    def run(self, auto_install: bool = False, no_install: bool = False) -> bool:
        """Run the auto-installer flow.

        Args:
            auto_install: Install everything without asking.
            no_install: Skip the prompt entirely (also skips if user
                        previously chose 'skip' and preference is saved).

        Returns:
            True if any packages were installed.
        """
        if no_install or self._skip_session or _load_skip_pref():
            return False

        missing = self.check_missing()
        if not missing:
            score = self.compute_capability_score()
            print(f"[AutoInstaller] All libraries available — capability score: {score}")
            return False

        score_before = self.compute_capability_score()

        print(f"\n[AutoInstaller] Missing libraries detected:")
        for mod_name, pip_name, capability in missing:
            print(f"  - {pip_name} (needed for: {capability})")

        if not auto_install:
            print(f"\nInstall missing libraries? [Y/n/skip]: ", end="", flush=True)
            try:
                choice = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "n"

            if choice == "skip":
                self._skip_session = True
                _save_skip_pref(True)
                print("[AutoInstaller] Skipped permanently (delete "
                      f"{_PREFS_PATH} to re-enable)")
                return False
            if choice not in ("", "y", "yes"):
                print("[AutoInstaller] Skipped")
                return False

        # Install packages
        installed_count = 0
        for mod_name, pip_name, capability in missing:
            print(f"[AutoInstaller] Installing {pip_name}...", end=" ", flush=True)
            success = self._pip_install(pip_name)
            if success:
                installed_count += 1
                print("OK")
            else:
                print("FAILED")

        score_after = self.compute_capability_score()

        # Determine newly unlocked tools
        unlocked = []
        for mod_name, pip_name, capability in missing:
            try:
                importlib.import_module(mod_name)
                unlocked.append(capability.split(",")[0].strip())
            except ImportError:
                pass

        print(f"\n[AutoInstaller] Installed {installed_count} libraries")
        print(f"[AutoInstaller] Capability score: {score_before} → {score_after}")
        if unlocked:
            print(f"[AutoInstaller] New tools unlocked: {', '.join(unlocked)}")

        return installed_count > 0

    def _pip_install(self, package: str) -> bool:
        """Install a package via pip."""
        cmd = [sys.executable, "-m", "pip", "install", package, "-q"]
        if self._is_linux:
            cmd.append("--break-system-packages")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False
