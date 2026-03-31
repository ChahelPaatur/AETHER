"""
AutoUpdater: checks for AETHER version updates and offers to apply them.

Usage:
    from aether.core.auto_updater import AutoUpdater
    updater = AutoUpdater()
    updater.run(auto_update=False, no_update=False)
"""
import os
import subprocess
import sys
from typing import Optional


CURRENT_VERSION = "3.0"


class AutoUpdater:
    """Checks for AETHER updates and applies them if requested."""

    def __init__(self, project_root: Optional[str] = None):
        self._root = project_root or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._version_file = os.path.join(self._root, "VERSION")
        self._has_git = os.path.isdir(os.path.join(self._root, ".git"))

    def get_current_version(self) -> str:
        """Read version from VERSION file, fallback to hardcoded."""
        if os.path.exists(self._version_file):
            try:
                with open(self._version_file) as f:
                    return f.read().strip()
            except OSError:
                pass
        return CURRENT_VERSION

    def check_for_update(self) -> Optional[str]:
        """Check if a newer version is available.

        For now, compares local VERSION file against CURRENT_VERSION.
        Returns the new version string if update available, else None.
        """
        local_version = self.get_current_version()
        # In future: fetch remote version from GitHub releases
        # For now, just report current version
        return None  # No update mechanism yet

    def run(self, auto_update: bool = False, no_update: bool = False) -> bool:
        """Run the update check flow.

        Args:
            auto_update: Update without asking.
            no_update: Skip the check entirely.

        Returns:
            True if an update was applied (process will restart).
        """
        if no_update:
            return False

        version = self.get_current_version()
        new_version = self.check_for_update()

        if new_version is None:
            print(f"[AutoUpdater] AETHER v{version} — up to date")
            return False

        print(f"[AutoUpdater] Update available: v{version} → v{new_version}")

        if not auto_update:
            print("Update now? [Y/n]: ", end="", flush=True)
            try:
                choice = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "n"

            if choice not in ("", "y", "yes"):
                print("[AutoUpdater] Skipped")
                return False

        return self._apply_update()

    def _apply_update(self) -> bool:
        """Apply the update via git pull or manual instructions."""
        if self._has_git:
            print("[AutoUpdater] Running git pull...")
            try:
                result = subprocess.run(
                    ["git", "pull"], capture_output=True, text=True,
                    timeout=30, cwd=self._root)
                if result.returncode == 0:
                    print("[AutoUpdater] Update applied — restarting...")
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                    return True  # unreachable after execv
                else:
                    print(f"[AutoUpdater] git pull failed: {result.stderr.strip()}")
                    return False
            except (subprocess.TimeoutExpired, OSError) as e:
                print(f"[AutoUpdater] git pull error: {e}")
                return False
        else:
            print("[AutoUpdater] No .git directory found.")
            print("[AutoUpdater] To update manually:")
            print("  1. Download the latest version from the project repository")
            print("  2. Replace the AETHER directory contents")
            print("  3. Restart: python main.py --mode agent")
            return False

    @staticmethod
    def ensure_version_file(project_root: str):
        """Create a VERSION file if one doesn't exist."""
        version_file = os.path.join(project_root, "VERSION")
        if not os.path.exists(version_file):
            try:
                with open(version_file, "w") as f:
                    f.write(CURRENT_VERSION + "\n")
            except OSError:
                pass
