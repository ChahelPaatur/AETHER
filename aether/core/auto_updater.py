"""
AutoUpdater: checks for AETHER version updates via GitHub Releases API.

Usage:
    from aether.core.auto_updater import AutoUpdater
    updater = AutoUpdater()
    updater.run(auto_update=False, no_update=False)
"""
import os
import subprocess
import sys
from typing import Dict, Optional, Tuple

CURRENT_VERSION = "3.0"
_GITHUB_RELEASES_URL = (
    "https://api.github.com/repos/ChahelPaatur/AETHER/releases/latest"
)
_API_TIMEOUT = 3  # seconds


def _parse_version(tag: str) -> Tuple[int, ...]:
    """Parse a version tag like 'v3.1.2' or '3.1' into a comparable tuple."""
    stripped = tag.lstrip("vV").strip()
    parts = []
    for seg in stripped.split("."):
        try:
            parts.append(int(seg))
        except ValueError:
            break
    return tuple(parts) or (0,)


class AutoUpdater:
    """Checks for AETHER updates via GitHub and applies them if requested."""

    def __init__(self, project_root: Optional[str] = None):
        self._root = project_root or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._version_file = os.path.join(self._root, "VERSION")
        self._has_git = os.path.isdir(os.path.join(self._root, ".git"))
        self._release_notes: str = ""

    def get_current_version(self) -> str:
        """Read version from VERSION file, fallback to hardcoded."""
        if os.path.exists(self._version_file):
            try:
                with open(self._version_file) as f:
                    return f.read().strip()
            except OSError:
                pass
        return CURRENT_VERSION

    def _fetch_latest_release(self) -> Optional[Dict]:
        """Fetch latest release info from GitHub. Returns None on failure."""
        try:
            import requests
        except ImportError:
            return None

        try:
            resp = requests.get(
                _GITHUB_RELEASES_URL,
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=_API_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def check_for_update(self) -> Optional[str]:
        """Check GitHub releases for a newer version.

        Returns the new version tag if an update is available, else None.
        Silently returns None on any network/API failure.
        """
        release = self._fetch_latest_release()
        if release is None:
            return None

        remote_tag = release.get("tag_name", "")
        if not remote_tag:
            return None

        local = _parse_version(self.get_current_version())
        remote = _parse_version(remote_tag)

        if remote > local:
            self._release_notes = release.get("body", "") or ""
            return remote_tag
        return None

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

        print(f"[AutoUpdater] Update available: v{version} → {new_version}")

        if self._release_notes:
            print(f"\n  Release notes:\n")
            for line in self._release_notes.splitlines():
                print(f"    {line}")
            print()

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
        """Apply the update via git pull origin main, then restart."""
        if self._has_git:
            print("[AutoUpdater] Running git pull origin main...")
            try:
                result = subprocess.run(
                    ["git", "pull", "origin", "main"],
                    capture_output=True, text=True,
                    timeout=30, cwd=self._root,
                )
                if result.returncode == 0:
                    print("[AutoUpdater] Update applied — restarting...")
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                    return True  # unreachable after execv
                else:
                    print(f"[AutoUpdater] git pull failed: "
                          f"{result.stderr.strip()}")
                    return False
            except (subprocess.TimeoutExpired, OSError) as e:
                print(f"[AutoUpdater] git pull error: {e}")
                return False
        else:
            print("[AutoUpdater] No .git directory found.")
            print("[AutoUpdater] To update manually:")
            print("  1. Download the latest release from GitHub")
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
