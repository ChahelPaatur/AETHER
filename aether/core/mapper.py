"""
EnvironmentMapper: builds a 2D occupancy grid during calibration.

When motors and camera are both available the mapper executes a grid
movement pattern (forward → right → forward → right ...), captures
frames at each position, computes optical flow between consecutive
frames, and builds a 100×100 occupancy grid from motion data.

Saves:
  configs/environment_map_TIMESTAMP.png   — heatmap visualisation
  configs/environment_map_TIMESTAMP.npy   — raw numpy array
"""
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")

# Grid dimensions
GRID_SIZE = 100

# Movement pattern: forward, right, forward, right, forward, right, forward
_PATTERN = ["forward", "right", "forward", "right", "forward", "right", "forward"]

# Default durations (seconds)
_MOVE_DURATION = 0.5
_TURN_DURATION = 0.3
_CAPTURE_DELAY = 0.2


class EnvironmentMapper:
    """Builds a 2D occupancy grid using camera + motor exploration.

    The mapper drives the robot through a rectangular sweep pattern,
    capturing frames at each waypoint.  Optical flow between consecutive
    frames is used to infer occupied vs free cells in the grid.

    Parameters
    ----------
    camera_tool : object
        Any object with a ``capture_frame()`` method that returns a numpy
        array (H×W×C) or a dict with key ``"frame"`` containing one.
    motor_fn : callable
        ``motor_fn(action: str, duration: float)`` — drives motors.
        *action* is ``"forward"`` or ``"right"``.
    configs_dir : str, optional
        Directory to save the map files.  Defaults to ``configs/``.
    """

    def __init__(
        self,
        camera_tool: Any,
        motor_fn: Callable,
        configs_dir: Optional[str] = None,
    ):
        self._camera = camera_tool
        self._motor_fn = motor_fn
        self._configs_dir = os.path.abspath(configs_dir or _CONFIGS_DIR)
        self._grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self._frames: List[np.ndarray] = []
        self._flow_vectors: List[np.ndarray] = []

    # ── Public API ───────────────────────────────────────────────

    def run(self) -> Dict:
        """Execute the full mapping sequence.

        Returns
        -------
        dict
            ``{"grid": np.ndarray, "png_path": str, "npy_path": str,
               "waypoints": int, "occupied_cells": int}``
        """
        self._frames.clear()
        self._flow_vectors.clear()
        self._grid[:] = 0

        # Capture initial frame
        frame = self._capture()
        if frame is not None:
            self._frames.append(frame)

        # Execute movement pattern, capturing after each step
        for action in _PATTERN:
            duration = _TURN_DURATION if action == "right" else _MOVE_DURATION
            try:
                self._motor_fn(action, duration)
            except Exception:
                pass  # best-effort movement
            time.sleep(_CAPTURE_DELAY)

            frame = self._capture()
            if frame is not None:
                self._frames.append(frame)

        # Compute optical flow between consecutive frames
        self._compute_flow()

        # Build occupancy grid from flow data
        self._build_grid()

        # Save artefacts
        png_path, npy_path = self._save()

        occupied = int(np.sum(self._grid > 0))
        return {
            "grid": self._grid.copy(),
            "png_path": png_path,
            "npy_path": npy_path,
            "waypoints": len(self._frames),
            "occupied_cells": occupied,
        }

    # ── Internals ────────────────────────────────────────────────

    def _capture(self) -> Optional[np.ndarray]:
        """Grab a single frame from the camera tool."""
        try:
            raw = self._camera.capture_frame()
            if isinstance(raw, dict):
                raw = raw.get("frame", raw.get("image"))
            if raw is None:
                return None
            frame = np.asarray(raw, dtype=np.uint8)
            # Convert to greyscale if colour
            if frame.ndim == 3 and frame.shape[2] >= 3:
                frame = (
                    0.299 * frame[:, :, 0]
                    + 0.587 * frame[:, :, 1]
                    + 0.114 * frame[:, :, 2]
                ).astype(np.uint8)
            return frame
        except Exception:
            return None

    def _compute_flow(self) -> None:
        """Compute frame-to-frame optical flow using simple differencing."""
        self._flow_vectors.clear()
        for i in range(1, len(self._frames)):
            prev = self._frames[i - 1].astype(np.float32)
            curr = self._frames[i].astype(np.float32)

            # Resize to common shape if needed
            if prev.shape != curr.shape:
                h = min(prev.shape[0], curr.shape[0])
                w = min(prev.shape[1], curr.shape[1])
                prev = prev[:h, :w]
                curr = curr[:h, :w]

            diff = np.abs(curr - prev)
            # Threshold: pixels with >25 intensity change count as motion
            motion = (diff > 25.0).astype(np.float32)
            self._flow_vectors.append(motion)

    def _build_grid(self) -> None:
        """Project motion vectors onto the 100×100 occupancy grid.

        Each flow frame is down-sampled to GRID_SIZE and accumulated.
        Cells with cumulative motion above the mean are marked occupied.
        """
        if not self._flow_vectors:
            return

        accumulator = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        for motion in self._flow_vectors:
            resized = self._resize(motion, GRID_SIZE, GRID_SIZE)
            accumulator += resized

        # Normalise to [0, 1]
        mx = accumulator.max()
        if mx > 0:
            accumulator /= mx

        # Threshold: above-mean cells are occupied
        threshold = float(np.mean(accumulator))
        self._grid = (accumulator > threshold).astype(np.float32)

    @staticmethod
    def _resize(arr: np.ndarray, h: int, w: int) -> np.ndarray:
        """Simple nearest-neighbour resize using numpy (no cv2 dependency)."""
        src_h, src_w = arr.shape[:2]
        row_idx = (np.arange(h) * src_h // h).astype(int)
        col_idx = (np.arange(w) * src_w // w).astype(int)
        return arr[np.ix_(row_idx, col_idx)]

    def _save(self) -> Tuple[str, str]:
        """Save grid as .npy and .png heatmap."""
        os.makedirs(self._configs_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        npy_path = os.path.join(
            self._configs_dir, f"environment_map_{ts}.npy")
        np.save(npy_path, self._grid)

        png_path = os.path.join(
            self._configs_dir, f"environment_map_{ts}.png")
        self._save_heatmap_png(self._grid, png_path)

        return png_path, npy_path

    @staticmethod
    def _save_heatmap_png(grid: np.ndarray, path: str) -> None:
        """Save grid as a PNG heatmap using matplotlib if available,
        otherwise fall back to a raw greyscale PGM file."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(grid, cmap="hot", interpolation="nearest",
                      origin="lower", vmin=0, vmax=1)
            ax.set_title("Environment Occupancy Map")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            # Fallback: write a plain-text PGM image (always works)
            h, w = grid.shape
            pixels = (grid * 255).astype(np.uint8)
            with open(path, "wb") as f:
                f.write(f"P5\n{w} {h}\n255\n".encode())
                f.write(pixels.tobytes())
