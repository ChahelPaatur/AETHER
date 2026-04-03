"""Allow ``python -m aether`` and the ``aether`` console_scripts entry point.

This module is the single entry point for all pip-installed invocations.
It works from any directory because all imports use the ``aether.*`` namespace.
"""
import sys
import os


def main():
    # Ensure the aether package's parent is on sys.path so that
    # ``import aether.core...`` resolves whether we're running from
    # source (python -m aether) or from a pip-installed console script.
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))   # aether/
    _parent = os.path.dirname(_pkg_dir)                     # parent of aether/

    if _parent not in sys.path:
        sys.path.insert(0, _parent)

    # Verify core modules are reachable
    core_dir = os.path.join(_pkg_dir, "core")
    if not os.path.isdir(core_dir):
        print("ERROR: AETHER core modules not found.")
        print("The package may not be correctly installed.")
        print("Try: pip uninstall aether-robotics && pip install aether-robotics")
        sys.exit(1)

    # Import and run the self-contained app module
    try:
        from aether.app import main as _app_main
        _app_main()
    except ImportError as e:
        print(f"ERROR: Could not start AETHER: {e}")
        print("Please run from source: cd AETHER && python main.py --mode agent")
        sys.exit(1)


if __name__ == "__main__":
    main()
