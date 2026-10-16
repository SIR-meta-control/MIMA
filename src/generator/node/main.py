#!/usr/bin/env python3
"""Compatibility entry point for the generator ROS adapter."""

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from robot_config_generator import main


if __name__ == "__main__":
    main()
