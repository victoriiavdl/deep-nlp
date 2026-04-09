#!/usr/bin/env python
"""Build the model leaderboard from training results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import build_leaderboard

if __name__ == "__main__":
    build_leaderboard()
