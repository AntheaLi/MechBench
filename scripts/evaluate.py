"""Compatibility wrapper for `python3 -m mechbench.cli evaluate`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechbench.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
