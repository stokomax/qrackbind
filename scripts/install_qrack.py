#!/usr/bin/env python3
"""
scripts/install_qrack.py

Python wrapper around scripts/install_qrack.sh, invoked by:

    uv run install-qrack
    uv run install-qrack-cpu
    uv run install-qrack-cuda

Simply delegates to the shell script with the appropriate flags.
"""
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).parent / "install_qrack.sh"


def main() -> None:
    cmd = ["bash", str(_SCRIPT)] + sys.argv[1:]
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"ERROR: bash not found. Run the script directly:\n  bash {_SCRIPT}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
