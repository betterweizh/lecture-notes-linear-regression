#!/usr/bin/env python3
"""Run the Python code blocks of the chapters, in order, one namespace per file.

Usage (from the repository root)::

    python scripts/run_code_cells.py              # every chapter with Python code
    python scripts/run_code_cells.py docs/en/linear-regression-python.md

A chapter passes when all of its blocks run without an exception. The
printed results are not compared with the "# Expected:" comments here; check
those by hand when a block changes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.S | re.M)


def python_blocks(path: Path) -> list[str]:
    return [m.group(1) for m in FENCE.finditer(path.read_text(encoding="utf-8"))]


def run_file(path: Path) -> bool:
    blocks = python_blocks(path)
    if not blocks:
        return True
    namespace: dict = {"__name__": "__chapter__"}
    for number, code in enumerate(blocks, start=1):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, f"{path.name}#block{number}", "exec"), namespace)
        except Exception:
            print(f"FAIL {path.relative_to(ROOT)} block {number}")
            traceback.print_exc()
            return False
    print(f"ok   {path.relative_to(ROOT)} ({len(blocks)} blocks)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    files = args.files or sorted((ROOT / "docs" / "en").glob("*.md"))
    results = [run_file((ROOT / f) if not f.is_absolute() else f) for f in files]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
