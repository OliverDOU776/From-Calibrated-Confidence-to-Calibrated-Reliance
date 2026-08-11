#!/usr/bin/env python3
"""Remove only known generated analysis outputs from the repository worktree."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "results" / "generated"
ALLOWED_SUFFIXES = {".csv", ".json", ".png", ".pdf", ".txt"}


def main() -> None:
    if not GENERATED.is_dir():
        return
    removed = 0
    for path in GENERATED.iterdir():
        if path.is_file() and path.name != ".gitkeep" and path.suffix in ALLOWED_SUFFIXES:
            path.unlink()
            removed += 1
    print(f"Removed {removed} generated file(s) from {GENERATED}")


if __name__ == "__main__":
    main()
