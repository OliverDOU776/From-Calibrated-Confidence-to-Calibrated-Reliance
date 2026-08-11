#!/usr/bin/env python3
"""Download and verify the public HAIID files used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "HAIID"
UPSTREAM_COMMIT = "24881cc7586180a9c9742a7dd838aea97d008235"
RAW_BASE = (
    "https://raw.githubusercontent.com/kailas-v/human-ai-interactions/"
    f"{UPSTREAM_COMMIT}"
)


@dataclass(frozen=True)
class SourceFile:
    name: str
    sha256: str

    @property
    def url(self) -> str:
        return f"{RAW_BASE}/{self.name}"


FILES = (
    SourceFile(
        "haiid_dataset.csv",
        "be9223b6bf34f996cdace9b1c0d43876df0e480bcb9322e6a7f774de0f2f0eed",
    ),
    SourceFile(
        "haiid_dataset_description.csv",
        "2d5fe97cf0af1ae67bff402eae073f6bc1a92442a648af73d8470ef8c691560d",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, expected: str) -> bool:
    return path.is_file() and sha256(path) == expected


def download(source: SourceFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{source.name}.", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            request = urllib.request.Request(
                source.url,
                headers={"User-Agent": "calibrated-reliance-reproducibility/1.0"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                shutil.copyfileobj(response, temporary)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    observed = sha256(temporary_path)
    if observed != source.sha256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {source.name}: expected {source.sha256}, observed {observed}"
        )
    temporary_path.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the pinned HAIID dataset files and verify their SHA-256 checksums."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify existing files without downloading anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even when a valid local copy exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    for source in FILES:
        destination = args.output / source.name
        if verified(destination, source.sha256) and not args.force:
            print(f"OK   {destination}")
            continue
        if args.check_only:
            state = "missing" if not destination.exists() else "checksum mismatch"
            failures.append(f"{destination}: {state}")
            continue
        print(f"GET  {source.url}")
        try:
            download(source, destination)
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            failures.append(f"{source.name}: {exc}")
        else:
            print(f"OK   {destination}")

    if failures:
        print("Data setup failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"HAIID ready at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
