#!/usr/bin/env python3
"""Verify and safely prepare the released MIMA ablation datasets."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = {
    "module": (
        "module_level_ablation_assets.zip",
        ROOT / "experiments/module_level_ablation",
        "datasets/",
    ),
    "system": (
        "system_level_ablation_assets.zip",
        ROOT / "experiments/system_level_ablation/data_bundle",
        "",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_archive_hash(name: str) -> str:
    for line in (ROOT / "data/SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        if filename.lstrip("*") == name:
            return digest
    raise ValueError(f"No published checksum for {name}")


def verify_archive(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Fetch {path.name} with git lfs pull first")
    with path.open("rb") as handle:
        if handle.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{path.name} is a Git LFS pointer; run git lfs pull")
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}; extraction cancelled")


def safe_member_path(destination: Path, name: str) -> Path:
    # ZIP names are POSIX paths even when extracted on Windows.
    relative = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError(f"Unsafe archive member: {name!r}")
    target = destination.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"Archive member escapes destination: {name!r}")
    # A pre-existing symlink may redirect writes even when the ZIP has no links.
    for ancestor in (target, *target.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Symlink in extraction path: {ancestor}")
        if ancestor == destination:
            break
    return target


def extract_verified_zip(archive: Path, destination: Path, prefix: str = "") -> int:
    destination = destination.absolute()
    planned = []
    with ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if not info.filename.startswith(prefix):
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Symlink archive member: {info.filename!r}")
            target = safe_member_path(destination, info.filename)
            planned.append((info, target))
        if not planned:
            raise ValueError(f"Archive has no members under {prefix!r}")
        # Validate every path before writing any archive contents.
        files = 0
        for info, target in planned:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                temporary = Path(temp.name)
                try:
                    with bundle.open(info) as source:
                        shutil.copyfileobj(source, temp)
                except BaseException:
                    temp.close()
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                if target.exists():
                    if not target.is_file() or sha256(target) != sha256(temporary):
                        raise FileExistsError(f"Existing file differs: {target}")
                else:
                    # Hard-link creation is atomic and refuses an existing target.
                    # It also avoids replacing another process's newly created file.
                    os.link(temporary, target)
                files += 1
            finally:
                temporary.unlink(missing_ok=True)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", choices=tuple(BUNDLES))
    args = parser.parse_args()
    name, destination, prefix = BUNDLES[args.bundle]
    archive = ROOT / "data" / name
    print(f"Verifying {name}...", flush=True)
    verify_archive(archive, expected_archive_hash(name))
    count = extract_verified_zip(archive, destination, prefix)
    print(f"Prepared {count} files under {destination}", flush=True)


if __name__ == "__main__":
    main()
