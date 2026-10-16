#!/usr/bin/env python3
"""Check MIMA source, documentation, metadata and published artifact integrity."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
VENDORED = (
    "MLLM/internvl_chat/",
    "MLLM/teacher_student/vendor/",
    "MLLM/internvl_chat_llava/",
    "MLLM/lmdeploy/",
    "MLLM/segmentation/",
    "MLLM/internvl_g/",
    "src/third_party/",
    "src/point_lio_crimson/",
    "src/motor_interface/dynamixel/dynamixel_sdk/",
)
REQUIRED = (
    "README.md",
    "LICENSE.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "requirements-energy.txt",
    "requirements-dev.txt",
    "scripts/check_ros_layout.py",
    "pyproject.toml",
    ".github/workflows/ci.yml",
    "docs/INSTALLATION.md",
    "docs/DOWNLOADS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/MODEL_CARD.md",
    "docs/THIRD_PARTY_NOTICES.md",
)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entries(text: str):
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
        if not match:
            raise ValueError(f"Malformed manifest line {number}")
        digest, name = match.groups()
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
            raise ValueError(f"Unsafe manifest path on line {number}")
        yield digest.lower(), name


def repository_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("Run the checker from a Git checkout with Git installed")
    return sorted({ROOT / p for p in result.stdout.decode("utf-8").split("\0") if p})


def first_party(path: Path) -> bool:
    return not path.relative_to(ROOT).as_posix().startswith(VENDORED)


def local_link_errors(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    targets = re.findall(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", text)
    targets += re.findall(r'<(?:img|a)\b[^>]*(?:src|href)=["\']([^"\']+)', text)
    errors = []
    for target in targets:
        target = target.strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith(("//", "#")):
            continue
        relative = unquote(parsed.path)
        if not relative:
            continue
        resolved = path.parent / relative
        if not resolved.exists():
            errors.append(f"{path.relative_to(ROOT)}: broken local link {target}")
    return errors


def find_bash() -> str | None:
    found = shutil.which("bash")
    if found:
        return found
    for candidate in (Path("C:/Program Files/Git/bin/bash.exe"),):
        if candidate.is_file():
            return str(candidate)
    return None


def check_manifest(path: Path) -> int:
    count = 0
    for expected, name in manifest_entries(path.read_text(encoding="utf-8-sig")):
        artifact = path.parent / name
        if not artifact.is_file():
            raise ValueError(f"Missing artifact: {artifact.relative_to(ROOT)}")
        with artifact.open("rb") as handle:
            if handle.read(100).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                raise ValueError(
                    f"Git LFS pointer: {artifact.relative_to(ROOT)}; run git lfs pull"
                )
        if digest_file(artifact) != expected:
            raise ValueError(f"Checksum mismatch: {artifact.relative_to(ROOT)}")
        count += 1
    return count


def check_zip_manifests(path: Path) -> int:
    count = 0
    with ZipFile(path) as bundle:
        manifests = [
            n
            for n in bundle.namelist()
            if PurePosixPath(n).name in {"MANIFEST.sha256", "SHA256SUMS"}
        ]
        for manifest in manifests:
            base = PurePosixPath(manifest).parent
            text = bundle.read(manifest).decode("utf-8-sig")
            for expected, name in manifest_entries(text):
                member = (base / name).as_posix()
                digest = hashlib.sha256()
                with bundle.open(member) as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f"Archive checksum mismatch: {path.name}:{member}")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-data", action="store_true", help="Hash all Git LFS data files"
    )
    parser.add_argument(
        "--verify-archives",
        action="store_true",
        help="Also verify internal ZIP manifests",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    failures = []
    checks = {}
    files = repository_files()
    for name in REQUIRED:
        if not (ROOT / name).is_file():
            failures.append(f"Missing release file: {name}")
    python_count = 0
    shell_count = 0
    bash = find_bash()
    for path in files:
        if not path.is_file() or not first_party(path):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
                python_count += 1
            except (SyntaxError, UnicodeError) as exc:
                failures.append(f"{relative}: {exc}")
        elif path.suffix == ".sh":
            if bash:
                result = subprocess.run(
                    [bash, "-n", str(path)], capture_output=True, text=True
                )
                if result.returncode:
                    failures.append(f"{relative}: {result.stderr.strip()}")
                shell_count += 1
            else:
                failures.append("Bash is required for shell syntax checks")
        elif path.suffix == ".md":
            failures.extend(local_link_errors(path))
            if "<your_repo_url>" in path.read_text(encoding="utf-8-sig"):
                failures.append(f"{relative}: placeholder clone URL")
        if path.name == "package.xml":
            tree = ET.parse(path).getroot()
            licenses = [x.text or "" for x in tree.findall("license")]
            if not licenses or any("TODO" in x.upper() for x in licenses):
                failures.append(f"{relative}: missing package license")
            for node in tree.findall("maintainer"):
                email = node.attrib.get("email", "")
                if (
                    not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email)
                    or "todo.todo" in email
                ):
                    failures.append(f"{relative}: invalid maintainer email")
    checks["python_syntax"] = python_count
    checks["shell_syntax"] = shell_count
    from check_ros_layout import check_ros_layout

    ros_checks, ros_errors = check_ros_layout(ROOT)
    checks.update(ros_checks)
    failures.extend(ros_errors)
    try:
        checks["module_manifest"] = check_manifest(
            ROOT / "experiments/module_level_ablation/MANIFEST.sha256"
        )
        baseline = ROOT / "MLLM/mima_requirement_vector/models/baselines"
        artifacts = json.loads((baseline / "checksums.json").read_text())["artifacts"]
        for artifact in artifacts:
            if digest_file(baseline / artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Baseline checksum mismatch: {artifact['path']}")
        checks["baseline_weights"] = len(artifacts)
    except (OSError, ValueError, KeyError) as exc:
        failures.append(str(exc))
    # The runtime and ablation copies must implement the same shared model code.
    models = ROOT / "experiments/module_level_ablation/models/gvae"
    runtime = ROOT / "src/generator/generation/gvae"
    shared = 0
    for model_path in models.rglob("*.py"):
        other = runtime / model_path.relative_to(models)
        if other.is_file():
            if model_path.read_text(encoding="utf-8") != other.read_text(
                encoding="utf-8"
            ):
                failures.append(
                    f"Generator source copies differ: {model_path.relative_to(models)}"
                )
            shared += 1
    checks["shared_generator_sources"] = shared
    if args.verify_data or args.verify_archives:
        try:
            checks["data_manifest"] = check_manifest(ROOT / "data/SHA256SUMS")
            if args.verify_archives:
                checks["archive_entries"] = sum(
                    check_zip_manifests(path) for path in (ROOT / "data").glob("*.zip")
                )
        except (OSError, ValueError, KeyError) as exc:
            failures.append(str(exc))
    report = {"checks": checks, "failures": failures, "ok": not failures}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, count in checks.items():
        print(f"PASS {name}: {count}")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
