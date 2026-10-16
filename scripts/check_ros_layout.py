"""Check ROS package dependencies and source/install entry points without ROS."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def cmake_code(path: Path, package_name: str) -> str:
    text = re.sub(r"#[^\n]*", "", path.read_text(encoding="utf-8-sig"))
    return text.replace("${PROJECT_NAME}", package_name)


def commands(code: str, name: str):
    return re.findall(r"\b" + name + r"\s*\((.*?)\)", code, flags=re.S | re.I)


def installed_executables(code: str) -> set[str]:
    executables = set()
    for block in commands(code, "catkin_install_python") + commands(code, "install"):
        tokens = block.split()
        if not tokens or tokens[0] not in {"PROGRAMS", "TARGETS"}:
            continue
        for token in tokens[1:]:
            if token in {"DESTINATION", "ARCHIVE", "LIBRARY", "RUNTIME"}:
                break
            executables.add(Path(token).name)
    return executables


def check_ros_layout(root: Path) -> tuple[dict[str, int], list[str]]:
    package_files = list((root / "src").rglob("package.xml"))
    packages = {}
    for path in package_files:
        tree = ET.parse(path).getroot()
        packages[tree.findtext("name")] = (path.parent, tree)
    known = set(packages)
    errors = []
    executable_maps = {}
    for name, (directory, tree) in packages.items():
        cmake_path = directory / "CMakeLists.txt"
        if not cmake_path.is_file():
            errors.append(f"{name}: missing CMakeLists.txt")
            continue
        code = cmake_code(cmake_path, name)
        dependencies = {n.text for n in tree if n.tag.endswith("depend")}
        for block in commands(code, "find_package"):
            tokens = block.split()
            if tokens[:1] == ["catkin"] and "COMPONENTS" in tokens:
                required = set(tokens[tokens.index("COMPONENTS") + 1 :])
                for missing in sorted(required - dependencies):
                    errors.append(
                        f"{name}: CMake dependency absent from package.xml: {missing}"
                    )
        executable_maps[name] = installed_executables(code)
        for block in commands(code, "catkin_install_python"):
            tokens = block.split()
            for relative in tokens[1 : tokens.index("DESTINATION")]:
                if "$" not in relative and not (directory / relative).is_file():
                    errors.append(f"{name}: missing installed Python entry: {relative}")
        if name == "dynamixel_sdk":
            continue
        imports = set()
        for path in directory.rglob("*"):
            if path.suffix == ".py":
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        imports.add(node.module.split(".")[0])
                    elif isinstance(node, ast.Import):
                        imports.update(alias.name.split(".")[0] for alias in node.names)
            elif path.suffix in {".cpp", ".h", ".hpp"}:
                imports.update(
                    re.findall(
                        r'#include\s*[<"]([\w]+)/',
                        path.read_text(encoding="utf-8-sig", errors="replace"),
                    )
                )
        for missing in sorted((imports & known) - dependencies - {name}):
            errors.append(
                f"{name}: imported local package absent from package.xml: {missing}"
            )
    launch_nodes = 0
    for name, (directory, _) in packages.items():
        for path in directory.rglob("*.launch"):
            for element in ET.parse(path).iter():
                if element.tag == "node" and element.get("pkg") in known:
                    target_package, executable = element.get("pkg"), element.get("type")
                    if executable and "$" not in executable:
                        launch_nodes += 1
                        if executable not in executable_maps.get(target_package, set()):
                            errors.append(
                                f"{path.relative_to(root)}: {target_package}/{executable} has no install entry"
                            )
                for attr in ("file", "textfile", "binfile"):
                    value = element.get(attr, "")
                    for package, relative in re.findall(
                        r'\$\(find ([^)]+)\)([^\s"\']*)', value
                    ):
                        if package in known and "$" not in relative:
                            target = packages[package][0] / relative.lstrip("/")
                            if not target.exists():
                                errors.append(
                                    f"{path.relative_to(root)}: missing launch resource: {value}"
                                )
    return {"ros_packages": len(packages), "ros_launch_nodes": launch_nodes}, errors
