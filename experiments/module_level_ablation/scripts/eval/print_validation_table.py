#!/usr/bin/env python3
"""Print a Markdown comparison table from validation metric JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_CSV_OUTPUT = Path("outputs/validation_metrics_table.csv")
AXES = ("x", "y", "z")
METHOD_NAME_ALIASES = {
    "Full pipeline": "Full generator",
    "CVAE + Scale GNN": "Full generator",
    "w/o Scale GNN -> MLP": "Scale GNN -> MLP",
    "CVAE + Scale MLP": "Scale GNN -> MLP",
    "w/o CVAE -> MLP": "CVAE -> MLP",
    "MLP + Scale GNN": "CVAE -> MLP",
    "Direct MLP baseline": "CVAE+Scale GNN -> MLP",
    "Direct MLP": "CVAE+Scale GNN -> MLP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine structure-generator validation metrics into one table."
    )
    parser.add_argument("metrics", nargs="+", help="Metric JSON files to compare.")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_CSV_OUTPUT}).",
    )
    return parser.parse_args()


def display_name(result: dict) -> str:
    if result.get("method_name"):
        name = str(result["method_name"])
        return METHOD_NAME_ALIASES.get(name, name)
    if (
        result.get("model_family") == "direct_mlp"
        and result.get("scale_mode") == "gnn"
    ):
        return "CVAE -> MLP"
    if result.get("model_family") == "direct_mlp":
        return "CVAE+Scale GNN -> MLP"
    if result.get("scale_mode") == "gnn":
        return "Full generator"
    if result.get("scale_mode") == "mlp":
        return "Scale GNN -> MLP"
    return str(result.get("model_family", "Model"))


def require_metric(mapping: dict, key: str, source: str) -> float:
    if key not in mapping:
        raise KeyError(
            f"{source} does not contain '{key}'. "
            "Please rerun scripts/eval/evaluate_structure_generator.py first."
        )
    return float(mapping[key])


def scale_mae_mm(scale_metrics: dict) -> dict[str, float]:
    mae_cm = scale_metrics["mae_cm"]
    return {axis: float(mae_cm[axis]) * 10.0 for axis in AXES}


def table_row(result: dict) -> tuple[list[str], list[str]]:
    reconstruction = result["reconstruction"]
    prior = result["prior_generation"]["overall"]
    samples = int(prior["samples_per_query"])
    scale = scale_mae_mm(reconstruction["scale_end_to_end"])
    source = str(result.get("model", "metric JSON"))
    headers = [
        "Method",
        "Orientation (rad) ↓",
        "Location (mm) ↓",
        "Ql (rad) ↓",
        "w x/y/z (mm) ↓",
        "Achievement Rate ↑",
        "Valid Rate ↑",
        f"Coverage@{samples} ↑",
        f"Diversity@{samples} ↑",
    ]
    values = [
        display_name(result),
        f"{require_metric(reconstruction, 'node_rotation_mean_rad', source):.2f}",
        f"{require_metric(reconstruction, 'node_position_mae_mm', source):.2f}",
        f"{require_metric(reconstruction, 'leg_angle_mae_rad', source):.2f}",
        f"{scale['x']:.2f}/{scale['y']:.2f}/{scale['z']:.2f}",
        f"{prior[f'success_at_{samples}']:.2%}",
        f"{prior['valid_sample_rate']:.2%}",
        f"{prior[f'coverage_at_{samples}']:.2%}",
        f"{prior[f'diversity_at_{samples}']:.2f}",
    ]
    return headers, values


def format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(values: list[str]) -> str:
        cells = []
        for index, value in enumerate(values):
            if index == 0:
                cells.append(value.ljust(widths[index]))
            else:
                cells.append(value.rjust(widths[index]))
        return "| " + " | ".join(cells) + " |"

    separator = [
        ":" + "-" * max(widths[0] - 1, 2)
    ] + [
        "-" * max(width - 1, 2) + ":"
        for width in widths[1:]
    ]
    return "\n".join(
        [
            format_row(headers),
            "| " + " | ".join(separator) + " |",
            *(format_row(row) for row in rows),
        ]
    )


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = []
    headers = None
    for path_text in args.metrics:
        result = json.loads(Path(path_text).read_text())
        current_headers, values = table_row(result)
        if headers is None:
            headers = current_headers
        elif headers != current_headers:
            raise ValueError(
                "All metric files must use the same samples_per_query value"
            )
        rows.append(values)

    print(format_markdown_table(headers, rows))
    write_csv(args.csv_output, headers, rows)
    print(f"\nCSV saved to: {args.csv_output}")


if __name__ == "__main__":
    main()
