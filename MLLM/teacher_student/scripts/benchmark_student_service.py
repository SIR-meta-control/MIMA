from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import requests


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the deployed sensor-to-v_r service."
    )
    parser.add_argument("--url", default="http://127.0.0.1:2610/predict")
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.warmup < 0 or args.requests <= 0:
        raise ValueError(
            "--warmup must be non-negative and --requests must be positive"
        )
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    session = requests.Session()
    for _ in range(args.warmup):
        response = session.post(args.url, json=payload, timeout=30)
        response.raise_for_status()

    round_trip: list[float] = []
    inference: list[float] = []
    server_total: list[float] = []
    last: dict[str, object] | None = None
    for _ in range(args.requests):
        start = time.perf_counter()
        response = session.post(args.url, json=payload, timeout=30)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.raise_for_status()
        last = response.json()
        timing = last["timing"]
        if not isinstance(timing, dict):
            raise TypeError("Service response timing must be an object")
        round_trip.append(elapsed_ms)
        inference.append(float(timing["inference_ms"]))
        server_total.append(float(timing["server_total_ms"]))

    assert last is not None
    result = {
        "url": args.url,
        "warmup_requests": args.warmup,
        "measured_requests": args.requests,
        "model": last.get("model"),
        "backend": last.get("backend"),
        "success_metric_version": last.get("success_metric_version"),
        "v_r": last.get("v_r"),
        **_latency_summary("round_trip", round_trip),
        **_latency_summary("inference", inference),
        **_latency_summary("server_total", server_total),
    }
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    print(encoded, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


def _latency_summary(prefix: str, values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean_ms": round(float(array.mean()), 3),
        f"{prefix}_median_ms": round(float(statistics.median(values)), 3),
        f"{prefix}_p90_ms": round(float(np.percentile(array, 90)), 3),
        f"{prefix}_p95_ms": round(float(np.percentile(array, 95)), 3),
        f"{prefix}_std_ms": round(float(array.std(ddof=1)), 3),
    }


if __name__ == "__main__":
    main()
