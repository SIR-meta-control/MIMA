#!/usr/bin/env python3
"""Serve the current GVAE generator through a small HTTP JSON API."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def find_package_dir():
    source_candidate = Path(__file__).resolve().parents[1]
    if (source_candidate / "generation" / "runtime.py").is_file():
        return source_candidate
    try:
        import rospkg
    except ImportError as exc:
        raise RuntimeError("rospkg is required to locate the generator package") from exc
    try:
        return Path(rospkg.RosPack().get_path("generator"))
    except rospkg.ResourceNotFound as exc:
        raise RuntimeError("could not locate the generator ROS package") from exc


PACKAGE_DIR = find_package_dir()
GENERATION_DIR = PACKAGE_DIR / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from runtime import GVAEGeneratorRuntime
from service_contract import normalize_generation_request


LOGGER = logging.getLogger("gvae_generator_service")
MAX_REQUEST_BYTES = 1024 * 1024


class GeneratorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, runtime, defaults):
        super().__init__(address, handler)
        self.runtime = runtime
        self.defaults = defaults
        self.inference_lock = threading.Lock()
        self.started_at = time.time()


class GeneratorRequestHandler(BaseHTTPRequestHandler):
    server_version = "MIMA-GVAE/1.0"

    def log_message(self, fmt, *args):
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, status, payload):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type.lower():
            raise ValueError("Content-Type must be application/json")
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if content_length <= 0:
            raise ValueError("request body must not be empty")
        if content_length > MAX_REQUEST_BYTES:
            raise ValueError("request body exceeds 1 MiB")
        try:
            return json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must contain valid UTF-8 JSON") from exc

    def do_GET(self):
        if self.path.rstrip("/") != "/health":
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(
            200,
            {
                "status": "ok",
                "service": "gvae-generator",
                "uptime_s": time.time() - self.server.started_at,
                **self.server.runtime.metadata,
            },
        )

    def do_POST(self):
        if self.path.rstrip("/") != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            request = normalize_generation_request(
                self._read_json(),
                default_samples_per_bar=self.server.defaults["samples_per_bar"],
                default_top_k=self.server.defaults["top_k"],
                max_samples_per_bar=self.server.defaults["max_samples_per_bar"],
                max_top_k=self.server.defaults["max_top_k"],
            )
            started_at = time.perf_counter()
            with self.server.inference_lock:
                result = self.server.runtime.generate(**request)
            result["timing"] = {
                "inference_s": time.perf_counter() - started_at,
            }
            self._send_json(200, result)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            LOGGER.exception("generator request failed")
            self._send_json(500, {"error": "generator inference failed: %s" % exc})


def parse_args():
    default_model = GENERATION_DIR / "checkpoints" / "full_generator.pt"
    default_classifier = GENERATION_DIR / "checkpoints" / "bar_classifier.pt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--bar-classifier", type=Path, default=default_classifier)
    parser.add_argument(
        "--graph-imputation",
        type=Path,
        default=GENERATION_DIR / "graph_imputation.yaml",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--default-samples-per-bar", type=int, default=64)
    parser.add_argument("--default-top-k", type=int, default=10)
    parser.add_argument("--max-samples-per-bar", type=int, default=4096)
    parser.add_argument("--max-top-k", type=int, default=256)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if args.default_samples_per_bar <= 0 or args.max_samples_per_bar <= 0:
        raise ValueError("sample limits must be positive")
    if args.default_top_k <= 0 or args.max_top_k <= 0:
        raise ValueError("Top-K limits must be positive")
    if args.default_samples_per_bar > args.max_samples_per_bar:
        raise ValueError("default samples-per-bar exceeds service maximum")
    if args.default_top_k > args.max_top_k:
        raise ValueError("default top-k exceeds service maximum")

    LOGGER.info("loading GVAE generator on %s", args.device)
    runtime = GVAEGeneratorRuntime(
        model_path=args.model,
        bar_classifier_path=args.bar_classifier,
        graph_imputation_path=args.graph_imputation,
        device=args.device,
    )
    defaults = {
        "samples_per_bar": args.default_samples_per_bar,
        "top_k": args.default_top_k,
        "max_samples_per_bar": args.max_samples_per_bar,
        "max_top_k": args.max_top_k,
    }
    server = GeneratorHTTPServer(
        (args.host, args.port),
        GeneratorRequestHandler,
        runtime,
        defaults,
    )
    LOGGER.info(
        "GVAE service ready at http://%s:%d (device=%s)",
        args.host,
        args.port,
        runtime.device,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        LOGGER.info("shutdown requested")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
