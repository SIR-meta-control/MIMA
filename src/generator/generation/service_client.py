"""Standard-library HTTP client for the GVAE generator service."""

from __future__ import annotations

import json
from urllib import error as urllib_error
from urllib import request as urllib_request

from service_contract import validate_generation_response, validate_vreq


class GeneratorServiceError(RuntimeError):
    pass


class GeneratorServiceClient:
    def __init__(self, base_url, timeout_s=120.0):
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)
        if not self.base_url:
            raise ValueError("generator service URL must not be empty")
        if self.timeout_s <= 0.0:
            raise ValueError("generator service timeout must be positive")

    def _request_json(self, method, path, payload=None):
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib_request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
                status = int(response.status)
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except (ValueError, AttributeError):
                pass
            raise GeneratorServiceError(
                "generator service returned HTTP %d: %s" % (exc.code, detail)
            ) from exc
        except urllib_error.URLError as exc:
            raise GeneratorServiceError(
                "generator service is unavailable: %s" % exc.reason
            ) from exc

        if status < 200 or status >= 300:
            raise GeneratorServiceError(
                "generator service returned HTTP %d" % status
            )
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise GeneratorServiceError(
                "generator service returned invalid JSON"
            ) from exc

    def health(self):
        result = self._request_json("GET", "/health")
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise GeneratorServiceError("generator service health check failed")
        return result

    def generate(
        self,
        vreq,
        samples_per_bar=64,
        top_k=10,
        bar_types="auto",
        temperature=1.0,
        diversity_threshold=0.02,
        min_per_bar=1,
        seed=7,
    ):
        payload = {
            "vreq": validate_vreq(vreq),
            "samples_per_bar": int(samples_per_bar),
            "top_k": int(top_k),
            "bar_types": bar_types,
            "temperature": float(temperature),
            "diversity_threshold": float(diversity_threshold),
            "min_per_bar": int(min_per_bar),
            "seed": int(seed),
        }
        result = self._request_json("POST", "/generate", payload)
        return validate_generation_response(result)
