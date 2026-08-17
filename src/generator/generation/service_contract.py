"""Shared validation for the GVAE HTTP service and ROS adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


GVAE_VREQ_KEYS = ("x", "y", "z", "load", "inspect", "pack")
MIMA_VR_KEYS = ("wp_m", "hp_m", "dp_m", "hs_m", "fl", "fi", "fp")
BAR_TYPES = ("4-bar", "8-bar", "6-bar")


def _finite_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a JSON number" % field)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % field)
    return result


def validate_vreq(values):
    """Validate the native six-dimensional GVAE request."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("vreq must be an array")
    if len(values) != 6:
        raise ValueError("vreq must contain 6 values, got %d" % len(values))

    vreq = [
        _finite_number(value, "vreq[%d]" % index)
        for index, value in enumerate(values)
    ]
    if any(value <= 0.0 for value in vreq[:3]):
        raise ValueError("vreq x/y/z limits must be positive")
    if any(value < 0.0 or value > 1.0 for value in vreq[3:]):
        raise ValueError("vreq task values must be in [0, 1]")
    return vreq


def mima_vector_to_vreq(values):
    """Map [wp, hp, dp, hs, fl, fi, fp] to [dp, wp, hp, fl, fi, fp]."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("MIMA requirement vector must be an array")
    if len(values) != 7:
        raise ValueError(
            "MIMA requirement vector must contain 7 values, got %d" % len(values)
        )
    vector = [
        _finite_number(value, "v_r[%d]" % index)
        for index, value in enumerate(values)
    ]
    return validate_vreq(
        [vector[2], vector[0], vector[1], vector[4], vector[5], vector[6]]
    )


def mima_mapping_to_vreq(values):
    """Map the named seven-field MIMA response to the native GVAE request."""
    if not isinstance(values, Mapping):
        raise ValueError("v_r must be an object or a seven-value array")
    missing = [key for key in MIMA_VR_KEYS if key not in values]
    if missing:
        raise ValueError("v_r is missing fields: %s" % ", ".join(missing))
    return mima_vector_to_vreq([values[key] for key in MIMA_VR_KEYS])


def request_to_vreq(payload):
    """Extract either native ``vreq`` or MIMA ``v_r`` from a request object."""
    if not isinstance(payload, Mapping):
        raise ValueError("request body must be a JSON object")
    if "vreq" in payload:
        return validate_vreq(payload["vreq"])
    if "v_r" in payload:
        raw = payload["v_r"]
        if isinstance(raw, Mapping):
            return mima_mapping_to_vreq(raw)
        return mima_vector_to_vreq(raw)
    raise ValueError("request must contain 'vreq' or 'v_r'")


def normalize_bar_types(value):
    if value is None:
        return "auto"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("auto", "all"):
            return text
        names = [item.strip() for item in text.split(",") if item.strip()]
    elif isinstance(value, Sequence):
        names = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError("bar_types must be 'auto', 'all', or a list")
    unknown = [name for name in names if name not in BAR_TYPES]
    if unknown:
        raise ValueError("unknown bar types: %s" % ", ".join(unknown))
    if not names:
        raise ValueError("bar_types must not be empty")
    return ",".join(dict.fromkeys(names))


def _bounded_int(payload, key, default, minimum, maximum):
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s must be an integer" % key)
    if value < minimum or value > maximum:
        raise ValueError("%s must be in [%d, %d]" % (key, minimum, maximum))
    return int(value)


def _positive_float(payload, key, default, allow_zero=False):
    value = _finite_number(payload.get(key, default), key)
    invalid = value < 0.0 if allow_zero else value <= 0.0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError("%s must be %s" % (key, qualifier))
    return value


def normalize_generation_request(
    payload,
    default_samples_per_bar=64,
    default_top_k=10,
    max_samples_per_bar=4096,
    max_top_k=256,
):
    """Validate the complete POST /generate request."""
    vreq = request_to_vreq(payload)
    return {
        "vreq": vreq,
        "bar_types": normalize_bar_types(payload.get("bar_types", "auto")),
        "samples_per_bar": _bounded_int(
            payload,
            "samples_per_bar",
            default_samples_per_bar,
            1,
            max_samples_per_bar,
        ),
        "top_k": _bounded_int(payload, "top_k", default_top_k, 1, max_top_k),
        "temperature": _positive_float(payload, "temperature", 1.0),
        "diversity_threshold": _positive_float(
            payload,
            "diversity_threshold",
            0.02,
            allow_zero=True,
        ),
        "min_per_bar": _bounded_int(payload, "min_per_bar", 1, 0, max_top_k),
        "seed": _bounded_int(payload, "seed", 7, 0, 2**31 - 1),
    }


def validate_generation_response(payload):
    """Reject malformed service responses before they reach ROS messages."""
    if not isinstance(payload, Mapping):
        raise ValueError("generator response must be a JSON object")
    candidates = payload.get("candidates")
    summary = payload.get("summary")
    if not isinstance(candidates, list):
        raise ValueError("generator response is missing candidate list")
    if not isinstance(summary, Mapping):
        raise ValueError("generator response is missing summary object")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate %d must be an object" % index)
        if candidate.get("bar_type") not in BAR_TYPES:
            raise ValueError("candidate %d has invalid bar_type" % index)
        structure = candidate.get("structure")
        if not isinstance(structure, Mapping):
            raise ValueError("candidate %d is missing structure" % index)
        if len(structure.get("nodes", [])) != 8:
            raise ValueError("candidate %d must contain 8 nodes" % index)
        if len(structure.get("edges", [])) != 8:
            raise ValueError("candidate %d must contain 8 edges" % index)
    return payload
