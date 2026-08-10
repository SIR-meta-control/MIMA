import math

import pytest

from mima_vr.service_client import normalize_service_prediction


VALID_VR = {
    "wp_m": 0.4,
    "hp_m": 0.3,
    "dp_m": 0.0,
    "hs_m": 0.0,
    "fl": 0.0,
    "fi": 1.0,
    "fp": 0.0,
}


def test_normalize_service_prediction_requires_all_seven_numeric_fields():
    result = normalize_service_prediction(
        {"model": "student", "v_r": VALID_VR},
        sample_id="sample",
        scenario="tunnel",
        model_key="mllm_distilled",
    )
    assert result["v_r"] == VALID_VR
    assert result["model_key"] == "mllm_distilled"


@pytest.mark.parametrize("value", ["0.4", True, math.nan, math.inf])
def test_normalize_service_prediction_rejects_invalid_numbers(value):
    vector = dict(VALID_VR)
    vector["wp_m"] = value
    with pytest.raises(ValueError):
        normalize_service_prediction(
            {"v_r": vector},
            sample_id="sample",
            scenario="tunnel",
            model_key="student",
        )


def test_normalize_service_prediction_rejects_missing_dimensions():
    vector = dict(VALID_VR)
    del vector["fp"]
    with pytest.raises(ValueError, match="missing keys"):
        normalize_service_prediction(
            {"v_r": vector},
            sample_id="sample",
            scenario="tunnel",
            model_key="student",
        )
