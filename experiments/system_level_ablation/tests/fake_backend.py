"""Deterministic contract fixture; not an experimental model implementation."""

from mima_ablation.records import parse_gt


def run_request(request):
    wp_m, hp_m = parse_gt(request["sample"]["sample_id"])
    if request["mode"] == "timing":
        return {"success": True, "execution_time_s": 0.25}
    return {
        "success": True,
        "predicted_v_r": {"wp_m": wp_m, "hp_m": hp_m},
        "actual_v_r": {"wp_m": wp_m - 0.01, "hp_m": hp_m - 0.01},
        "energy_j": 100.0 + request["seed"],
    }
