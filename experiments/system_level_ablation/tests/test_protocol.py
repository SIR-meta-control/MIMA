from pathlib import Path

import pytest

from mima_ablation.config import resolve_under, validate_run_assets
from mima_ablation.methods import METHOD_ORDER, METHOD_SPECS
from mima_ablation.records import strict_success


def test_method_order_and_component_changes_match_reported_table():
    assert METHOD_ORDER == (
        "Full-MIMA",
        "MLLM-distilled",
        "MLLM -> RF",
        "MLLM -> DT",
        "MLLM -> GBT",
        "cVAE -> MLP",
        "w/o Energy optimizer",
    )
    without_energy = METHOD_SPECS["without_energy_optimizer"]
    assert without_energy.requirement_source == "full_mima_teacher"
    assert without_energy.structure_generator == "cvae"
    assert without_energy.use_energy_optimizer is False
    assert without_energy.posthoc_energy_audit is True


def test_strict_success_uses_chain_physical_and_prediction_conditions():
    row = {
        "sample_id": "sample_w0.50_h0.35",
        "success": True,
        "actual_wp_m": 0.49,
        "actual_hp_m": 0.34,
        "pred_wp_m": 0.5105,
        "pred_hp_m": 0.3395,
    }
    assert strict_success(row, 0.0105)
    assert not strict_success({**row, "success": False}, 0.0105)
    assert not strict_success({**row, "actual_wp_m": 0.51}, 0.0105)
    assert not strict_success({**row, "pred_hp_m": 0.3394}, 0.0105)


def test_bundle_paths_cannot_escape_bundle_root():
    bundle_root = (Path.cwd() / "bundle_root").resolve()
    assert resolve_under(bundle_root, "raw/details.csv") == bundle_root / "raw" / "details.csv"
    with pytest.raises(ValueError, match="bundle-relative"):
        resolve_under(bundle_root, str(Path(bundle_root.anchor) / "outside.csv"))
    with pytest.raises(ValueError, match="escapes bundle"):
        resolve_under(bundle_root, "../outside.csv")


def test_run_config_rejects_unset_assets():
    config = {
        "assets": {
            "requirement_sources": {"full_mima_teacher": ""},
            "structure_models": {"cvae": ""},
            "energy_model": "",
            "robot_model": "",
        }
    }
    with pytest.raises(ValueError, match="unset required assets"):
        validate_run_assets(config, [METHOD_SPECS["full_mima"]])
