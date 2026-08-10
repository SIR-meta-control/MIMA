"""Canonical method identities and component-level ablation definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    requirement_source: str
    structure_generator: str
    use_energy_optimizer: bool
    candidate_selection_policy: str
    posthoc_energy_audit: bool = False


METHOD_SPECS = {
    "full_mima": MethodSpec(
        key="full_mima",
        label="Full-MIMA",
        requirement_source="full_mima_teacher",
        structure_generator="cvae",
        use_energy_optimizer=True,
        candidate_selection_policy="energy_ranked_feasible_fallback",
    ),
    "mllm_distilled": MethodSpec(
        key="mllm_distilled",
        label="MLLM-distilled",
        requirement_source="teacher_only_et32",
        structure_generator="cvae",
        use_energy_optimizer=True,
        candidate_selection_policy="energy_ranked_feasible_fallback",
    ),
    "mllm_to_rf": MethodSpec(
        key="mllm_to_rf",
        label="MLLM -> RF",
        requirement_source="rf",
        structure_generator="cvae",
        use_energy_optimizer=True,
        candidate_selection_policy="energy_ranked_feasible_fallback",
    ),
    "mllm_to_dt": MethodSpec(
        key="mllm_to_dt",
        label="MLLM -> DT",
        requirement_source="dt",
        structure_generator="cvae",
        use_energy_optimizer=True,
        candidate_selection_policy="energy_ranked_feasible_fallback",
    ),
    "mllm_to_gbt": MethodSpec(
        key="mllm_to_gbt",
        label="MLLM -> GBT",
        requirement_source="gbt",
        structure_generator="cvae",
        use_energy_optimizer=True,
        candidate_selection_policy="energy_ranked_feasible_fallback",
    ),
    "cvae_to_mlp": MethodSpec(
        key="cvae_to_mlp",
        label="cVAE -> MLP",
        requirement_source="full_mima_teacher",
        structure_generator="mlp",
        use_energy_optimizer=True,
        candidate_selection_policy="energy_ranked_feasible_fallback",
    ),
    "without_energy_optimizer": MethodSpec(
        key="without_energy_optimizer",
        label="w/o Energy optimizer",
        requirement_source="full_mima_teacher",
        structure_generator="cvae",
        use_energy_optimizer=False,
        candidate_selection_policy="first_generated_candidate",
        posthoc_energy_audit=True,
    ),
}

METHOD_ORDER = tuple(spec.label for spec in METHOD_SPECS.values())

_ALIASES = {
    "full-mllm": "Full-MIMA",
    "full-mima": "Full-MIMA",
    "mllm-distilled": "MLLM-distilled",
    "mima-distilled": "MLLM-distilled",
    "w/o mllm-rf": "MLLM -> RF",
    "w/o mllm -> rf": "MLLM -> RF",
    "mllm -> rf": "MLLM -> RF",
    "w/o mllm-dt": "MLLM -> DT",
    "w/o mllm -> dt": "MLLM -> DT",
    "mllm -> dt": "MLLM -> DT",
    "w/o mllm-gbt": "MLLM -> GBT",
    "w/o mllm -> gbt": "MLLM -> GBT",
    "mllm -> gbt": "MLLM -> GBT",
    "w/o cvae -> mlp": "cVAE -> MLP",
    "cvae -> mlp": "cVAE -> MLP",
    "mlp_direct": "cVAE -> MLP",
    "w/o energy optimizer": "w/o Energy optimizer",
}


def canonical_method(value: str) -> str:
    normalized = " ".join(value.strip().split()).lower()
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown method label: {value!r}") from exc


def selected_specs(keys: list[str] | tuple[str, ...]) -> tuple[MethodSpec, ...]:
    unknown = sorted(set(keys) - set(METHOD_SPECS))
    if unknown:
        raise ValueError(
            f"unknown method keys {unknown}; choose from {sorted(METHOD_SPECS)}"
        )
    if not keys:
        raise ValueError("at least one method key is required")
    return tuple(METHOD_SPECS[key] for key in keys)
