# Full-Chain Backend API

The batch scripts accept a Python callable using the form:

```text
python.module:function
```

The callable receives one JSON-compatible request dictionary and returns one
mapping. It is responsible for model loading, requirement inference, structure
generation, energy ranking, motion planning, optional replay, and geometry
measurement.

## Request

```python
{
    "mode": "success_energy",  # or "timing"
    "method": {
        "key": "full_mima",
        "label": "Full-MIMA",
        "requirement_source": "full_mima_teacher",
        "structure_generator": "cvae",
        "use_energy_optimizer": True,
        "candidate_selection_policy": "energy_ranked_feasible_fallback",
        "posthoc_energy_audit": False,
    },
    "seed": 1,
    "sample": {
        "sample_id": "...",
        "scenario": "tunnel",
        "rgb_path": "/resolved/path/rgb.png",
        "depth_path": "/resolved/path/depth.npy",
        "point_cloud_path": "/resolved/path/point_cloud.npy",
        # Other manifest fields are preserved.
    },
    "assets": {
        "requirement_sources": {...},
        "structure_models": {...},
        "energy_model": "...",
        "robot_model": "...",
    },
    "protocol": {...},
}
```

The backend must use the supplied method definition. In particular, the
`without_energy_optimizer` method selects the first generated candidate and
may compute energy only after selection; post-hoc energy must not influence the
chosen candidate.

## Success and energy response

```python
{
    "success": True,
    "predicted_v_r": {"wp_m": 0.42, "hp_m": 0.30},
    "actual_v_r": {"wp_m": 0.41, "hp_m": 0.29},
    "energy_j": 138.2,
    "error_type": "",
    "error": "",
}
```

Flat `pred_wp_m`, `pred_hp_m`, `actual_wp_m`, and `actual_hp_m` fields are also
accepted. A missing or non-finite energy value is retained as an unavailable
estimate and excluded from energy mean/SD calculations; it must not be replaced
with zero.

## Timing response

```python
{
    "success": True,
    "execution_time_s": 0.74,
    "error_type": "",
    "error": "",
}
```

`execution_time_s` must use the command-ready boundary defined in the module
README. It must be measured inside the backend so one-time Python import,
process creation, logging, file serialization, replay, and post-hoc audit are
not accidentally included. A timing backend should load and warm its models
before the first measured request.

## Error handling

The backend may raise an exception or return `error_type` and `error`. Batch
scripts preserve failures as rows rather than dropping them. Scientific code
must not convert failed executions into successful rows or impute unavailable
energy/timing values.
