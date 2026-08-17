"""In-process GVAE inference engine used exclusively by the HTTP service."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from gvae.core.constants import BAR_ORDER, BAR_TO_V
from gvae.core.training import select_device, set_seed
from gvae.networks.bar_classifier import (
    load_bar_classifier,
    predict_bar_probabilities,
)
from gvae.networks.generator_loader import load_generator
from gvae.robot.geometry import EdgeAngleRecoverer, GraphImputationLayer
from gvae.robot.inference import (
    ConstraintThresholds,
    HardConstraintDecoder,
    eligible_bar_indices,
    select_diverse_top_k,
)


class GVAEGeneratorRuntime:
    """Load the current generator once and serve repeated Top-K requests."""

    def __init__(
        self,
        model_path,
        bar_classifier_path,
        graph_imputation_path,
        device="auto",
        thresholds=None,
    ):
        self.model_path = Path(model_path)
        self.bar_classifier_path = Path(bar_classifier_path)
        self.graph_imputation_path = Path(graph_imputation_path)
        for path in (
            self.model_path,
            self.bar_classifier_path,
            self.graph_imputation_path,
        ):
            if not path.is_file():
                raise FileNotFoundError("required generator artifact not found: %s" % path)

        self.device = select_device(device)
        self.model, self.checkpoint = load_generator(self.model_path, self.device)
        self.bar_classifier, self.bar_mean, self.bar_std, _ = load_bar_classifier(
            self.bar_classifier_path,
            map_location="cpu",
        )
        graph_layer = GraphImputationLayer(self.graph_imputation_path).to(self.device)
        angle_recoverer = EdgeAngleRecoverer(
            np.asarray(
                self.checkpoint["edge_angle_static_transforms"],
                dtype=np.float32,
            )
        ).to(self.device)
        self.thresholds = thresholds or ConstraintThresholds()
        self.decoder = HardConstraintDecoder(
            graph_layer,
            angle_recoverer,
            self.thresholds,
        )

    @property
    def metadata(self):
        model_config = self.checkpoint.get("model_config", {})
        return {
            "model": self.model_path.name,
            "model_family": model_config.get("model_family"),
            "scale_mode": model_config.get("scale_mode"),
            "bar_classifier": self.bar_classifier_path.name,
            "device": str(self.device),
        }

    @staticmethod
    def _bar_indices(text, vreq):
        value = str(text).strip().lower()
        if value == "auto":
            return eligible_bar_indices(vreq)
        if value == "all":
            return list(range(len(BAR_ORDER)))

        names = [part.strip() for part in value.split(",") if part.strip()]
        unknown = [name for name in names if name not in BAR_TO_V]
        if unknown:
            raise ValueError("unknown bar types: %s" % ", ".join(unknown))
        if not names:
            raise ValueError("bar_types must not be empty")
        return list(dict.fromkeys(BAR_TO_V[name] for name in names))

    @staticmethod
    def _tensor_row(values, index):
        return values[index].detach().cpu().numpy().astype(float).tolist()

    def _candidate(self, decoded, index, vreq, v, bar_probability):
        edge_angles = decoded["edge_angles"][index, :, None]
        edges = torch.cat([edge_angles, decoded["edge_pose"][index]], dim=-1)
        checks = {
            name: bool(values[index].detach().cpu().item())
            for name, values in decoded["checks"].items()
        }
        metrics = {
            name: float(values[index].detach().cpu().item())
            for name, values in decoded["metrics"].items()
        }
        scores = {
            name: float(values[index].detach().cpu().item())
            for name, values in decoded["scores"].items()
        }
        return {
            "rank": None,
            "v": int(v),
            "bar_type": BAR_ORDER[v],
            "bar_probability": float(bar_probability),
            "valid": bool(decoded["valid"][index].detach().cpu().item()),
            "score": scores["overall"],
            "confidence": scores["confidence"],
            "scores": scores,
            "constraints": {"checks": checks, "metrics": metrics},
            "vreq": vreq.astype(float).tolist(),
            "structure": {
                "nodes": self._tensor_row(decoded["nodes"], index),
                "edges": edges.detach().cpu().numpy().astype(float).tolist(),
                "global": {
                    "scale": self._tensor_row(decoded["scale"], index),
                    "leg_base": self._tensor_row(decoded["leg_base"], index),
                    "leg_angle": self._tensor_row(decoded["leg_angle"], index),
                },
            },
        }

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
        set_seed(int(seed))
        vreq_np = np.asarray(vreq, dtype=np.float32)
        hard_eligible = eligible_bar_indices(vreq_np)
        bar_indices = self._bar_indices(bar_types, vreq_np)

        raw_bar_probabilities = predict_bar_probabilities(
            self.bar_classifier,
            vreq_np,
            self.bar_mean,
            self.bar_std,
            temperature=float(temperature),
        )
        bar_probabilities = raw_bar_probabilities.copy()
        eligibility_mask = np.zeros_like(bar_probabilities)
        eligibility_mask[hard_eligible] = 1.0
        bar_probabilities *= eligibility_mask
        probability_sum = float(bar_probabilities.sum())
        if probability_sum > 0.0:
            bar_probabilities /= probability_sum
        else:
            bar_probabilities[hard_eligible] = 1.0 / len(hard_eligible)

        valid_candidates = []
        generated_by_bar = Counter()
        valid_by_bar = Counter()
        rejection_counts = Counter()

        with torch.inference_mode():
            for v in bar_indices:
                vreq_tensor = torch.as_tensor(
                    vreq_np,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                bar_v = torch.tensor([v], dtype=torch.long, device=self.device)
                prediction = self.model.sample(
                    vreq_tensor,
                    bar_v,
                    num_samples=int(samples_per_bar),
                    temperature=float(temperature),
                )

                repeated_vreq = vreq_tensor.repeat(int(samples_per_bar), 1)
                repeated_bar = bar_v.repeat(int(samples_per_bar))
                probability = float(bar_probabilities[v])
                repeated_probability = torch.full(
                    (int(samples_per_bar),),
                    probability,
                    dtype=torch.float32,
                    device=self.device,
                )
                decoded = self.decoder.decode_and_evaluate(
                    prediction["nodes"],
                    prediction["leg_angle"],
                    prediction["scale"],
                    repeated_vreq,
                    repeated_bar,
                    repeated_probability,
                )

                for index in range(int(samples_per_bar)):
                    candidate = self._candidate(
                        decoded,
                        index,
                        vreq_np,
                        v,
                        probability,
                    )
                    bar_name = BAR_ORDER[v]
                    generated_by_bar[bar_name] += 1
                    if candidate["valid"]:
                        valid_by_bar[bar_name] += 1
                        valid_candidates.append(candidate)
                    else:
                        for name, passed in candidate["constraints"]["checks"].items():
                            if not passed:
                                rejection_counts[name] += 1

        selected = select_diverse_top_k(
            valid_candidates,
            top_k=int(top_k),
            min_distance=float(diversity_threshold),
            min_per_bar=int(min_per_bar),
        )
        return {
            "schema_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "vreq": vreq_np.astype(float).tolist(),
            "model": self.model_path.name,
            "bar_classifier": self.bar_classifier_path.name,
            "seed": int(seed),
            "raw_bar_probabilities": {
                BAR_ORDER[v]: float(raw_bar_probabilities[v])
                for v in range(len(BAR_ORDER))
            },
            "bar_probabilities_after_hard_rules": {
                BAR_ORDER[v]: float(bar_probabilities[v])
                for v in range(len(BAR_ORDER))
            },
            "sampled_bar_types": [BAR_ORDER[v] for v in bar_indices],
            "thresholds": self.thresholds.to_dict(),
            "summary": {
                "generated": int(sum(generated_by_bar.values())),
                "valid": len(valid_candidates),
                "returned": len(selected),
                "generated_by_bar": dict(generated_by_bar),
                "valid_by_bar": dict(valid_by_bar),
                "rejection_counts": dict(rejection_counts),
            },
            "candidates": selected,
        }
