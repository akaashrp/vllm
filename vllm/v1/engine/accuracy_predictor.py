"""
Raw accuracy predictor for routing across heterogeneous models.

This module loads the LightGBM regression booster trained by
``vllm_utils/train_accuracy_regressor.py`` and exposes a thread-safe scorer that
produces accuracy estimates :math:`\\hat A(x, m)` for each request/model pair.
If the training metadata specifies a link function (e.g. ``logit``), the
predictor applies the corresponding inverse link at inference time.

Callers are responsible for combining this accuracy estimate with explicit
cost / wait-time (TTFT) penalties during routing:
  U(x,m) = \\hat A(x,m) - \\lambda * Cost(x,m) - \\delta * TTFT(x,m).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

import lightgbm as lgb
from vllm.logger import init_logger
from vllm.v1.engine.output_length_predictor import (
    AdmissionFeatures,
    HashingSemanticProjector,
    PromptFeatureExtractor,
)

LOGGER = init_logger(__name__)


@dataclass(frozen=True)
class ModelDescriptor:
    """Static, admission-time descriptors for a specific model."""

    log_num_params: float
    max_context_length: float


class AccuracyFeatureBuilder:
    """Composes request + model features for the accuracy regression model."""

    EXTRA_FEATURES = [
        "model_log_num_params",
        "model_max_context_length",
    ]

    def __init__(
        self,
        *,
        prompt_extractor: PromptFeatureExtractor,
        model_descriptors: Mapping[str, Mapping[str, float]],
        feature_order: Optional[Sequence[str]] = None,
    ) -> None:
        self._prompt_extractor = prompt_extractor
        self._feature_order = (
            list(feature_order)
            if feature_order
            else prompt_extractor.feature_names + list(self.EXTRA_FEATURES)
        )
        parsed: Dict[str, ModelDescriptor] = {}
        for key, value in model_descriptors.items():
            parsed[key] = ModelDescriptor(
                log_num_params=float(value.get("log_num_params", 0.0)),
                max_context_length=float(value.get("max_context_length", 0.0)),
            )
        if "__default__" not in parsed:
            parsed["__default__"] = ModelDescriptor(0.0, 0.0)
        self._model_descriptors = parsed

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_order)

    def _descriptor_for(self, model_id: str) -> ModelDescriptor:
        if model_id in self._model_descriptors:
            return self._model_descriptors[model_id]
        return self._model_descriptors["__default__"]

    def build_feature_row(self, admission: AdmissionFeatures) -> np.ndarray:
        base = self._prompt_extractor.build_feature_row(admission)
        desc = self._descriptor_for(admission.model_id)
        extra = np.asarray(
            [desc.log_num_params, desc.max_context_length],
            dtype=np.float32,
        )
        return np.concatenate([base, extra]).astype(np.float32)


class AccuracyPredictor:
    """Thread-safe LightGBM scorer for raw accuracy estimates."""

    def __init__(
        self,
        model_dir: str,
    ) -> None:
        self._model_dir = Path(model_dir)
        metadata_path = self._model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Accuracy predictor metadata not found at {metadata_path}"
            )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self._target_link = str(metadata.get("target_link", "identity")).strip().lower()
        if self._target_link not in {"identity", "logit"}:
            raise ValueError(
                f"Unsupported accuracy predictor target_link={self._target_link!r}"
            )

        semantic = HashingSemanticProjector(
            dimension=metadata["hash_dim"],
            mean=metadata["pca_mean"],
            components=metadata["pca_components"],
        )
        prompt_extractor = PromptFeatureExtractor(
            model_lookup=metadata["model_id_lookup"],
            semantic_projector=semantic,
            feature_order=metadata.get("prompt_feature_order"),
        )
        self._feature_builder = AccuracyFeatureBuilder(
            prompt_extractor=prompt_extractor,
            model_descriptors=metadata["model_descriptors"],
            feature_order=metadata.get("feature_order"),
        )

        # Regression model saved by the updated trainer.
        model_path = self._model_dir / "accuracy_model.txt"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing LightGBM model for accuracy predictor at {model_path}"
            )
        self._booster = lgb.Booster(model_file=str(model_path))
        self._lock = threading.Lock()

    def _inverse_link(self, value: float) -> float:
        if self._target_link == "identity":
            return value
        if self._target_link == "logit":
            return float(1.0 / (1.0 + np.exp(-value)))
        raise ValueError(f"Unsupported target link: {self._target_link!r}")

    def predict(self, admission: AdmissionFeatures) -> float:
        """Return \\hat A(x, m) for a single request/model pair."""
        if admission.prompt_token_count <= 0:
            raise ValueError("prompt_token_count must be positive for scoring.")
        features = self._feature_builder.build_feature_row(admission)
        with self._lock:
            pred = float(self._booster.predict(features.reshape(1, -1))[0])
        return self._inverse_link(pred)

    def predict_batch(self, admissions: Sequence[AdmissionFeatures]) -> list[float]:
        """Predict accuracy for multiple request/model pairs in a single LightGBM call."""
        if not admissions:
            return []
        feature_matrix = np.vstack(
            [self._feature_builder.build_feature_row(adm) for adm in admissions]
        )
        with self._lock:
            preds = self._booster.predict(feature_matrix)
        return [self._inverse_link(float(val)) for val in preds]


def load_predictor_if_available(
    config: Optional[dict],
) -> Optional[AccuracyPredictor]:
    """
    Helper mirroring ``output_length_predictor.load_predictor_if_available``.

    Expected config key:
      - accuracy_model_path: path to directory containing metadata.json and accuracy_model.txt
    """
    if not config:
        return None
    model_dir = config.get("accuracy_model_path")
    if not model_dir:
        return None
    model_dir = str(Path(model_dir).expanduser())
    try:
        return AccuracyPredictor(model_dir)
    except Exception as exc:  # pragma: no cover - best effort initialization
        LOGGER.warning(
            "Failed to initialize accuracy predictor from %s: %s",
            model_dir,
            exc,
        )
        return None
