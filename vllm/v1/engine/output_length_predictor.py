"""
Output-length prediction utilities for wait-time simulation.

This module provides both feature engineering and inference helpers for
tree-based regressors trained to predict total output lengths at request
admission time. The predictor consumes prompt-level features, including a
hashed semantic embedding reduced with PCA, and returns mean output-length
estimates in token space.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

import lightgbm as lgb
from vllm.logger import init_logger

LOGGER = init_logger(__name__)


@dataclass(frozen=True)
class OutputLengthPrediction:
    """Container for the predicted request length statistics."""

    mean_tokens: float


@dataclass(frozen=True)
class AdmissionFeatures:
    """Available request metadata at admission time."""

    model_id: str
    prompt_text: Optional[str]
    prompt_token_count: int


@dataclass(frozen=True)
class PromptFeatureContext:
    """Prompt-only feature cache reused across models/predictors."""

    prompt_values: np.ndarray
    semantic_projection: np.ndarray
    semantic_hash_dim: int


class HashingSemanticProjector:
    """Lightweight semantic encoder using hashing + PCA."""

    _TOKEN_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")

    def __init__(
        self,
        *,
        dimension: int,
        mean: Optional[Sequence[float]] = None,
        components: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self.dimension = max(32, int(dimension))
        self._mean = (
            np.asarray(mean, dtype=np.float32)
            if mean is not None
            else None
        )
        self._components = (
            np.asarray(components, dtype=np.float32)
            if components is not None
            else None
        )

    @staticmethod
    def _stable_hash(text: str) -> int:
        h = 2166136261
        for ch in text:
            h ^= ord(ch)
            h *= 16777619
            h &= 0xFFFFFFFF
        return h

    def hash_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._TOKEN_RE.findall(text.lower())
        if not tokens:
            return vec
        for token in tokens:
            h = self._stable_hash(token)
            idx = h % self.dimension
            sign = 1.0 if (h >> 1) & 1 else -1.0
            vec[idx] += sign
        vec /= max(len(tokens), 1)
        return vec

    def project_hashed(self, hashed: np.ndarray) -> np.ndarray:
        if self._components is None or self._mean is None:
            raise RuntimeError(
                "Semantic projector has not been initialized with PCA weights."
            )
        centered = np.asarray(hashed, dtype=np.float32) - self._mean
        return centered @ self._components.T

    def project(self, text: Optional[str]) -> np.ndarray:
        if self._components is None or self._mean is None:
            raise RuntimeError(
                "Semantic projector has not been initialized with PCA weights."
            )
        if not text:
            return np.zeros(self._components.shape[0], dtype=np.float32)
        return self.project_hashed(self.hash_text(text))

    def metadata(self) -> dict:
        return {
            "hash_dim": self.dimension,
            "pca_mean": self._mean.tolist() if self._mean is not None else None,
            "pca_components": (
                self._components.tolist() if self._components is not None else None
            ),
        }


class PromptFeatureExtractor:
    """Derives structured features from prompts at admission time."""

    _SENTENCE_RE = re.compile(r"[.!?]+")
    _LENGTH_RE = re.compile(
        r"(?P<num>\d+)\s*(?P<unit>words?|sentences?|paragraphs?|chars?|characters?)",
        re.IGNORECASE,
    )
    _BULLET_RE = re.compile(
        r"(?P<num>\d+)\s*(?:bullets?|items?|steps?|points?)",
        re.IGNORECASE,
    )
    _TURNS_RE = re.compile(
        r"(?:^|\n)\s*(user|assistant|system|human|bot)\s*[:：]",
        re.IGNORECASE,
    )
    _XML_RE = re.compile(r"<[^>]+>")
    _HF_CACHE_MODEL_RE = re.compile(r"models--[^/]+--([^/]+)", re.IGNORECASE)
    _LIST_KEYWORDS = re.compile(r"\b(list|bullet|enumerate|steps?)\b", re.IGNORECASE)
    _SUMMARY_KEYWORDS = re.compile(r"\b(summary|summarize|synopsis|tl;dr)\b", re.IGNORECASE)
    _EXPLAIN_KEYWORDS = re.compile(r"\b(explain|reason|why)\b", re.IGNORECASE)
    _CODE_KEYWORDS = re.compile(r"\b(code|function|class|python|java|c\+\+|javascript)\b", re.IGNORECASE)
    _TRANSLATE_KEYWORDS = re.compile(r"\b(translate|translation)\b", re.IGNORECASE)
    _LANGUAGE_RE = re.compile(
        r"\b(?:to|into)\s+(english|chinese|spanish|french|german|japanese|korean|hindi|arabic)\b",
        re.IGNORECASE,
    )

    BASE_FEATURES = [
        "model_id_index",
        "prompt_tokens",
        "prompt_chars",
        "prompt_lines",
        "prompt_sentences",
        "avg_chars_per_token",
        "has_length_constraint",
        "requested_words",
        "requested_bullets",
        "list_request",
        "summary_request",
        "explanation_request",
        "code_request",
        "translation_request",
        "conversation_turns",
        "has_system_text",
        "has_delimiters",
    ]

    def __init__(
        self,
        *,
        model_lookup: Dict[str, int],
        semantic_projector: HashingSemanticProjector,
        feature_order: Optional[Sequence[str]] = None,
    ) -> None:
        self._model_lookup = dict(model_lookup)
        if "__unknown__" not in self._model_lookup:
            self._model_lookup["__unknown__"] = max(self._model_lookup.values(), default=0)
        self._model_alias_lookup: Dict[str, int] = {}
        for model_key, model_index in self._model_lookup.items():
            if model_key == "__unknown__":
                continue
            for alias in self._iter_model_aliases(model_key):
                self._model_alias_lookup.setdefault(alias, model_index)
        self._semantic = semantic_projector
        semantic_dim = len(semantic_projector.metadata().get("pca_components") or [])
        if feature_order:
            self._feature_names = list(feature_order)
        else:
            self._feature_names = list(self.BASE_FEATURES)
            for idx in range(max(semantic_dim, 0)):
                self._feature_names.append(f"semantic_{idx}")

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @classmethod
    def _iter_model_aliases(cls, model_id: str) -> list[str]:
        text = str(model_id).strip()
        if not text:
            return []

        aliases: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            normalized = value.strip().lower().replace("_", "-")
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            aliases.append(normalized)

        normalized = text.replace("\\", "/")
        add(normalized)

        parts = [part for part in normalized.split("/") if part]
        if parts:
            add(parts[-1])
        if len(parts) >= 2:
            add(parts[-2])

        for match in cls._HF_CACHE_MODEL_RE.finditer(normalized):
            add(match.group(1))

        if normalized.lower().startswith("models--"):
            model_bits = [bit for bit in normalized.split("--") if bit]
            if model_bits:
                add(model_bits[-1])

        return aliases

    def model_index(self, model_id: str) -> int:
        if model_id in self._model_lookup:
            return self._model_lookup[model_id]
        for alias in self._iter_model_aliases(model_id):
            alias_index = self._model_alias_lookup.get(alias)
            if alias_index is not None:
                return alias_index

        # Final fallback: match canonical ids embedded in long path-like model names.
        normalized = str(model_id).lower().replace("\\", "/").replace("_", "-")
        for alias, alias_index in self._model_alias_lookup.items():
            if alias and alias in normalized:
                return alias_index
        return self._model_lookup["__unknown__"]

    @staticmethod
    def _count_lines(text: str) -> int:
        return text.count("\n") + 1 if text else 0

    def _count_sentences(self, text: str) -> int:
        return len(self._SENTENCE_RE.findall(text))

    def _detect_length_constraint(self, text: str) -> tuple[bool, int, int]:
        has_constraint = False
        requested_words = 0
        requested_bullets = 0
        for match in self._LENGTH_RE.finditer(text):
            has_constraint = True
            unit = match.group("unit").lower()
            value = int(match.group("num"))
            if unit.startswith("word"):
                requested_words = max(requested_words, value)
        for match in self._BULLET_RE.finditer(text):
            has_constraint = True
            requested_bullets = max(requested_bullets, int(match.group("num")))
        return has_constraint, requested_words, requested_bullets

    def _conversation_turns(self, text: str) -> int:
        turns = len(self._TURNS_RE.findall(text))
        if turns == 0:
            turns = text.count("<|start_header_id|>")
        return turns

    def _has_system_text(self, text_lower: str) -> bool:
        if not text_lower:
            return False
        if "<<sys>>" in text_lower or "system prompt" in text_lower:
            return True
        return "system:" in text_lower or "[system]" in text_lower

    def _has_delimiters(self, text: str) -> bool:
        return "```" in text or bool(self._XML_RE.search(text))

    def build_prompt_context(
        self,
        *,
        prompt_text: Optional[str],
        prompt_token_count: int,
    ) -> PromptFeatureContext:
        text = prompt_text or ""
        text_lower = text.lower()
        prompt_chars = len(text)
        prompt_tokens = max(prompt_token_count, 0)
        lines = self._count_lines(text)
        sentences = self._count_sentences(text)
        avg_chars = (prompt_chars / prompt_tokens) if prompt_tokens > 0 else 0.0
        has_length, requested_words, requested_bullets = self._detect_length_constraint(text)
        list_request = (
            requested_bullets > 0
            or bool(self._LIST_KEYWORDS.search(text))
            or "\n- " in text
            or "\n* " in text
        )
        summary_request = bool(self._SUMMARY_KEYWORDS.search(text))
        explanation_request = bool(self._EXPLAIN_KEYWORDS.search(text))
        code_request = bool(self._CODE_KEYWORDS.search(text)) or "```" in text
        translation_request = bool(self._TRANSLATE_KEYWORDS.search(text)) or bool(
            self._LANGUAGE_RE.search(text)
        )
        conversation_turns = self._conversation_turns(text)
        has_system_text = self._has_system_text(text_lower)
        has_delimiters = self._has_delimiters(text)

        if text:
            semantic = self._semantic.project_hashed(self._semantic.hash_text(text))
        else:
            semantic = np.zeros(
                len(self._feature_names) - len(self.BASE_FEATURES),
                dtype=np.float32,
            )

        prompt_values = np.asarray(
            [
                float(prompt_tokens),
                float(prompt_chars),
                float(lines),
                float(sentences),
                float(avg_chars),
                1.0 if has_length else 0.0,
                float(requested_words),
                float(requested_bullets),
                1.0 if list_request else 0.0,
                1.0 if summary_request else 0.0,
                1.0 if explanation_request else 0.0,
                1.0 if code_request else 0.0,
                1.0 if translation_request else 0.0,
                float(conversation_turns),
                1.0 if has_system_text else 0.0,
                1.0 if has_delimiters else 0.0,
            ],
            dtype=np.float32,
        )
        return PromptFeatureContext(
            prompt_values=prompt_values,
            semantic_projection=np.asarray(semantic, dtype=np.float32),
            semantic_hash_dim=self._semantic.dimension,
        )

    def _context_compatible(self, prompt_context: PromptFeatureContext) -> bool:
        semantic = np.asarray(prompt_context.semantic_projection, dtype=np.float32)
        semantic_dim = len(self._feature_names) - len(self.BASE_FEATURES)
        return (
            int(prompt_context.semantic_hash_dim) == int(self._semantic.dimension)
            and prompt_context.prompt_values.shape == (16,)
            and semantic.shape == (semantic_dim,)
        )

    def build_feature_row_from_context(
        self,
        *,
        model_id: str,
        prompt_context: PromptFeatureContext,
    ) -> np.ndarray:
        semantic = np.asarray(prompt_context.semantic_projection, dtype=np.float32)
        prompt_values = np.asarray(prompt_context.prompt_values, dtype=np.float32)
        feature_vector = np.concatenate(
            [
                np.asarray([float(self.model_index(model_id))], dtype=np.float32),
                prompt_values,
            ]
        )
        if semantic.size:
            feature_vector = np.concatenate([feature_vector, semantic])
        return feature_vector.astype(np.float32)

    def build_feature_row(
        self,
        admission: AdmissionFeatures,
        *,
        prompt_context: Optional[PromptFeatureContext] = None,
    ) -> np.ndarray:
        if (
            prompt_context is not None
            and self._context_compatible(prompt_context)
        ):
            return self.build_feature_row_from_context(
                model_id=admission.model_id,
                prompt_context=prompt_context,
            )

        prompt_text = admission.prompt_text or ""
        prompt_lower = prompt_text.lower()
        prompt_chars = len(prompt_text)
        prompt_tokens = max(admission.prompt_token_count, 0)
        lines = self._count_lines(prompt_text)
        sentences = self._count_sentences(prompt_text)
        avg_chars = (prompt_chars / prompt_tokens) if prompt_tokens > 0 else 0.0
        has_length, requested_words, requested_bullets = self._detect_length_constraint(prompt_text)
        list_request = (
            requested_bullets > 0
            or bool(self._LIST_KEYWORDS.search(prompt_text))
            or "\n- " in prompt_text
            or "\n* " in prompt_text
        )
        summary_request = bool(self._SUMMARY_KEYWORDS.search(prompt_text))
        explanation_request = bool(self._EXPLAIN_KEYWORDS.search(prompt_text))
        code_request = bool(self._CODE_KEYWORDS.search(prompt_text)) or "```" in prompt_text
        translation_request = bool(self._TRANSLATE_KEYWORDS.search(prompt_text)) or bool(
            self._LANGUAGE_RE.search(prompt_text)
        )
        conversation_turns = self._conversation_turns(prompt_text)
        has_system_text = self._has_system_text(prompt_lower)
        has_delimiters = self._has_delimiters(prompt_text)

        semantic = self._semantic.project(prompt_text)
        base_values = [
            float(self.model_index(admission.model_id)),
            float(prompt_tokens),
            float(prompt_chars),
            float(lines),
            float(sentences),
            float(avg_chars),
            1.0 if has_length else 0.0,
            float(requested_words),
            float(requested_bullets),
            1.0 if list_request else 0.0,
            1.0 if summary_request else 0.0,
            1.0 if explanation_request else 0.0,
            1.0 if code_request else 0.0,
            1.0 if translation_request else 0.0,
            float(conversation_turns),
            1.0 if has_system_text else 0.0,
            1.0 if has_delimiters else 0.0,
        ]
        feature_vector = np.asarray(base_values, dtype=np.float32)
        if semantic.size:
            feature_vector = np.concatenate([feature_vector, semantic.astype(np.float32)])
        return feature_vector


class OutputLengthPredictor:
    """Loads LightGBM-based regressors and provides admission-time estimates."""

    def __init__(
        self,
        model_dir: str,
        *,
        preferred_quantile: Optional[float] = None,
    ) -> None:
        self._model_dir = Path(model_dir)
        metadata_path = self._model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Output-length metadata not found at {metadata_path}"
            )
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        semantic = HashingSemanticProjector(
            dimension=metadata["hash_dim"],
            mean=metadata["pca_mean"],
            components=metadata["pca_components"],
        )
        feature_order = metadata.get("feature_order")
        self._feature_extractor = PromptFeatureExtractor(
            model_lookup=metadata["model_id_lookup"],
            semantic_projector=semantic,
            feature_order=feature_order,
        )

        mean_model_filename = metadata.get("mean_model_file", "mean_model.txt")
        mean_model_path = self._model_dir / mean_model_filename
        self._mean_model_outputs_log = False
        if mean_model_path.exists():
            self._mean_model = self._load_booster(mean_model_path)
            mean_model_target = metadata.get("mean_model_target", "tokens")
            self._mean_model_outputs_log = mean_model_target != "tokens"
        else:
            LOGGER.warning(
                "Mean output-length model not found at %s; using median head as mean fallback.",
                mean_model_path,
            )
            self._mean_model = self._load_booster(self._model_dir / "median_model.txt")
            self._mean_model_outputs_log = True

        self._lock = threading.Lock()

    @staticmethod
    def _load_booster(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Missing LightGBM model at {path}")
        booster = lgb.Booster(model_file=str(path))
        return booster

    def predict(
        self,
        admission: AdmissionFeatures,
        *,
        prompt_context: Optional[PromptFeatureContext] = None,
    ) -> Optional[OutputLengthPrediction]:
        """Return mean output-length estimate (tokens) for the provided request."""
        predictions = self.predict_batch(
            [admission],
            prompt_context=prompt_context,
        )
        return predictions[0] if predictions else None

    def predict_batch(
        self,
        admissions: Sequence[AdmissionFeatures],
        *,
        prompt_context: Optional[PromptFeatureContext] = None,
    ) -> list[Optional[OutputLengthPrediction]]:
        """Predict mean output lengths for many request/model pairs in one pass."""
        if not admissions:
            return []
        feature_rows: list[np.ndarray] = []
        valid_indices: list[int] = []
        results: list[Optional[OutputLengthPrediction]] = [None] * len(admissions)
        for idx, admission in enumerate(admissions):
            if admission.prompt_token_count <= 0:
                continue
            features = self._feature_extractor.build_feature_row(
                admission,
                prompt_context=prompt_context,
            )
            feature_rows.append(features)
            valid_indices.append(idx)
        if not feature_rows:
            return results
        feature_matrix = np.vstack(feature_rows)
        with self._lock:
            mean_preds = self._mean_model.predict(feature_matrix)

        for offset, admission_idx in enumerate(valid_indices):
            mean_pred = float(mean_preds[offset])
            if self._mean_model_outputs_log:
                mean_tokens = max(math.expm1(mean_pred), 1.0)
            else:
                mean_tokens = max(mean_pred, 1.0)
            results[admission_idx] = OutputLengthPrediction(
                mean_tokens=mean_tokens,
            )
        return results


def load_predictor_if_available(
    config: Optional[dict],
) -> Optional[OutputLengthPredictor]:
    """Helper for lazy construction through SchedulerConfig dicts."""
    if not config:
        return None
    model_dir = config.get("output_length_model_path")
    if not model_dir:
        return None
    model_dir = os.path.expandvars(os.path.expanduser(model_dir))
    try:
        return OutputLengthPredictor(
            model_dir,
            preferred_quantile=config.get("output_length_tail_quantile"),
        )
    except Exception as exc:  # pragma: no cover - log and disable gracefully
        LOGGER.warning(
            "Failed to initialize output-length predictor from %s: %s",
            model_dir,
            exc,
        )
        return None
