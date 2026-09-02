from typing import Any, Mapping

import numpy as np
import xgboost

from voice2gender._model import booster, FEATURE_NAMES
from voice2gender._preprocess import (
    extract_features_from_pcm_sequence,
    Sequence,
    SAMPLE_RATE,
)


def _predict(
    features: Mapping[str, float], include_features: bool = False
) -> dict[str, Any | float]:
    if not isinstance(features, Mapping):
        raise TypeError("features must be a mapping")
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    vector = np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Model features contain NaN or infinite values")
    prediction_matrix = xgboost.DMatrix(vector.reshape(1, -1), feature_names=FEATURE_NAMES)
    female_probability = float(booster.predict(prediction_matrix)[0])
    male_probability = 1.0 - female_probability
    confidence = abs(female_probability - 0.5) * 2.0
    result: dict[str, Any | float] = {
        "male_probability": male_probability,
        "female_probability": female_probability,
        "confidence": confidence,
    }
    if include_features:
        result["features"] = {name: float(features[name]) for name in FEATURE_NAMES}
    return result


def predict(
    pcm_sequence: Sequence[bytes],
    sample_rate: int = SAMPLE_RATE,
    include_features: bool = False,
) -> dict[str, Any | float]:
    return _predict(
        extract_features_from_pcm_sequence(
            pcm_sequence=pcm_sequence, sample_rate=sample_rate
        ),
        include_features=include_features,
    )
