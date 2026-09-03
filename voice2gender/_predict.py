"""Run the bundled gender classifier on extracted acoustic features.

The public prediction function keeps 44,100 Hz as its default input rate, while
passing an explicitly supplied positive integer rate through feature extraction.
"""

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
    """Predict gender probabilities from one ordered feature mapping.

    Args:
        features: Mapping containing every feature named in FEATURE_NAMES.
        include_features: Whether to include the input features in the result.

    Returns:
        A mapping with male probability, female probability, confidence, and
        optionally the model input features.

    Raises:
        TypeError: features is not a mapping.
        ValueError: A required feature is missing or contains a non-finite value.
    """
    if not isinstance(features, Mapping):
        raise TypeError("features must be a mapping")
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    vector = np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("Model features contain NaN or infinite values")
    prediction_matrix = xgboost.DMatrix(vector.reshape(1, -1), feature_names=FEATURE_NAMES)
    # The XGBoost output is the female-class probability; the complementary
    # value is exposed as the male-class probability.
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
    """Extract features and return the bundled model's gender probabilities.

    Args:
        pcm_sequence: Consecutive mono int16 little-endian PCM chunks from one speaker.
        sample_rate: Positive integer PCM sample rate in Hz; defaults to 44,100 Hz
            and may be overridden by the caller.
        include_features: Whether to include the extracted features in the result.

    Returns:
        A mapping containing male probability, female probability, and confidence,
        plus features when include_features is true.

    Raises:
        TypeError: The PCM sequence or one of its chunks has an invalid type.
        ValueError: The sample rate or audio data is invalid for feature extraction.
    """
    return _predict(
        extract_features_from_pcm_sequence(
            pcm_sequence=pcm_sequence, sample_rate=sample_rate
        ),
        include_features=include_features,
    )
