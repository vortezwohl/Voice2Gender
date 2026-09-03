"""Extract acoustic features from PCM audio for gender-model inference.

The module keeps 44,100 Hz as the default sample rate while allowing callers to
provide any other positive integer rate explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from voice2gender._model import FEATURE_NAMES

SAMPLE_RATE = 44_100
TRACK_FRAME_LENGTH = 2_048
F0_FRAME_STEP = TRACK_FRAME_LENGTH // 2
DOMINANT_FRAME_STEP = TRACK_FRAME_LENGTH
MIN_F0_HZ = 50.0
MAX_F0_HZ = 280.0
SPECTRUM_MAX_HZ = 280.0
DOMINANT_MAX_HZ = 7_000.0
AMPLITUDE_THRESHOLD_PERCENT = 5.0
ENERGY_THRESHOLD_RATIO = AMPLITUDE_THRESHOLD_PERCENT / 100.0
EPSILON = 1e-12


def _decode_pcm_sequence(pcm_sequence: Sequence[bytes], sample_rate: int) -> np.ndarray:
    """Decode little-endian int16 PCM chunks and normalize them to [-1, 1).

    Args:
        pcm_sequence: Mono PCM byte chunks ordered by time.
        sample_rate: Positive integer PCM sample rate in Hz. It is validated
            here and then used by downstream frequency calculations; it may
            differ from :data:`SAMPLE_RATE`.

    Returns:
        A float64 mono sample array.

    Raises:
        TypeError: The PCM sequence or one of its elements has an invalid type.
        ValueError: The sample rate, byte length, or audio duration is invalid.
    """
    if isinstance(pcm_sequence, (bytes, bytearray, memoryview)) or not isinstance(
        pcm_sequence, Sequence
    ):
        raise TypeError("pcm_sequence must be an ordered sequence of bytes")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, np.integer))
        or sample_rate <= 0
    ):
        raise ValueError("sample_rate must be a positive integer")
    if not pcm_sequence:
        raise ValueError("pcm_sequence cannot be empty")
    if any(not isinstance(chunk, bytes) for chunk in pcm_sequence):
        raise TypeError("Every pcm_sequence element must be bytes")
    raw = b"".join(pcm_sequence)
    if not raw:
        raise ValueError("PCM data cannot be empty")
    if len(raw) % 2:
        raise ValueError("The int16 PCM byte length must be even")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if samples.size < TRACK_FRAME_LENGTH:
        raise ValueError(
            f"PCM must contain at least one complete {TRACK_FRAME_LENGTH}-sample analysis frame"
        )
    peak = float(np.max(np.abs(samples)))
    if peak <= EPSILON:
        raise ValueError("PCM contains no usable non-silent signal")
    return samples


def _frame_signal(
    samples: np.ndarray,
    frame_length: int,
    frame_step: int,
    *,
    apply_window: bool = True,
    pad_end: bool = True,
) -> np.ndarray:
    """Split a signal into frames with configurable windowing and end padding.

    Args:
        samples: Normalized mono samples.
        frame_length: Analysis frame length.
        frame_step: Distance between consecutive frame starts.
        apply_window: Whether to apply a Hann window.
        pad_end: Whether to retain and zero-pad the final incomplete frame.

    Returns:
        An analysis frame matrix with shape (frame count, frame_length).

    Raises:
        ValueError: The frame length or step is invalid, or samples are insufficient.
    """
    if frame_length <= 0 or frame_step <= 0:
        raise ValueError("frame_length and frame_step must be positive integers")
    if samples.size < frame_length:
        raise ValueError("Not enough samples for one complete analysis frame")
    if pad_end:
        frame_count = int(np.ceil((samples.size - frame_length) / frame_step)) + 1
    else:
        frame_count = (samples.size - frame_length) // frame_step + 1
    padded_size = (frame_count - 1) * frame_step + frame_length
    padded = np.pad(samples, (0, max(0, padded_size - samples.size)))
    starts = np.arange(frame_count) * frame_step
    frames = padded[starts[:, None] + np.arange(frame_length)]
    if apply_window:
        frames = frames * np.hanning(frame_length)
    return frames


def _spectrum_features(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Compute whole-signal spectral statistics following seewave::specprop.

    Args:
        samples: Original normalized mono PCM samples; threshold gating is not applied.
        sample_rate: Caller-selected PCM sample rate in Hz.

    Returns:
        Twelve spectral features matching voice.csv names and units; frequencies are in kHz.

    Raises:
        ValueError: The 0--280 Hz band has no energy or spectral statistics are degenerate.
    """
    sample_count = samples.size
    if sample_count < 2:
        raise ValueError("Not enough samples to compute the spectrum")
    windowed = samples * np.hanning(sample_count)
    spectrum = np.abs(np.fft.rfft(windowed))[: sample_count // 2]
    frequencies_hz = np.arange(sample_count // 2, dtype=np.float64)
    frequencies_hz *= sample_rate / sample_count
    band = frequencies_hz <= SPECTRUM_MAX_HZ
    frequencies_hz = frequencies_hz[band]
    amplitudes = spectrum[band]
    total = float(np.sum(amplitudes))
    if total <= EPSILON:
        raise ValueError("The 0--280 Hz band has no usable spectral energy")
    probabilities = amplitudes / total
    cumulative = np.cumsum(probabilities)

    mean_hz = float(np.sum(probabilities * frequencies_hz))
    variance_hz = float(np.sum(probabilities * (frequencies_hz - mean_hz) ** 2))
    sd_hz = float(np.sqrt(max(variance_hz, 0.0)))
    amplitude_mean = float(np.mean(probabilities))
    amplitude_sd = float(np.std(probabilities, ddof=1))
    if amplitude_sd <= EPSILON:
        skew = 0.0
        kurt = 0.0
    else:
        centered_amplitudes = probabilities - amplitude_mean
        skew = float(
            np.sum(centered_amplitudes**3)
            / ((probabilities.size - 1) * amplitude_sd**3)
        )
        kurt = float(
            np.sum(centered_amplitudes**4)
            / ((probabilities.size - 1) * amplitude_sd**4)
        )

    entropy_probabilities = np.maximum(probabilities, 1e-7)
    entropy_probabilities /= float(np.sum(entropy_probabilities))
    entropy = float(
        -np.sum(entropy_probabilities * np.log(entropy_probabilities))
        / np.log(probabilities.size)
    )

    flatness_values = probabilities
    if flatness_values.size > 4000:
        undersample_step = max(1, int(round(flatness_values.size / 256)))
        flatness_values = flatness_values[::undersample_step]
    flatness_values = np.maximum(flatness_values, 1e-5)
    sfm = float(
        np.exp(np.mean(np.log(flatness_values)))
        / max(float(np.mean(flatness_values)), EPSILON)
    )

    def quantile_frequency(level: float) -> float:
        """Return the first frequency whose cumulative energy reaches a quantile.

        Args:
            level: Cumulative-energy quantile between zero and one.

        Returns:
            The corresponding frequency in Hz.
        """
        index = int(np.searchsorted(cumulative, level, side="right"))
        index = min(index, frequencies_hz.size - 1)
        return float(frequencies_hz[index])

    median_hz = quantile_frequency(0.5)
    q25_hz = quantile_frequency(0.25)
    q75_hz = quantile_frequency(0.75)
    mode_hz = float(frequencies_hz[int(np.argmax(probabilities))])
    return {
        "meanfreq": mean_hz / 1000.0,
        "sd": sd_hz / 1000.0,
        "median": median_hz / 1000.0,
        "Q25": q25_hz / 1000.0,
        "Q75": q75_hz / 1000.0,
        "IQR": (q75_hz - q25_hz) / 1000.0,
        "skew": skew,
        "kurt": kurt,
        "sp.ent": entropy,
        "sfm": sfm,
        "mode": mode_hz / 1000.0,
        "centroid": mean_hz / 1000.0,
    }


def _track_f0_and_dominant(
    samples: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extract F0 and dominant frequency using seewave::fund and seewave::dfreq frame definitions.

    Args:
        samples: Normalized mono samples.
        sample_rate: Caller-selected PCM sample rate in Hz.

    Returns:
        Two Hz sequences for valid F0 frames and dominant-frequency frames.

    Raises:
        ValueError: No valid voiced or spectral frames are available.
    """
    # Apply the same 5% whole-signal peak gate used by the reference pipeline.
    threshold = float(np.max(np.abs(samples))) * ENERGY_THRESHOLD_RATIO
    filtered = np.where(np.abs(samples) <= threshold, 0.0, samples)

    f0_frames = _frame_signal(
        filtered,
        TRACK_FRAME_LENGTH,
        F0_FRAME_STEP,
        apply_window=False,
        pad_end=False,
    )
    f0_values: list[float] = []
    fmaxi = sample_rate // int(MAX_F0_HZ)
    cepstrum_limit = TRACK_FRAME_LENGTH // 2
    for frame in f0_frames:
        frame = np.where(frame == 0.0, 1e-6, frame)
        spectrum = np.abs(np.fft.fft(frame))
        log_spectrum = np.log(np.maximum(spectrum, EPSILON))
        cepstrum = np.real(np.fft.ifft(log_spectrum))
        cepstrum = np.where(np.isfinite(cepstrum), cepstrum, 0.0)
        search = cepstrum[fmaxi:cepstrum_limit]
        if search.size == 0:
            continue
        peak_index = int(np.argmax(search))
        if peak_index == 0:
            continue
        period_samples = fmaxi + peak_index
        f0_hz = sample_rate / period_samples
        if MIN_F0_HZ <= f0_hz <= MAX_F0_HZ:
            f0_values.append(float(f0_hz))

    dominant_frames = _frame_signal(
        filtered,
        TRACK_FRAME_LENGTH,
        DOMINANT_FRAME_STEP,
        apply_window=True,
        pad_end=False,
    )
    frequencies_hz = np.fft.rfftfreq(
        TRACK_FRAME_LENGTH,
        d=1.0 / sample_rate,
    )[:-1]
    dominant_band = frequencies_hz <= DOMINANT_MAX_HZ
    dominant_values: list[float] = []
    for frame in dominant_frames:
        spectrum = np.abs(np.fft.rfft(frame))[:-1]
        band_spectrum = spectrum[dominant_band]
        if band_spectrum.size == 0 or float(np.max(band_spectrum)) <= EPSILON:
            continue
        peak_index = int(np.argmax(band_spectrum))
        dominant_values.append(float(frequencies_hz[dominant_band][peak_index]))

    if not f0_values or not dominant_values:
        raise ValueError(
            "Not enough valid speech frames to extract F0 or dominant frequency"
        )
    return np.asarray(f0_values), np.asarray(dominant_values)


def _dominant_features(dominant_hz: np.ndarray) -> dict[str, float]:
    """Aggregate dominant-frequency statistics and calculate modulation.

    Args:
        dominant_hz: One-dimensional array of dominant-frequency values in Hz.

    Returns:
        Dominant-frequency statistics in kHz and the dimensionless modulation
        index expected by the model.
    """
    dominant_khz = dominant_hz / 1000.0
    minimum = float(np.min(dominant_khz))
    maximum = float(np.max(dominant_khz))
    frequency_range = maximum - minimum
    if dominant_khz.size < 2 or frequency_range <= EPSILON:
        modulation_index = 0.0
    else:
        modulation_index = float(
            np.mean(np.abs(np.diff(dominant_khz))) / frequency_range
        )
    return {
        "meandom": float(np.mean(dominant_khz)),
        "mindom": minimum,
        "maxdom": maximum,
        "dfrange": frequency_range,
        "modindx": modulation_index,
    }


def _fundamental_features(f0_hz: np.ndarray) -> dict[str, float]:
    """Aggregate F0 statistics and express the result in kHz.

    Args:
        f0_hz: One-dimensional array of valid fundamental-frequency values in Hz.

    Returns:
        Mean, minimum, and maximum F0 values in kHz using the model feature names.
    """
    f0_khz = f0_hz / 1000.0
    return {
        "meanfun": float(np.mean(f0_khz)),
        "minfun": float(np.min(f0_khz)),
        "maxfun": float(np.max(f0_khz)),
    }


def extract_features_from_pcm_sequence(
    pcm_sequence: Sequence[bytes], sample_rate: int = SAMPLE_RATE
) -> dict[str, float]:
    """Convert a PCM sequence into the 20 features required by XGBoost.

    Args:
        pcm_sequence: Consecutive mono int16 little-endian PCM chunks from one speaker.
        sample_rate: PCM sample rate in Hz; defaults to 44,100 Hz but is not
            restricted to that value.

    Returns:
        A dictionary of 20 finite floating-point features keyed by fixed feature names, ready for
        XGBoost inference.

    Raises:
        TypeError: The input type is invalid.
        ValueError: The audio is empty, too short, silent, malformed, or has no valid F0.
    """
    samples = _decode_pcm_sequence(pcm_sequence, sample_rate)
    features = _spectrum_features(samples, sample_rate)
    f0_hz, dominant_hz = _track_f0_and_dominant(samples, sample_rate)
    features.update(_fundamental_features(f0_hz))
    features.update(_dominant_features(dominant_hz))
    if tuple(features) != tuple(FEATURE_NAMES):
        features = {name: float(features[name]) for name in FEATURE_NAMES}
    values = np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Feature extraction produced NaN or infinite values")
    return features
