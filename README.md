<div align="center">

# Voice2Gender

**A tiny, local-first voice gender classifier for raw PCM audio.**

*Extract 20 acoustic features and run a bundled XGBoost model with one Python call.*

<p>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="https://xgboost.readthedocs.io/"><img src="https://img.shields.io/badge/XGBoost-3.4%2B-189AB4?style=flat-square" alt="XGBoost 3.4+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square" alt="MIT License" /></a>
  <a href="https://github.com/vortezwohl/Voice2Gender/stargazers"><img src="https://img.shields.io/github/stars/vortezwohl/Voice2Gender?style=flat-square&label=Stars" alt="GitHub stars" /></a>
</p>

<p>
  <a href="#quick-start">Quick start</a> &middot;
  <a href="#installation">Installation</a> &middot;
  <a href="#api">API</a> &middot;
  <a href="#how-it-works">How it works</a> &middot;
  <a href="#training-and-reproduction">Training</a> &middot;
  <a href="#limitations-and-responsible-use">Responsible use</a>
</p>

</div>

---

## Why Voice2Gender?

Voice2Gender is a deliberately small inference package for applications that need a fast, offline estimate from a speaker's acoustic signal. There is no web service, account, or model download step at runtime: the XGBoost booster is bundled in the package and inference runs on the CPU.

The public surface is intentionally simple:

```python
from voice2gender import predict

result = predict(pcm_chunks, sample_rate=44_100)
```

You get class probabilities, a compact confidence score, and (optionally) the extracted feature vector for inspection.

> [!WARNING]
> This project predicts the dataset's **binary `male`/`female` labels from vocal acoustics**. It does not identify a person, determine gender identity, or replace consent-based user research. Read [Limitations and responsible use](#limitations-and-responsible-use) before deploying it.

## Highlights

| Capability | Details |
| --- | --- |
| Local inference | The model is bundled in `voice2gender._model`; no network request is needed after installation. |
| Small API | `voice2gender.predict(...)` accepts an ordered sequence of PCM byte chunks and returns a JSON-friendly dictionary. |
| Reproducible features | The preprocessor mirrors the Voice Gender feature schema with 20 finite acoustic features. |
| Defensive validation | Empty, malformed, silent, too-short, wrong-rate, or non-finite inputs fail loudly with `TypeError`/`ValueError`. |
| Inspectable output | Pass `include_features=True` to include all model inputs in the result. |
| Re-trainable experiment | `experiment/train.py` rebuilds the XGBoost model from the feature CSV and writes metrics and feature-importance artifacts. |

## Installation

Voice2Gender supports Python 3.10 and newer. Install the published package directly into your application environment. [`uv`](https://docs.astral.sh/uv/) is recommended for fast, reproducible dependency management:

```bash
uv add -U voice2gender
```

If you prefer pip:

```bash
pip install -U voice2gender
```

The user installation contains the inference package and bundled model. You do not need to clone this repository or install the development extra to call `voice2gender.predict`.

> [!TIP]
> Starting a new project? Run `uv add -U voice2gender`. It records the dependency in `pyproject.toml` and keeps the environment in sync for you.

## Quick start

### Predict from a WAV file

The model expects **mono, signed little-endian int16 PCM at exactly 44,100 Hz**. This example uses only Python's standard library to read a compatible WAV file and forwards it in chunks:

```python
from pathlib import Path
import wave

from voice2gender import predict


def read_pcm_chunks(path: str | Path, chunk_frames: int = 8_192) -> tuple[list[bytes], int]:
    """Read a mono int16 WAV file as ordered PCM chunks."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("WAV must be mono, 16-bit PCM")
        sample_rate = wav.getframerate()
        chunks: list[bytes] = []
        while data := wav.readframes(chunk_frames):
            chunks.append(data)
    return chunks, sample_rate


pcm_chunks, sample_rate = read_pcm_chunks("sample.wav")
result = predict(pcm_chunks, sample_rate=sample_rate, include_features=True)
print(result)
```

Example result shape:

```python
{
    "male_probability": 0.99,
    "female_probability": 0.01,
    "confidence": 0.98,
    "features": {"meanfreq": 0.18, "meanfun": 0.12, "...": "..."},
}
```

`confidence` is `abs(female_probability - 0.5) * 2`, so values near `1` indicate a stronger model preference and values near `0` indicate an uncertain split. Treat probabilities as model scores, not human attributes.

### Microphone demo

`demo/live_demo.py` shows how to stream microphone bytes with `sounddevice`. Configure its `audio_stream_pc` call to `samplerate=44_100` before passing chunks to `predict`; its historical default of 16 kHz is not accepted by the bundled model.

```bash
uv run --extra dev python demo/live_demo.py
```

Microphone backends are platform-specific. If PortAudio cannot open the selected device, check OS audio permissions and your `sounddevice` installation.

## API

### `voice2gender.predict`

```python
predict(
    pcm_sequence: Sequence[bytes],
    sample_rate: int = 44_100,
    include_features: bool = False,
) -> dict[str, Any | float]
```

- `pcm_sequence`: non-empty, ordered mono PCM chunks. Each chunk must be `bytes` containing little-endian signed 16-bit samples.
- `sample_rate`: must be exactly `44_100` Hz.
- `include_features`: when `True`, adds the 20 extracted features under the `features` key.

Possible failures include malformed byte lengths, silence, fewer than one analysis frame (2,048 samples), and audio without valid voiced/spectral frames. Catch `TypeError` and `ValueError` at application boundaries when user audio is untrusted.

## How it works

```text
int16 PCM chunks
        |
        v
validation + normalization
        |
        v
spectral statistics (12) + F0 statistics (3) + dominant-frequency statistics (5)
        |
        v
bundled XGBoost binary classifier
        |
        v
male_probability / female_probability / confidence
```

The feature extractor follows the 20-column schema used by the Kaggle Voice Gender dataset: spectral distribution statistics, fundamental-frequency (`F0`) summaries, and dominant-frequency summaries. Frequencies are represented in kHz in the model feature vector, matching the source CSV.

## Model and benchmark

The bundled model is an XGBoost `gbtree` classifier trained with the repository's `v2` configuration (250 estimators, learning rate 0.06, histogram tree method, fixed random seed 42). The checked-in holdout report was produced with a stratified 80/20 split:

| Metric | Holdout result |
| --- | ---: |
| Samples | 634 (317 male / 317 female) |
| Accuracy | **97.95%** |
| Male precision / recall | 98.72% / 97.16% |
| Female precision / recall | 97.20% / 98.74% |
| Confusion matrix | `[[308, 9], [4, 313]]` |

These numbers describe one split of this dataset and are not a guarantee of real-world accuracy across microphones, languages, ages, accents, noise conditions, or speakers unseen during collection.

## Training and reproduction

The repository includes the feature table at `experiment/data/voice.csv` and the training script at `experiment/train.py`. The script expects the 20 acoustic columns plus a `label` column (`male` or `female`); it does **not** extract features from raw audio.

Run the interactive trainer with the default dataset and output directory:

```bash
uv run --extra dev python experiment/train.py
```

Press Enter at each prompt to accept the defaults. Artifacts are written to `experiment/train/v2/` (or the output directory you provide):

- `voice_gender_xgboost.json` — serialized XGBoost booster;
- `metrics.json` — split configuration, accuracy, report, and confusion matrix;
- `feature_importance.json` — gain-based feature importances;
- `training_parameters.json` — model and data-split parameters.

### Dataset

Training data comes from the [Voice Gender dataset on Kaggle](https://www.kaggle.com/datasets/primaryobjects/voicegender), originally published by Primary Objects. Review Kaggle's current dataset terms and attribution requirements before redistributing the data or a model trained from it. The repository copy is used for reproducible experiments; it is not a claim that the dataset represents every speaker or demographic group.

## Development

To work on the repository, run experiments, lint the code, or try the microphone demo:

```bash
git clone https://github.com/vortezwohl/Voice2Gender.git
cd Voice2Gender
uv sync --extra dev
```

The `dev` extra provides the optional training, linting, and microphone dependencies.

## Repository layout

```text
voice2gender/
|-- __init__.py       # public package entry point
|-- _predict.py       # feature-to-probability inference
|-- _preprocess.py    # PCM decoding and acoustic feature extraction
`-- _model.py         # bundled XGBoost booster and feature schema
demo/
`-- live_demo.py      # optional microphone streaming example
experiment/
|-- data/voice.csv    # Kaggle-derived feature table
|-- train.py          # reproducible training script
`-- train/v2/         # model, metrics, and importance artifacts
```

## Limitations and responsible use

- The classifier is binary because the source labels are binary; it cannot infer gender identity or non-binary identities from voice.
- Acoustic features are affected by recording hardware, room noise, codec artifacts, language, accent, age, health, and deliberate voice modification.
- A high confidence score means the model is far from its decision threshold, not that the prediction is objectively true.
- Do not use this model for hiring, access control, healthcare, law enforcement, identity verification, or any decision with material impact on a person.
- Obtain consent before recording or processing speech, minimize retention, and avoid storing raw audio unless there is a clear, lawful reason.
- Evaluate on your target population and microphone pipeline before relying on the output; report uncertainty and provide a human fallback.

## Contributing

Small, evidence-backed improvements are welcome. Before opening a pull request:

1. Keep the public input contract and feature order compatible, or document a deliberate breaking change.
2. Add a focused test or reproducible example for behavior changes.
3. Run the available checks:

   ```bash
   uv run --extra dev ruff check .
   ```

4. Explain dataset, benchmark, and responsible-use implications when changing the model or feature extractor.

## License

The source code is released under the [MIT License](LICENSE). The Kaggle dataset remains subject to its own terms; see the [dataset page](https://www.kaggle.com/datasets/primaryobjects/voicegender) for attribution and usage conditions.

## Citation

If Voice2Gender is useful in a paper, benchmark, or product evaluation, cite the repository:

```bibtex
@software{Wu_Voice2Gender_2026,
  author = {Wu, Zihao},
  title = {{Voice2Gender}},
  url = {https://github.com/vortezwohl/Voice2Gender},
  version = {0.1.0},
  year = {2026}
}
```
