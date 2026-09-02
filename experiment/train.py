"""使用 XGBoost 根据声学特征识别人声性别。

本脚本复现 Voice Gender 数据集公开分析中表现最高的 XGBoost 配置：200 棵
最大深度为 5 的树，并使用 0.1 的学习率、0.8 的样本采样率和特征采样率。
输入 CSV 必须包含固定的 20 个声学特征和 ``label`` 列；脚本不会从原始音频
提取特征。运行后会在输出目录保存 XGBoost JSON 模型、评估指标和特征重要性，
便于后续端侧转换和复现实验。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

VERSION = 'v1'

FEATURE_COLUMNS = (
    "meanfreq",
    "sd",
    "median",
    "Q25",
    "Q75",
    "IQR",
    "skew",
    "kurt",
    "sp.ent",
    "sfm",
    "mode",
    "centroid",
    "meanfun",
    "minfun",
    "maxfun",
    "meandom",
    "mindom",
    "maxdom",
    "dfrange",
    "modindx",
)

LABEL_MAPPING = {"male": 0, "female": 1}


def read_inputs() -> tuple[Path, Path, float]:
    """通过标准输入读取训练参数。

    Returns:
        数据集路径、输出目录和测试集比例。

    Raises:
        ValueError: 测试集比例无法转换为浮点数时抛出。
    """
    project_root = Path(__file__).resolve().parents[1]
    default_data_path = project_root / "experiment" / "data" / "voice.csv"
    default_output_dir = Path(__file__).resolve().with_suffix("") / "artifacts"

    data_input = input("Enter training CSV path (press Enter for default): ").strip()
    output_input = input("Enter output directory (press Enter for default): ").strip()
    test_input = input("Enter test set ratio (press Enter for 0.2): ").strip()

    data_path = Path(data_input) if data_input else default_data_path
    output_dir = Path(output_input) if output_input else default_output_dir
    test_size = float(test_input) if test_input else 0.2
    return data_path, output_dir, test_size


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """读取并校验 Voice Gender 特征表。

    Args:
        data_path: 包含声学特征和 ``label`` 列的 CSV 路径。

    Returns:
        特征数据框和编码为 0/1 的标签序列。

    Raises:
        FileNotFoundError: The dataset file does not exist.
        ValueError: Required columns are missing, labels are invalid, or numeric
            features contain non-finite values.
    """
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {data_path}")

    data = pd.read_csv(data_path)
    required_columns = set(FEATURE_COLUMNS) | {"label"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    features = data.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not features.notna().all().all():
        raise ValueError("Acoustic features contain null or non-numeric values")

    labels = data["label"].astype(str).str.strip().str.casefold()
    unknown_labels = sorted(set(labels) - set(LABEL_MAPPING))
    if unknown_labels:
        raise ValueError(f"Label contains unsupported values: {unknown_labels}")
    return features, labels.map(LABEL_MAPPING).astype("int8")


def build_model() -> XGBClassifier:
    """创建公开 Notebook 使用的 XGBoost 二分类器。

    Returns:
        配置固定、可复现实验结果的 XGBoost 分类器。
    """
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
        eval_metric="logloss",
    )


def train_and_evaluate(
    features: pd.DataFrame,
    labels: pd.Series,
    test_size: float,
) -> tuple[XGBClassifier, dict[str, Any]]:
    """分层切分数据、训练模型并生成测试集指标。

    Args:
        features: 按 ``FEATURE_COLUMNS`` 排列的数值特征。
        labels: 0 表示 male、1 表示 female 的标签序列。
        test_size: 测试集占全部样本的比例。

    Returns:
        训练完成的模型和可 JSON 序列化的评估结果。

    Raises:
        ValueError: The test set ratio is outside the valid range.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test).astype(int)
    probabilities = model.predict_proba(x_test)[:, 1]

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["male", "female"],
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "dataset_size": int(len(features)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "test_ratio": test_size,
        "random_state": 42,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "probability_range": [float(probabilities.min()), float(probabilities.max())],
        "feature_columns": list(FEATURE_COLUMNS),
        "label_mapping": LABEL_MAPPING,
        "model_parameters": model.get_params(),
    }
    return model, metrics


def save_artifacts(
    model: XGBClassifier, metrics: dict[str, Any], output_dir: Path
) -> None:
    """保存模型、评估指标和特征重要性。

    Args:
        model: 已训练的 XGBoost 模型。
        metrics: 测试集评估结果。
        output_dir: 输出目录，不存在时自动创建。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "voice_gender_xgboost.json")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    feature_importance = {
        feature: float(importance)
        for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_)
    }
    (output_dir / "feature_importance.json").write_text(
        json.dumps(feature_importance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    data_path, output_dir, test_size = read_inputs()
    features, labels = load_dataset(data_path)
    model, metrics = train_and_evaluate(features, labels, test_size)
    save_artifacts(model, metrics, output_dir)

    print(f"Test accuracy: {metrics['accuracy']:.6f}")
    print("Confusion matrix (rows=true, columns=predicted; order=male/female):")
    print(metrics["confusion_matrix"])
    print(f"Model saved to: {output_dir / 'voice_gender_xgboost.json'}")


if __name__ == "__main__":
    main()
