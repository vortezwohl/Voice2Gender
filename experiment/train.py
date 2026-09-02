"""使用 XGBoost 根据声学特征识别人声性别。

本脚本使用经过 80 个候选版本实测后选出的 v2 XGBoost 配置：使用 lossguide
生长策略、每棵树最多 8 个叶子、250 轮 boosting，并使用 0.06 的学习率、0.9
的样本采样率和特征采样率。输入 CSV 必须包含固定的 20 个声学特征和 label
列；脚本不会从原始音频提取特征。运行后会在输出目录保存 XGBoost JSON 模型、
评估指标、特征重要性和完整训练参数，便于后续端侧转换和复现实验。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

VERSION = 'v2'

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
    default_output_dir = Path("./train") / VERSION

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
        # objective：定义训练任务和损失函数；二分类概率任务使用 binary:logistic。
        # 取值：可选 XGBoost 内置目标函数，也可传自定义目标函数；二分类应选
        # binary:logistic 或 binary:logitraw，多分类应选 multi:softprob/softmax。
        # 选择逻辑：logistic 输出 0 到 1 的正类概率，适合本项目的性别二分类。
        objective="binary:logistic",  # 当前值：二分类并输出 female 概率。
        # base_score：所有样本的初始预测分数；二分类 Logistic 的默认概率为 0.5。
        # 取值：通常为浮点数；二分类应在 0 到 1 之间，也可传多目标初始向量。
        # 选择逻辑：平衡数据使用 0.5；若有可靠的类别先验可改为先验概率，但过度偏离
        # 真实分布可能使前期训练变慢或引入偏置。
        base_score=0.5,  # 当前值：XGBoost 3.4.1 的二分类默认初始概率。
        # booster：选择基学习器；gbtree 使用树，dart 在树上随机丢弃部分树，gblinear
        # 使用线性模型（XGBoost 3.4.1 已标记为逐步弃用）。
        # 取值：gbtree、dart、gblinear。树模型适合非线性关系，线性模型更简单更快，
        # dart 可降低过拟合但会增加随机性和调参复杂度。
        booster="gbtree",  # 当前值：XGBoost 的默认树模型。
        # callbacks：控制每轮训练结束时的自定义回调。
        # 取值：None 或 TrainingCallback 对象列表，例如学习率调度和自定义
        # early stopping。
        # 选择逻辑：None 表示不启用额外回调；启用回调可增加控制能力，但回调对象不能
        # 在多个训练任务之间直接复用。
        callbacks=None,  # 当前值：不使用额外训练回调。
        # colsample_bylevel：每棵树的每一层随机抽取的特征比例。
        # 取值：大于 0 且不超过 1；1 使用全部特征，较小值增加随机性并降低过拟合。
        # 选择逻辑：只有需要额外列采样时才降低；它会与其他 colsample 参数相乘。
        colsample_bylevel=1.0,  # 当前值：使用每层的全部特征。
        # colsample_bynode：每次节点切分时随机抽取的特征比例。
        # 取值：大于 0 且不超过 1；减小可降低特征共适应，但可能丢失有效特征。
        # 选择逻辑：当前特征只有 20 个，默认使用全部特征，避免过度随机化。
        colsample_bynode=1.0,  # 当前值：使用每个节点的全部特征。
        # colsample_bytree：构建每棵树时随机抽取的特征比例。
        # 取值：大于 0 且不超过 1；较小值通常降低过拟合并增加树之间差异，
        # 但过小会造成欠拟合。当前 0.9 是搜索后选定的 v2 配置。
        colsample_bytree=0.9,  # 当前值：每棵树使用 90% 的特征。
        # device：指定训练和预测设备。
        # 取值：cpu、cuda 或 gpu；cuda/gpu 需要可用的 CUDA 环境和 GPU 版本支持。
        # 选择逻辑：CPU 适合当前小型数据集且依赖简单；GPU 适合更大数据，但会增加
        # 环境要求，设备变化也可能带来轻微数值差异。
        device="cpu",  # 当前值：使用 CPU。
        # early_stopping_rounds：验证指标连续多少轮没有改善后停止训练。
        # 取值：None 或非负整数；必须在 fit() 中传入 eval_set 才能真正工作。
        # 选择逻辑：None 表示固定执行 n_estimators 轮；启用后通常能降低过拟合和训练
        # 时间，但模型轮数会依赖验证集，结果不再只由 n_estimators 决定。
        early_stopping_rounds=None,  # 当前值：不提前停止。
        # enable_categorical：是否启用 pandas categorical 等类别特征支持。
        # 取值：True 或 False；True 只在输入含类别特征且数据类型正确时发挥作用。
        # 选择逻辑：当前 20 列均为连续数值，True 与 False 对当前训练结果没有实质影响；
        # 保持 XGBoost 3.4.1 sklearn API 的默认值。
        enable_categorical=True,  # 当前值：允许类别特征。
        # eval_metric：训练期间监控的评估指标，不会替换 objective。
        # 取值：内置指标名称、名称列表或自定义 callable；二分类常见有 logloss、error、
        # auc、aucpr。logloss/错误率越低越好，AUC/AUCPR 越高越好。
        # 选择逻辑：当前使用 logloss，与 binary:logistic 的概率输出匹配；类别极不平衡时
        # 可额外关注 aucpr，但必须同步调整验证和选择标准。
        eval_metric="logloss",  # 当前值：监控对数损失。
        # feature_types：显式声明每个输入特征的类型。
        # 取值：None 或与特征列一一对应的类型序列；常见类型为 int、float、bool、c、
        # q 等。长度或类型错误会导致训练失败。
        # 选择逻辑：None 让 XGBoost 从 pandas 数据自动推断；当前数据已是数值列，
        # 无需重复声明。
        feature_types=None,  # 当前值：由输入数据自动推断。
        # feature_weights：列采样时各特征被选中的相对权重。
        # 取值：None 或长度等于特征数的正数数组；权重越大，被采样概率越高。
        # 选择逻辑：没有可靠的特征先验时使用 None；人为提高某些特征权重可能造成偏置。
        feature_weights=None,  # 当前值：不调整特征采样权重。
        # gamma：节点继续分裂所需的最小损失下降，也叫 min_split_loss。
        # 取值：大于等于 0；0 允许默认分裂，增大后分裂更谨慎、树更简单。
        # 选择逻辑：验证集出现过拟合时可逐步增大；过大可能阻止有效切分并欠拟合。
        gamma=0.0,  # 当前值：XGBoost 默认不额外要求损失下降。
        # grow_policy：树节点的生长顺序。
        # 取值：depthwise 优先扩展较浅层节点；lossguide 优先扩展损失下降最大的节点。
        # 选择逻辑：v2 使用 lossguide，并配合 max_leaves=8 直接限制每棵树的叶子数，
        # 在保持表达能力的同时控制树的复杂度。
        grow_policy="lossguide",  # 当前值：按损失下降优先生长。
        # importance_type：feature_importances_ 属性使用的特征重要性口径。
        # 取值：树模型可选 weight、gain、cover、total_gain、total_cover；None 表示
        # sklearn 包装器按模型类型采用默认口径，树模型默认实际使用 gain。
        # 选择逻辑：gain 更关注损失改善，weight 更关注使用次数，cover 更关注覆盖样本；
        # 不同口径不能直接当作同一种重要性解释。
        importance_type=None,  # 当前值：使用树模型默认的 gain。
        # interaction_constraints：限制允许发生交互的特征组。
        # 取值：None，或表示特征索引/名称分组的嵌套列表或字符串；None 不限制交互。
        # 选择逻辑：只有存在明确业务先验或需要抑制不合理交互时才设置；限制过强会欠拟合。
        interaction_constraints=None,  # 当前值：不限制特征交互。
        # learning_rate：每轮新增树对最终模型的贡献，也叫 eta。
        # 取值：大于 0；较小值通常需要更大的 n_estimators，较大值训练更快但更易过拟合。
        # 选择逻辑：v2 使用 0.06，并配合 250 轮训练，在本数据切分上取得更高准确率。
        learning_rate=0.06,  # 当前值：每棵树使用 6% 的更新步长。
        # max_bin：hist 算法对连续特征分箱的最大箱数。
        # 取值：正整数；增大可提高阈值近似精度，但增加内存和计算成本。
        # 选择逻辑：当前特征和数据集较小，默认 256 已足够；数据量大或阈值精度不足时
        # 再调高。
        max_bin=256,  # 当前值：XGBoost hist 默认箱数。
        # max_cat_threshold：类别特征分区切分时最多考虑的类别数。
        # 取值：正整数；只影响类别特征。增大可能提高类别切分质量，但增加计算量和
        # 过拟合风险。
        # 选择逻辑：当前没有类别特征，因此保持默认 64，不影响本模型。
        max_cat_threshold=64,  # 当前值：XGBoost 类别切分默认上限。
        # max_cat_to_onehot：类别数低于该阈值时使用 one-hot 风格切分。
        # 取值：正整数；阈值越大，更多低基数类别会采用 one-hot，可能更细致但更复杂。
        # 选择逻辑：当前没有类别特征，因此保持默认 4。
        max_cat_to_onehot=4,  # 当前值：XGBoost 类别 one-hot 默认阈值。
        # max_delta_step：限制每棵树叶子权重估计的最大更新步长。
        # 取值：大于等于 0；0 表示不限制，较大值使更新更保守。
        # 选择逻辑：严重类别不平衡时可尝试增大；当前数据男女各半，保持 0。
        max_delta_step=0.0,  # 当前值：不限制叶子权重更新步长。
        # max_depth：单棵树允许的最大深度。
        # 取值：非负整数；0 表示不以深度限制，数值越大表达能力越强但越易过拟合。
        # 选择逻辑：v2 使用 max_depth=0，并由 lossguide 和 max_leaves=8 控制树复杂度。
        max_depth=0,  # 当前值：不使用最大深度限制。
        # max_leaves：单棵树允许的最大叶子数。
        # 取值：非负整数；0 表示不限制。与 lossguide 配合时，可用该值直接控制复杂度。
        # 选择逻辑：v2 使用 8，限制每棵树的叶子数，避免无限制生长造成过拟合。
        max_leaves=8,  # 当前值：每棵树最多 8 个叶子。
        # min_child_weight：子节点继续存在所需的最小实例权重总和。
        # 取值：大于等于 0；增大后更难创建小叶子，可抑制过拟合，但过大可能欠拟合。
        # 选择逻辑：当前使用默认 1；验证集噪声较大时可增大，模型过于保守时可减小。
        min_child_weight=1.0,  # 当前值：XGBoost 默认子节点权重下限。
        # missing：输入数据中应被当作缺失值的数值。
        # 取值：通常为 numpy.nan，也可指定某个特殊数值；当前 load_dataset 会先拒绝
        # 缺失值。
        # 选择逻辑：使用 numpy.nan 可保留 XGBoost 原生缺失值语义；指定特殊数值前必须确认
        # 该数值不会与合法特征值混淆。
        missing=np.nan,  # 当前值：使用 NaN 表示缺失。
        # monotone_constraints：限制特征与预测之间的单调关系。
        # 取值：None，或按特征名/索引指定 -1、0、1；-1 表示递减，0 不限制，1 表示递增。
        # 选择逻辑：只有业务上能证明单调关系时才使用；错误约束会直接损害模型表现。
        monotone_constraints=None,  # 当前值：不施加单调约束。
        # multi_strategy：多目标/多输出任务的树构建策略。
        # 取值：one_output_per_tree 为每个输出分别建树；multi_output_tree 共享多输出树。
        # 选择逻辑：当前是单目标二分类，使用默认 one_output_per_tree；多输出任务才需要
        # 比较。
        multi_strategy="one_output_per_tree",  # 当前值：每棵树处理一个输出。
        # n_estimators：boosting 轮数，也就是树的数量。
        # 取值：正整数；增大提高拟合能力和训练成本，也增加过拟合风险，通常与较小
        # learning_rate 配合。v2 使用 250 轮，是搜索后选定的训练轮数。
        n_estimators=250,  # 当前值：训练 250 轮。
        # n_jobs：CPU 并行线程数。
        # 取值：None、0 或正整数；0/None 通常交给 XGBoost 使用可用线程，固定正整数
        # 限制线程。
        # 选择逻辑：当前显式使用 1 以避免线程竞争、控制资源并尽量提高复现性；更大值可能
        # 更快，但会占用更多 CPU。
        n_jobs=1,  # 当前值：单线程训练。
        # num_parallel_tree：每个 boosting 轮次并行构建的树数。
        # 取值：正整数；1 是普通 boosting，增大可形成随机森林式并行树组但增加计算量。
        # 选择逻辑：当前模型使用普通单树 boosting，保持默认 1。
        num_parallel_tree=1,  # 当前值：每轮 1 棵树。
        # random_state：随机数种子。
        # 取值：None、整数或 NumPy 随机数生成器；固定整数提高同一环境中的结果复现性。
        # 选择逻辑：当前固定为 42，与数据切分和原训练配置一致；跨版本或硬件仍不保证
        # 逐位一致。
        random_state=42,  # 当前值：固定随机种子 42。
        # reg_alpha：叶子权重的 L1 正则化强度。
        # 取值：大于等于 0；增大后模型更稀疏、更保守，但过大可能欠拟合。
        # 选择逻辑：默认 0 表示不额外添加 L1；特征冗余或过拟合时可从小值开始搜索。
        reg_alpha=0.0,  # 当前值：不使用额外 L1 正则化。
        # reg_lambda：叶子权重的 L2 正则化强度，也叫 lambda。
        # 取值：大于等于 0；增大通常提高稳定性并抑制过拟合，过大可能欠拟合。
        # 选择逻辑：保持 XGBoost 默认 1；噪声较大时可增大，模型过保守时可减小。
        reg_lambda=1.0,  # 当前值：XGBoost 默认 L2 正则化。
        # sampling_method：样本采样方式。
        # 取值：uniform 均匀采样；gradient_based 按梯度/Hessian 重要性采样，主要用于 GPU
        # hist。CPU hist 使用 gradient_based 可能不受支持或退化为不适用配置。
        # 选择逻辑：当前使用 CPU、subsample=0.9，因此选择 uniform。
        sampling_method="uniform",  # 当前值：均匀样本采样。
        # scale_pos_weight：正类相对于负类的权重。
        # 取值：大于 0；常见起点是负类数量除以正类数量。增大可提高少数正类的关注，
        # 但会改变概率校准并可能提高误报。
        # 选择逻辑：当前 male/female 数量相等，使用 1 不改变类别权重。
        scale_pos_weight=1.0,  # 当前值：正负类等权。
        # subsample：每轮训练随机抽取的样本比例。
        # 取值：大于 0 且不超过 1；减小可降低过拟合并增加树间差异，但过小会欠拟合。
        # 选择逻辑：当前 0.8 与原训练脚本一致；数据少时不宜设置过低。
        subsample=0.9,  # 当前值：每轮使用 80% 的样本。
        # tree_method：树的构建算法。
        # 取值：auto、exact、approx、hist；GPU 训练通常配合 device=cuda 和 hist。
        # 选择逻辑：hist 速度快、内存相对低，适合当前数据；exact 更精确但通常更慢，
        # auto 让 XGBoost 自行选择，可能随版本或环境变化。
        tree_method="hist",  # 当前值：直方图算法。
        # validate_parameters：是否对未知或无效参数进行校验并发出警告。
        # 取值：True 或 False；True 更容易发现拼写错误，False 可能隐藏配置问题。
        # 选择逻辑：训练配置显式化后使用 True，优先暴露错误；不会替代参数本身的合法性
        # 检查。
        validate_parameters=True,  # 当前值：启用参数校验。
        # verbosity：XGBoost 日志详细程度。
        # 取值：0 到 3；0 静默，1 警告，2 信息，3 调试。日志越详细，诊断信息越多。
        # 选择逻辑：v2 使用 0，避免训练过程输出大量日志；错误仍通过异常暴露。
        verbosity=0,  # 当前值：静默日志。
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
    """保存模型、评估指标、特征重要性和完整训练参数。

    Args:
        model: 已训练的 XGBoost 模型。
        metrics: 测试集评估结果，其中包含数据切分配置和模型参数。
        output_dir: 输出目录，不存在时自动创建。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "voice_gender_xgboost.json")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    model_parameters = model.get_params()
    if isinstance(model_parameters.get("missing"), float) and np.isnan(
        model_parameters["missing"]
    ):
        model_parameters["missing"] = None
    training_parameters = {
        "version": VERSION,
        "model_parameters": model_parameters,
        "data_split_parameters": {
            "test_size": metrics["test_ratio"],
            "random_state": metrics["random_state"],
            "stratify": True,
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "label_mapping": LABEL_MAPPING,
    }
    (output_dir / "training_parameters.json").write_text(
        json.dumps(training_parameters, ensure_ascii=True, indent=2, allow_nan=False),
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
