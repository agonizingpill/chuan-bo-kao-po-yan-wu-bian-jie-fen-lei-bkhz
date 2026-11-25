"""版本化的延误边界分类器。

模型以固定训练方案在受控输入下拟合，序列化为带摘要的产物文件。
旧案件始终用登记时绑定的模型版本重算，缺失或损坏的旧产物不得用新版补算。
"""

import hashlib
import json
import pickle
from dataclasses import dataclass

import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 1074
BAND = 0.30
FEATURES = ('eta_offset_min', 'tide_reserve_min', 'wind_kn', 'tug_wait_min')
PARAMETERS = {'kernel': 'rbf', 'C': 1.7, 'gamma': 'scale', 'random_state': SEED}
ARTIFACT_VERSION = 1


class ModelUnavailable(Exception):
    """旧模型产物缺失或损坏，无法按登记版本复现分数。"""


class ModelMismatch(Exception):
    """重算结果与原始分数不一致，原模型无法忠实复现。"""


def snapshot_digest(features):
    return hashlib.sha256(json.dumps(features, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def history(rng, pairs):
    # Independent mirrored records around a borderline operating point.
    center = np.array([18, 25, 14, 12])
    offset = rng.uniform(-1, 1, (pairs, 4)) * [18, 24, 14, 12]
    noise = rng.normal(0.0, 2.8, pairs)
    score = offset @ [1.0, -0.62, 0.75, 0.85] + 3.2 * np.tanh(offset[:, 0] / 10) + noise
    priority = (score > 0).astype(int)
    return np.vstack([center + offset, center - offset]), np.concatenate([priority, 1 - priority])


def train(seed=SEED, parameters=None):
    """按固定方案拟合一个新版本；不同种子产生可区分的模型代际。"""
    parameters = dict(PARAMETERS if parameters is None else parameters)
    parameters.setdefault('random_state', seed)
    rng = np.random.default_rng(seed)
    train_x, train_y = history(rng, 360)
    validation_x, validation_y = history(rng, 120)
    model = make_pipeline(StandardScaler(), SVC(**parameters))
    model.fit(train_x, train_y)
    metrics = {
        'train_rows': len(train_y),
        'validation_rows': len(validation_y),
        'balanced_accuracy': round(float(balanced_accuracy_score(validation_y, model.predict(validation_x))), 4),
        'validation_margin_rate': round(float(np.mean(np.abs(model.decision_function(validation_x)) <= BAND)), 4),
        'support_vectors': int(model[-1].n_support_.sum()),
    }
    return model, metrics


def build_artifact(version, seed=SEED, parameters=None, note=''):
    model, metrics = train(seed, parameters)
    payload = {
        'artifact_version': ARTIFACT_VERSION,
        'model_version': version,
        'seed': seed,
        'band': BAND,
        'features': list(FEATURES),
        'parameters': dict(PARAMETERS if parameters is None else parameters),
        'metrics': metrics,
        'sklearn': sklearn.__version__,
        'note': note,
        'model': model,
    }
    blob = pickle.dumps(payload)
    payload['artifact_digest'] = hashlib.sha256(blob).hexdigest()
    return payload, blob


def artifact_path(directory, version):
    return directory / f'model-{version}.pkl'


def save_artifact(directory, version, seed=SEED, parameters=None, note=''):
    payload, blob = build_artifact(version, seed, parameters, note)
    path = artifact_path(directory, version)
    path.write_bytes(blob)
    return payload


@dataclass
class LoadedModel:
    version: str
    seed: int
    band: float
    parameters: dict
    metrics: dict
    sklearn: str
    note: str
    artifact_digest: str
    model: object


def load_artifact(directory, version):
    """装载并校验旧产物；损坏或留存缺失时显式报无法复现，绝不回退到新版。"""
    path = artifact_path(directory, version)
    if not path.exists():
        raise ModelUnavailable(f'模型版本 {version} 的产物未留存，无法按登记版本复现分数')
    try:
        payload = pickle.loads(path.read_bytes())
    except Exception as exc:  # 损坏的旧模型一律视为不可复现
        raise ModelUnavailable(f'模型版本 {version} 的产物已损坏：{exc}') from exc
    required = {'artifact_version', 'model_version', 'seed', 'band', 'features', 'parameters',
                'metrics', 'sklearn', 'model'}
    if not isinstance(payload, dict) or not required <= payload.keys():
        raise ModelUnavailable(f'模型版本 {version} 的产物内容不完整')
    if payload['model_version'] != version or payload['artifact_version'] != ARTIFACT_VERSION:
        raise ModelUnavailable(f'模型版本 {version} 的产物标识不一致')
    if tuple(payload['features']) != FEATURES:
        raise ModelUnavailable(f'模型版本 {version} 的输入字段与当前冻结口径不一致')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return LoadedModel(
        version=version, seed=payload['seed'], band=float(payload['band']),
        parameters=payload['parameters'], metrics=payload['metrics'], sklearn=payload['sklearn'],
        note=payload.get('note', ''), artifact_digest=digest, model=payload['model'],
    )


def score(loaded, features, expected_digest):
    if set(features) != set(FEATURES) or snapshot_digest(features) != expected_digest:
        raise ValueError('冻结输入不完整或摘要不一致')
    row = np.array([[features[key] for key in FEATURES]], dtype=float)
    if not np.isfinite(row).all():
        raise ValueError('冻结输入包含不可计算的数值')
    value = float(loaded.model.decision_function(row)[0])
    return {
        'score': value, 'band': loaded.band, 'needs_review': abs(value) <= loaded.band,
        'prediction': 'priority' if value > 0 else 'routine',
        'model_version': loaded.version, 'seed': loaded.seed, 'input_digest': expected_digest,
        'parameters': loaded.parameters, 'metrics': loaded.metrics, 'sklearn': loaded.sklearn,
        'artifact_digest': loaded.artifact_digest,
    }
