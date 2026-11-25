"""模型版本登记、校验与切换。

切换规则：新版先以 staged 留存并通过校验，主管显式激活后才接单；
激活失败时旧 active 版本保持接单。任何旧案重算只认案件登记时绑定的版本。
"""

import logging
import os
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session as DatabaseSession

from . import classifier
from .models import ModelState, ModelVersion
from .planning import Rejected, audit, utc_now

log = logging.getLogger(__name__)
ARTIFACT_DIR = Path(os.environ.get('MODEL_ARTIFACT_DIR', '/var/berth/models'))
BASELINE_VERSION = 'v1'


def artifact_directory():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR


def list_versions(db: DatabaseSession):
    return list(db.scalars(select(ModelVersion).order_by(ModelVersion.created_at, ModelVersion.code)))


def active_version(db: DatabaseSession, *, lock=False):
    version = db.scalar(select(ModelVersion).where(ModelVersion.state == ModelState.ACTIVE)
                        .order_by(ModelVersion.activated_at.desc()).limit(1)
                        .with_for_update(skip_locked=lock))
    if version is None:
        raise Rejected('MODEL_NOT_READY', '没有可用的接单模型版本，请先完成模型登记与激活', 'model-versions')
    return version


def load_version(version_code: str):
    """按版本号装载产物；损坏/缺失统一抛出 ModelUnavailable，不回退新版。"""
    return classifier.load_artifact(artifact_directory(), version_code)


def register(db: DatabaseSession, actor: str, code: str, seed: int, note: str):
    if db.get(ModelVersion, code):
        raise Rejected('DUPLICATE_MODEL', f'模型版本编号 {code} 已存在', code)
    try:
        payload = classifier.save_artifact(artifact_directory(), code, seed=seed, note=note)
    except Exception as exc:
        log.exception('Model artifact build failed for %s', code)
        raise Rejected('MODEL_BUILD_FAILED', f'新版模型训练或留存失败：{exc}', code) from exc
    version = ModelVersion(
        code=code, state=ModelState.STAGED, seed=seed, parameters=payload['parameters'],
        metrics=payload['metrics'], sklearn_version=payload['sklearn'],
        artifact_path=str(classifier.artifact_path(artifact_directory(), code)),
        artifact_digest=payload['artifact_digest'], note=note, created_by=actor, created_at=utc_now(),
    )
    db.add(version)
    db.flush()
    _verify(db, version, actor, event='MODEL_STAGED')
    return version


def _check_artifact(version: ModelVersion):
    """只读校验留存产物，失败抛 ModelUnavailable。"""
    loaded = load_version(version.code)
    if loaded.artifact_digest != version.artifact_digest:
        raise classifier.ModelUnavailable('产物摘要与登记记录不一致')
    probe = dict(zip(classifier.FEATURES, (18.0, 25.0, 14.0, 12.0)))
    classifier.score(loaded, probe, classifier.snapshot_digest(probe))


def _verify(db: DatabaseSession, version: ModelVersion, actor: str, *, event):
    """用留存产物重新装载并跑通固定控制点，校验不过则抛错且不得接单。"""
    try:
        _check_artifact(version)
    except classifier.ModelUnavailable as exc:
        audit(db, actor, version.code, 'MODEL_DAMAGED', str(exc)[:400])
        raise Rejected('MODEL_DAMAGED', f'模型 {version.code} 校验未通过，保留旧版接单：{exc}', version.code) from exc
    audit(db, actor, version.code, event,
          f"平衡准确率 {version.metrics['balanced_accuracy']} · 支持向量 {version.metrics['support_vectors']}")


def activate(db: DatabaseSession, actor: str, code: str):
    """激活新版：任一校验失败都保留旧 active 版本继续接单。

    返回 (版本, 业务错误)。校验失败时在同一事务内把新版标记为 damaged 并提交，
    旧 active 版本保持接单；调用方在事务外抛出错误，避免整体回滚丢失损坏标记。
    """
    candidate = db.get(ModelVersion, code, with_for_update=True)
    if candidate is None:
        raise Rejected('NOT_FOUND', f'找不到模型版本 {code}', code, 404)
    if candidate.state == ModelState.ACTIVE:
        return candidate, None
    previous = active_version(db, lock=True)
    # 切换放进保存点：校验失败只回滚切换动作，损坏标记在同一事务内保留，旧版不解锁。
    try:
        with db.begin_nested():
            _check_artifact(candidate)
            candidate.state = ModelState.ACTIVE
            candidate.activated_at = utc_now()
            previous.state = ModelState.RETIRED
            audit(db, actor, code, 'MODEL_VERIFIED',
                  f"平衡准确率 {candidate.metrics['balanced_accuracy']} · 支持向量 {candidate.metrics['support_vectors']}")
            audit(db, actor, code, 'MODEL_ACTIVATED', f'{previous.code} 停止接单并转为留存；新案改用 {code}')
            audit(db, actor, previous.code, 'MODEL_RETIRED', f'被 {code} 替代，旧案仍按本版本重算')
    except classifier.ModelUnavailable as exc:
        candidate.state = ModelState.DAMAGED
        reason = f'模型 {code} 校验未通过，保留旧版接单：{exc}'
        audit(db, actor, code, 'MODEL_DAMAGED', str(exc)[:400])
        audit(db, actor, code, 'MODEL_ACTIVATION_ABORTED', f'{code} 校验未通过，保留 {previous.code} 接单')
        return candidate, Rejected('MODEL_DAMAGED', reason, code)
    return candidate, None


def ensure_baseline(db: DatabaseSession, actor: str = 'system'):
    """首次建库时登记并激活基线模型，保证总有版本接单。"""
    existing = db.get(ModelVersion, BASELINE_VERSION)
    if existing:
        return existing, False
    payload = classifier.save_artifact(artifact_directory(), BASELINE_VERSION,
                                       seed=classifier.SEED, note='初始基线模型（固定种子 1074）')
    version = ModelVersion(
        code=BASELINE_VERSION, state=ModelState.ACTIVE, seed=classifier.SEED,
        parameters=payload['parameters'], metrics=payload['metrics'],
        sklearn_version=payload['sklearn'],
        artifact_path=str(classifier.artifact_path(artifact_directory(), BASELINE_VERSION)),
        artifact_digest=payload['artifact_digest'], note=payload['note'],
        created_by=actor, created_at=utc_now(), activated_at=utc_now(),
    )
    db.add(version)
    audit(db, actor, BASELINE_VERSION, 'MODEL_ACTIVATED', '基线模型登记并激活，作为初始接单版本')
    return version, True
