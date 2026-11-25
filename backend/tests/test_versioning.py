"""模型版本治理端到端测试（SQLite 替身，无需 MySQL/Redis）。

运行：/workspace/.venv/bin/python tests/test_versioning.py
覆盖：版本绑定、切换失败保留旧版、重试不改版本、结案重算、损坏不可复现。
"""
import os
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

ARTIFACT_DIR = tempfile.mkdtemp(prefix='berth-models-')
DB_FILE = Path(tempfile.mkdtemp(prefix='berth-db-')) / 'test.db'
os.environ['MODEL_ARTIFACT_DIR'] = ARTIFACT_DIR

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src import database as dbmod

engine = create_engine(f'sqlite:///{DB_FILE}', future=True)
Session = sessionmaker(engine, expire_on_commit=False, future=True)
dbmod.engine = engine
dbmod.Session = Session

from src import classifier, incidents, model_registry, worker
from src.models import (
    Base, DelayClass, DelayIncident, IncidentState, ModelState, ModelVersion,
)
from src.planning import confirm, reserve_tug, revise
from src.schemas import CloseInput, DispatchInput, IncidentInput, MarginDecision, Schedule
from src.security import password_hash
from src.models import User
from scripts.seed_demo import seed

Base.metadata.create_all(engine)
with Session.begin() as db:
    model_registry.ensure_baseline(db)
    db.add(User(username='supervisor', role='supervisor', password_hash=password_hash('x')))
    db.add(User(username='operator', role='operator', password_hash=password_hash('x')))

with Session.begin() as db:
    counts = seed()
assert counts['model_versions'] == 1

with Session.begin() as db:
    active = model_registry.active_version(db)
    assert active.code == 'v1' and active.state == ModelState.ACTIVE
    assert (Path(ARTIFACT_DIR) / 'model-v1.pkl').exists()

# 规则 3：登记时绑定版本；worker 按绑定版本算出原分
processed = worker.process_next()
assert processed
with Session() as db:
    case = db.get(DelayIncident, 'DL-413')
    assert case.state in (IncidentState.MARGIN_REVIEW, IncidentState.READY)
    assert case.model_version == 'v1'
    assert case.result['model_version'] == 'v1'
    original_score = case.result['score']

# 新版先留存为 staged，旧版 v1 继续接单
with Session.begin() as db:
    v2 = model_registry.register(db, 'supervisor', 'v2', seed=2026, note='换型试验')
    assert v2.state == ModelState.STAGED
    still = model_registry.active_version(db)
    assert still.code == 'v1'

# staged 期间登记的新案仍绑定 v1
with Session.begin() as db:
    incidents.report(db, 'operator', IncidentInput(
        code='DL-414', request_code='BR-411', radio_reference='VHF-12/0950-21',
        observed_at='2026-11-21T09:50', wind_kn=10, tug_wait_min=5, event_note='二次试验案件。'))
with Session() as db:
    assert db.get(DelayIncident, 'DL-414').model_version == 'v1'
assert worker.process_next()
with Session() as db:
    assert db.get(DelayIncident, 'DL-414').result['model_version'] == 'v1'

# 规则 2：校验通过后激活，旧版转留存
with Session.begin() as db:
    activated, failure = model_registry.activate(db, 'supervisor', 'v2')
    assert failure is None
with Session() as db:
    assert db.get(ModelVersion, 'v1').state == ModelState.RETIRED
    assert db.get(ModelVersion, 'v2').state == ModelState.ACTIVE

# 激活后新案绑定 v2（先把 BR-412 备妥并确认）
with Session.begin() as db:
    reserve_tug(db, 'operator', 'BR-412', 'TG23')
    confirm(db, 'supervisor', 'BR-412')
    incidents.report(db, 'operator', IncidentInput(
        code='DL-415', request_code='BR-412', radio_reference='VHF-12/1110-22',
        observed_at='2026-11-21T11:10', wind_kn=16, tug_wait_min=20, event_note='换型后首案。'))
with Session() as db:
    assert db.get(DelayIncident, 'DL-415').model_version == 'v2'
assert worker.process_next()
with Session() as db:
    case415 = db.get(DelayIncident, 'DL-415')
    assert case415.result['model_version'] == 'v2'

# 规则 4：重试不改换版本——删掉 v2 产物制造失败，重试后恢复产物仍按 v2
artifact_v2 = Path(ARTIFACT_DIR) / 'model-v2.pkl'
hidden = artifact_v2.read_bytes()
artifact_v2.unlink()
# 先把 BR-410 改期到 B07 的空闲时段并确认，作为产物缺失演练的载体
with Session.begin() as db:
    revise(db, 'operator', 'BR-410', Schedule(window_code='TW-072', eta='2026-11-21T10:20', maneuver_minutes=25))
    reserve_tug(db, 'operator', 'BR-410', 'TG18')
    confirm(db, 'supervisor', 'BR-410')
with Session.begin() as db:
    fresh = incidents.report(db, 'operator', IncidentInput(
        code='DL-416', request_code='BR-410', radio_reference='VHF-12/1120-23',
        observed_at='2026-11-21T10:40', wind_kn=12, tug_wait_min=8, event_note='产物缺失演练。'))
    assert fresh.model_version == 'v2'
assert worker.process_next()
with Session() as db:
    failed = db.get(DelayIncident, 'DL-416')
    assert failed.state == IncidentState.FAILED
    assert '未留存' in failed.error
with Session.begin() as db:
    incidents.retry(db, 'supervisor', 'DL-416')
with Session() as db:
    retried = db.get(DelayIncident, 'DL-416')
    assert retried.state == IncidentState.QUEUED and retried.model_version == 'v2'
artifact_v2.write_bytes(hidden)
assert worker.process_next()
with Session() as db:
    assert db.get(DelayIncident, 'DL-416').result['model_version'] == 'v2'

# 规则 2（反面）：激活校验失败必须保留旧版 v2 接单
with Session.begin() as db:
    model_registry.register(db, 'supervisor', 'v3', seed=3030, note='损坏演练版')
(Path(ARTIFACT_DIR) / 'model-v3.pkl').write_bytes(b'not-a-model')
activation_rejected = False
with Session.begin() as db:
    activated, failure = model_registry.activate(db, 'supervisor', 'v3')
    if failure is not None and failure.code == 'MODEL_DAMAGED':
        activation_rejected = True
assert activation_rejected
with Session() as db:
    assert db.get(ModelVersion, 'v2').state == ModelState.ACTIVE
    assert db.get(ModelVersion, 'v3').state == ModelState.DAMAGED

# 走完 DL-413 的签署、指令、结案，再按原版本重算
with Session.begin() as db:
    case = db.get(DelayIncident, 'DL-413')
    if case.state == IncidentState.MARGIN_REVIEW:
        incidents.review_margin(db, 'supervisor', 'DL-413',
                                MarginDecision(decision=DelayClass.PRIORITY, reason='二次通报确认优先处置。'))
    incidents.dispatch(db, 'supervisor', 'DL-413',
                       DispatchInput(action='standby', instruction='TG18 现场待命，引航席保持联络。'))
    incidents.close(db, 'supervisor', 'DL-413', CloseInput(
        all_fast_at='2026-11-21T10:15', outcome='全缆系妥，指令执行无误，交班归档。'))
with Session() as db:
    assert db.get(DelayIncident, 'DL-413').state == IncidentState.CLOSED

# 规则 1 + 5：复盘重算，原分与重算分对照，原判和指令不动
with Session.begin() as db:
    incidents.recompute_score(db, 'supervisor', 'DL-413')
with Session() as db:
    case = db.get(DelayIncident, 'DL-413')
    assert case.recompute['model_version'] == 'v1'
    assert case.recompute['matches_original'] is True
    assert abs(case.recompute['score'] - original_score) <= 1e-9
    assert case.result['score'] == original_score  # 原始结果未被改写
    assert case.decision == DelayClass.PRIORITY

# 规则 5：旧模型损坏/未留存 → 明确无法复现，绝不拿新版补算
(Path(ARTIFACT_DIR) / 'model-v1.pkl').unlink()
with Session.begin() as db:
    incidents.recompute_score(db, 'supervisor', 'DL-413')
with Session() as db:
    case = db.get(DelayIncident, 'DL-413')
    assert case.recompute is None and case.recompute_error
    assert 'v1' in case.recompute_error and '无法' in case.recompute_error
    assert case.result['model_version'] == 'v1'  # 原判仍在，没有被 v2 覆盖
    assert case.decision == DelayClass.PRIORITY

# 损坏产物直接装载也必须报 ModelUnavailable，而不是回退
try:
    classifier.load_artifact(Path(ARTIFACT_DIR), 'v3')
    raise AssertionError('损坏产物应当装载失败')
except classifier.ModelUnavailable:
    pass

print('ALL VERSION-GOVERNANCE CHECKS PASSED')
