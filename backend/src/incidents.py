import logging

from redis.exceptions import RedisError
from sqlalchemy import or_, select
from sqlalchemy.orm import Session as DatabaseSession

from . import classifier, model_registry
from .classifier import snapshot_digest
from .database import WAKE_QUEUE, queue
from .models import Action, BerthRequest, DelayClass, DelayIncident, DispatchReview, IncidentState, RequestState
from .planning import Rejected, audit, find, require_window, tide_clearance, utc_now
from .schemas import CloseInput, DispatchInput, IncidentInput, MarginDecision

log = logging.getLogger(__name__)


def notify(code):
    try:
        queue.lpush(WAKE_QUEUE, code)
    except RedisError:
        log.exception('Redis wake-up failed for %s; worker scans durable queue', code)


def report(db: DatabaseSession, actor: str, data: IncidentInput):
    request = find(db, BerthRequest, data.request_code, lock=True)
    if request.state != RequestState.CONFIRMED:
        raise Rejected('CONFIRMATION_REQUIRED', '请先核验潮窗并确认靠泊', request.code)
    previous = db.scalar(select(DelayIncident.code).where(or_(
        DelayIncident.code == data.code,
        DelayIncident.request_code == request.code,
        DelayIncident.radio_reference == data.radio_reference,
    )).limit(1))
    if previous:
        raise Rejected('DUPLICATE_INCIDENT', f'关联申请、案件编号或无线电记录已登记在 {previous}', data.code)
    offset = (data.observed_at - request.eta).total_seconds() / 60
    if not 0 < offset <= 360:
        raise Rejected('OBSERVATION_TIME', '延误观测必须晚于计划引航时刻，且在六小时内', data.code)
    window = require_window(db, request.window_code, request.berth_code, request.code)
    reserve, _ = tide_clearance(request, window)
    features = {
        'eta_offset_min': round(offset, 3), 'tide_reserve_min': round(reserve, 3),
        'wind_kn': data.wind_kn, 'tug_wait_min': data.tug_wait_min,
    }
    # 排队案件按登记时刻的接单版本算；该绑定随后续重试保持不变。
    version = model_registry.active_version(db)
    incident = DelayIncident(code=data.code, request_code=request.code, radio_reference=data.radio_reference,
                             observed_at=data.observed_at, event_note=data.event_note, features=features,
                             input_digest=snapshot_digest(features), model_version=version.code,
                             state=IncidentState.QUEUED)
    db.add(incident)
    audit(db, actor, incident.code, 'INCIDENT_QUEUED',
          f'{request.code} · {incident.radio_reference} · 绑定模型 {version.code}')
    return incident


def retry(db: DatabaseSession, actor: str, code: str):
    incident = find(db, DelayIncident, code, lock=True)
    if incident.state != IncidentState.FAILED:
        raise Rejected('RETRY_STATE', '只有计算失败的案件可以重试', code)
    incident.state = IncidentState.QUEUED
    incident.error = None
    audit(db, actor, code, 'INCIDENT_REQUEUED',
          f'复用输入摘要 {incident.input_digest} · 仍按登记版本 {incident.model_version}，重试不改换版本')
    return incident


def review_margin(db: DatabaseSession, actor: str, code: str, data: MarginDecision):
    incident = find(db, DelayIncident, code, lock=True)
    if incident.state != IncidentState.MARGIN_REVIEW:
        raise Rejected('NOT_IN_MARGIN_REVIEW', '案件当前未等待间隔带复核', code)
    incident.decision = data.decision
    incident.decided_by = actor
    incident.decision_reason = data.reason
    incident.decided_at = utc_now()
    incident.state = IncidentState.READY
    audit(db, actor, code, 'MARGIN_REVIEWED', f'{data.decision} · {data.reason}')
    return incident


def dispatch(db: DatabaseSession, actor: str, code: str, data: DispatchInput):
    incident = find(db, DelayIncident, code, lock=True)
    if incident.state == IncidentState.MARGIN_REVIEW:
        raise Rejected('MARGIN_REVIEW_REQUIRED', '间隔带案件必须由主管签署人工分类', code)
    if incident.state != IncidentState.READY or not incident.decision:
        raise Rejected('DISPATCH_STATE', '处置仅适用于已完成分类且尚未发布动作的案件', code)
    if incident.decision == DelayClass.PRIORITY and data.action == Action.MONITOR:
        raise Rejected('PRIORITY_ACTION_REQUIRED', '优先处置须发布待命或改期指令', code)
    record = DispatchReview(incident_code=code, issued_by=actor, issued_at=utc_now(), **data.model_dump())
    incident.state = IncidentState.ACTIONED
    db.add(record)
    audit(db, actor, code, 'ACTION_ISSUED', f'{data.action} · {data.instruction}')
    return record


def close(db: DatabaseSession, actor: str, code: str, data: CloseInput):
    incident = find(db, DelayIncident, code, lock=True)
    if incident.state != IncidentState.ACTIONED:
        raise Rejected('CLOSE_STATE', '请先发布调度动作再结案', code)
    record = find(db, DispatchReview, code)
    request = find(db, BerthRequest, incident.request_code, lock=True)
    if data.all_fast_at < incident.observed_at:
        raise Rejected('ALL_FAST_TIME', '全缆系妥时刻不得早于延误观测', code)
    record.all_fast_at = data.all_fast_at
    record.outcome = data.outcome
    record.closed_at = utc_now()
    record.closed_by = actor
    incident.state = IncidentState.CLOSED
    request.state = RequestState.COMPLETED
    audit(db, actor, code, 'REVIEW_CLOSED', f'{data.all_fast_at:%m-%d %H:%M} · {data.outcome}')
    return record


def recompute_score(db: DatabaseSession, actor: str, code: str):
    """结案复盘：用案件登记时绑定的同一模型版本重算冻结输入。

    只追加核对记录，不改写原始结果、人工签署与调度指令；
    旧模型损坏或未留存时明确标注无法复现，绝不以当前新版补算。
    """
    incident = find(db, DelayIncident, code, lock=True)
    if incident.state != IncidentState.CLOSED:
        raise Rejected('RECOMPUTE_STATE', '只有已结案案件可在复盘页核对重算分数', code)
    if not incident.result:
        raise Rejected('ORIGINAL_SCORE_MISSING', '原模型分数缺失，无法建立原分与重算分对照', code)
    try:
        loaded = model_registry.load_version(incident.model_version)
        check = classifier.score(loaded, incident.features, incident.input_digest)
    except classifier.ModelUnavailable as exc:
        incident.recompute = None
        incident.recompute_error = str(exc)[:180]
        incident.recompute_at = utc_now()
        audit(db, actor, code, 'RECOMPUTE_UNAVAILABLE',
              f"登记版本 {incident.model_version} 无法复现：{incident.recompute_error}")
        return incident
    original = float(incident.result['score'])
    check['matches_original'] = abs(check['score'] - original) <= 1e-9
    check['original_score'] = original
    incident.recompute = check
    incident.recompute_error = None
    incident.recompute_at = utc_now()
    if not check['matches_original']:
        audit(db, actor, code, 'RECOMPUTE_MISMATCH',
              f"原分 {original:+.5f} 与重算 {check['score']:+.5f} 不一致；保留原判与指令")
    else:
        audit(db, actor, code, 'RECOMPUTE_MATCHED',
              f"版本 {incident.model_version} 重算 {check['score']:+.5f} 与原分一致，原判与指令保留")
    return incident
