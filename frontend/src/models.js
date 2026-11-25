import {modelStates, store, supervisor} from './store.js';
import {blank, esc, field, form, pill, portDate, title} from './ui.js';

function stageForm() {
  if (!supervisor()) return '';
  return `<details class="paper-fold"><summary>训练并留存新版模型</summary>${form('stageModel', '', `
    <div class="form-grid">${field('模型版本号', 'code', 'text', '', 'placeholder="v2" pattern="[A-Za-z0-9-]{2,32}"')}
    ${field('训练种子', 'seed', 'number', 2026, 'min="1" max="10000000"')}</div>
    ${field('版本说明', 'note', 'text', '', 'placeholder="例如 11 月班次换型，调整合成样本种子" minlength="2" maxlength="300"')}
    <p class="muted">新版先留存为「待校验」，校验通过并显式激活后才接新案；切换失败时旧版继续接单。</p>`, '留存并校验新版')}</details>`;
}

function versionCard(version) {
  const canActivate = supervisor() && version.state === 'staged';
  return `<article class="model-card ${esc(version.state)}">
    <div class="section-head"><h3 class="mono">${esc(version.code)}</h3>${pill(version.state, modelStates)}</div>
    <dl class="measurements">
      <div><dt>平衡准确率</dt><dd>${(version.metrics.balanced_accuracy * 100).toFixed(1)}%</dd></div>
      <div><dt>复核带占比</dt><dd>${(version.metrics.validation_margin_rate * 100).toFixed(1)}%</dd></div>
      <div><dt>支持向量</dt><dd>${esc(version.metrics.support_vectors)}</dd></div>
      <div><dt>训练种子</dt><dd>${esc(version.seed)}</dd></div>
      <div><dt>scikit-learn</dt><dd>${esc(version.sklearn_version)}</dd></div>
      <div><dt>登记 / 激活</dt><dd>${esc(portDate(version.created_at))}${version.activated_at ? `<br>${esc(portDate(version.activated_at))}` : ''}</dd></div>
    </dl>
    <p class="muted">${esc(version.note || '—')}</p>
    <p class="digest">SHA-256 ${esc(version.artifact_digest)}</p>
    ${canActivate ? form('activateModel', version.code, '<p class="muted">激活后仅影响新登记案件；排队和已结案件仍按各自绑定版本计算与重算。</p>', '校验并激活接单') : ''}
    ${version.state === 'damaged' ? '<p class="warning">产物损坏，旧案无法按此版本复现，不会用于补算。</p>' : ''}
  </article>`;
}

export function models() {
  const versions = store.data.model_versions || [];
  const active = versions.find(v => v.state === 'active');
  return `${title('模型版本', '新版校验通过后才切换；旧案始终按登记时的版本重算，损坏即标注无法复现。',
    active ? `<span class="quiet">当前接单 <strong class="mono">${esc(active.code)}</strong></span>` : '')}
  <section class="panel">
    <div class="section-head"><h2>版本登记册</h2><span>${versions.length} 个版本</span></div>
    ${versions.length ? `<div class="model-grid">${versions.map(versionCard).join('')}</div>` : blank('尚无模型版本', '系统启动时会自动登记基线模型。')}
    ${stageForm()}
  </section>`;
}
