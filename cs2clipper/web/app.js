'use strict';
/* CS2 自动剪辑 —— 前端逻辑 (无框架, 无构建链) */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const STAGES = [
  ['analyze_music', '分析音乐'],
  ['analyze_demo', '解析 demo'],
  ['plan', '编排剪辑'],
  ['save_artifacts', '保存产物'],
  ['render', '渲染画面'],
  ['compose', '拼接铺音乐'],
];

let META = null;
let CURRENT_JOB = null;
let EVENT_SOURCE = null;
let LAST_SEQ = 0;
let STATE = { edl: null, demo: null, music: null };

/* ================= 通用 ================= */
function fmtSize(n) {
  if (!n && n !== 0) return '-';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(0) + ' KB';
  if (n < 1024 * 1024 * 1024) return (n / 1048576).toFixed(1) + ' MB';
  return (n / 1073741824).toFixed(2) + ' GB';
}
function fmtTime(t) { return t ? String(t).replace('T', ' ').replace('+00:00', '') : '-'; }
function fmtSec(s) { if (!s && s !== 0) return '-'; const m = Math.floor(s / 60); return m ? `${m}分${(s % 60).toFixed(0)}秒` : `${s.toFixed(1)}秒`; }
function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function toast(msg, bad) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = 'toast' + (bad ? ' bad' : '');
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, bad ? 5200 : 2600);
}
async function api(path, opts) {
  const r = await fetch(path, Object.assign({ headers: { 'Content-Type': 'application/json' } }, opts || {}));
  let body = null;
  try { body = await r.json(); } catch (e) { body = { ok: false, error: `HTTP ${r.status}` }; }
  if (!r.ok || body.ok === false) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}

/* ================= 标签页 ================= */
$$('.tab').forEach((btn) => btn.addEventListener('click', () => {
  $$('.tab').forEach((b) => b.classList.toggle('active', b === btn));
  $$('.panel').forEach((p) => p.classList.toggle('active', p.id === 'tab-' + btn.dataset.tab));
  if (btn.dataset.tab === 'prefs') loadPrefs();
  if (btn.dataset.tab === 'runs') loadRuns();
  if (btn.dataset.tab === 'about') loadHealth();
}));

/* ================= 初始化 ================= */
async function boot() {
  try {
    META = await api('/api/meta');
  } catch (e) {
    $('#health').textContent = '服务不可用';
    $('#health').className = 'status-chip bad';
    toast('无法连接后端: ' + e.message, true);
    return;
  }
  $('#health').textContent = META.busy ? '有任务在跑' : '就绪';
  $('#health').className = 'status-chip ok';

  $('#aspect').innerHTML = '<option value="">跟随偏好</option>' +
    META.aspects.map((a) => `<option value="${a}">${a}</option>`).join('');

  // 参数框显示"当前偏好值"作为占位, 空值 = 用偏好
  const p = META.prefs;
  $('#fps').placeholder = '偏好 ' + (p.fps ?? META.defaults.fps);
  $('#max_clips').placeholder = '偏好 ' + (p.max_clips ?? META.defaults.max_clips);
  $('#max_cards').placeholder = '偏好 ' + (p.max_cards ?? META.defaults.max_cards);

  const dm = p.default_demo || META.default_demo;
  if (dm && META.default_demo_exists) $('#music-path').focus();
  await loadDemos();
  const last = localStorage.getItem('lastMusic');
  if (last) $('#music-path').value = last;
  $('#about-info').textContent = JSON.stringify(META, null, 2);
  buildStages();
  await refreshJobs();
}

function buildStages() {
  $('#stages').innerHTML = STAGES.map(([k, label]) =>
    `<li data-stage="${k}">${label}</li>`).join('');
}

/* ================= 素材选择 ================= */
async function loadDemos() {
  try {
    const r = await api('/api/demos');
    const sel = $('#demo-path');
    const cur = sel.value;
    sel.innerHTML = '<option value="">（用偏好里的默认 demo）</option>' +
      r.files.map((f) => `<option value="${esc(f.path)}">${esc(f.name)} · ${fmtSize(f.size)}</option>`).join('');
    if (cur) sel.value = cur;
    const prefDemo = (META && META.prefs.default_demo) || '';
    if (!sel.value && prefDemo) {
      const hit = r.files.find((f) => f.path === prefDemo);
      if (hit) sel.value = hit.path;
    }
    $('#demo-hint').textContent = r.files.length
      ? `找到 ${r.files.length} 个 demo（扫描工作区与常见下载目录）`
      : '没扫到 demo 文件：把 .dem 放进工作区，或用 --set default_demo <路径> 设定';
  } catch (e) { toast('扫描 demo 失败: ' + e.message, true); }
}
$('#btn-demo-refresh').addEventListener('click', loadDemos);

/* --- 音乐浏览弹窗 --- */
let BROWSE_DIR = null;

/** 统一的显隐入口.
 *
 * 只设 `el.hidden` 是不够可靠的坑: 浏览器 UA 样式里的 `[hidden]{display:none}`
 * 会被作者样式里的 display 覆盖 (`.modal{display:grid}`), 结果属性设上了、
 * 元素却照样显示 —— 表现为"关闭按钮点了没反应"。CSS 里已用
 * `[hidden]{display:none!important}` 兜住, 这里再同步一个 class,
 * 让"到底关没关"在 DOM 上看得见, 也便于静态检查。
 */
function show(el, on) {
  if (!el) return;
  el.hidden = !on;
  el.classList.toggle('is-hidden', !on);
}

async function openBrowser(dir) {
  show($('#modal-music'), true);
  await browse(dir || null);
}
async function browse(dir) {
  try {
    const q = dir ? '?dir=' + encodeURIComponent(dir) : '';
    const r = await api('/api/music' + q);
    BROWSE_DIR = r.dir;
    $('#browse-dir').value = r.dir;
    $('#browse-dirs').innerHTML = (r.parent ? `<div data-dir="${esc(r.parent)}">⬆ ..</div>` : '') +
      (r.dirs.length ? r.dirs.map((d) => `<div data-dir="${esc(d.path)}">📁 ${esc(d.name)}</div>`).join('')
        : '<div class="empty small">没有子目录</div>');
    $('#browse-files').innerHTML = r.files.length
      ? r.files.map((f) => `<div data-file="${esc(f.path)}">🎵 ${esc(f.name)} <span class="small">${fmtSize(f.size)}</span></div>`).join('')
      : '<div class="empty small">这个目录里没有音频文件</div>';
    $$('#browse-dirs [data-dir]').forEach((el) => el.addEventListener('click', () => browse(el.dataset.dir)));
    $$('#browse-files [data-file]').forEach((el) => el.addEventListener('click', () => {
      $('#music-path').value = el.dataset.file;
      localStorage.setItem('lastMusic', el.dataset.file);
      closeBrowser();
      validateInputs();
    }));
  } catch (e) { toast('目录读取失败: ' + e.message, true); }
}
function closeBrowser() { show($('#modal-music'), false); }

$('#btn-browse-music').addEventListener('click', () => openBrowser(BROWSE_DIR));
$('#modal-close').addEventListener('click', closeBrowser);
$('#modal-music').addEventListener('click', (e) => {
  // 点遮罩空白处也关 (点弹窗本体内部不关)
  if (e.target.id === 'modal-music') closeBrowser();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !$('#modal-music').hidden) closeBrowser();
});
$('#browse-go').addEventListener('click', () => browse($('#browse-dir').value.trim()));
$('#browse-up').addEventListener('click', () => { if (BROWSE_DIR) browse(BROWSE_DIR.replace(/[\\/][^\\/]+$/, '')); });
$('#music-path').addEventListener('change', () => {
  localStorage.setItem('lastMusic', $('#music-path').value.trim());
  validateInputs();
});

async function validateInputs() {
  const music = $('#music-path').value.trim();
  if (!music) return;
  try {
    const r = await api('/api/validate', { method: 'POST', body: JSON.stringify({ music_path: music, demo_path: $('#demo-path').value }) });
    toast(`素材就绪：${music.split(/[\\/]/).pop()} + ${String(r.demo).split(/[\\/]/).pop()}`);
  } catch (e) { toast('素材有问题: ' + e.message, true); }
}

/* ================= 出片 ================= */
function collectParams() {
  const val = (id) => { const v = $(id).value; return v === '' ? null : v; };
  const num = (id) => { const v = val(id); return v === null ? null : Number(v); };
  const llm = val('#use_llm');
  return {
    music_path: $('#music-path').value.trim(),
    demo_path: $('#demo-path').value,
    aspect: val('#aspect'),
    fps: num('#fps'),
    max_clips: num('#max_clips'),
    max_cards: num('#max_cards'),
    pacing: val('#pacing'),
    use_llm: llm === null ? null : llm === 'true',
    out_dir: $('#out_dir').value.trim() || null,
    use_prefs: $('#use_prefs').checked,
    record: $('#record').checked,
  };
}

$('#btn-run').addEventListener('click', async () => {
  const params = collectParams();
  if (!params.music_path) { toast('先选一首音乐', true); return; }
  $('#btn-run').disabled = true;
  $('#progress-card').hidden = false;
  show($('#result-card'), false);
  show($('#timeline-card'), false);
  $('#log').innerHTML = '';
  $('#bar').style.width = '0%';
  $('#percent').textContent = '0%';
  LAST_SEQ = 0;
  try {
    const r = await api('/api/jobs', { method: 'POST', body: JSON.stringify(params) });
    CURRENT_JOB = r.job;
    toast('任务已创建，开始分析…');
    attachStream(r.job.id);
    refreshJobs();
  } catch (e) {
    toast('无法启动: ' + e.message, true);
    $('#btn-run').disabled = false;
  }
});

$('#btn-cancel').addEventListener('click', async () => {
  if (!CURRENT_JOB) return;
  try { await api(`/api/jobs/${CURRENT_JOB.id}/cancel`, { method: 'POST' }); toast('已请求停止'); }
  catch (e) { toast('停止失败: ' + e.message, true); }
});

function logLine(text, cls) {
  const el = document.createElement('div');
  if (cls) el.className = cls;
  el.textContent = text;
  const box = $('#log');
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
  while (box.childElementCount > 500) box.removeChild(box.firstChild);
}

function attachStream(jobId) {
  if (EVENT_SOURCE) { EVENT_SOURCE.close(); EVENT_SOURCE = null; }
  const es = new EventSource(`/api/jobs/${jobId}/events?since=${LAST_SEQ}`);
  EVENT_SOURCE = es;
  es.onmessage = (ev) => {
    let d;
    try { d = JSON.parse(ev.data); } catch (e) { return; }
    if (d.kind === '__close__') { es.close(); EVENT_SOURCE = null; finishJob(jobId); return; }
    handleEvent(d);
  };
  es.onerror = () => { /* 浏览器会自动重连, 服务端按 since 重放 */ };
}

function handleEvent(d) {
  if (d.seq) LAST_SEQ = Math.max(LAST_SEQ, d.seq);
  // 服务端已保证百分比单调不减; 前端再取一次 max 只是防御性写法
  if (d.affects_percent && d.percent) {
    const cur = parseFloat($('#percent').textContent) || 0;
    if (d.percent >= cur) {
      $('#bar').style.width = d.percent + '%';
      $('#percent').textContent = d.percent.toFixed(0) + '%';
    }
  }
  if (d.stage_label) $('#stage-label').textContent = d.stage_label;
  if (d.stage) {
    const idx = STAGES.findIndex(([k]) => k === d.stage);
    $$('#stages li').forEach((li, i) => {
      li.classList.toggle('done', d.kind === 'stage_done' ? i <= idx : i < idx);
      li.classList.toggle('active', d.kind !== 'stage_done' && i === idx);
    });
  }
  const cls = d.kind === 'error' ? 'err' : (d.kind === 'done' ? 'ok' : (d.kind === 'stage_done' ? 'hl' : ''));
  if (d.message) logLine(`${d.stage_label ? '[' + d.stage_label + '] ' : ''}${d.message}`, cls);
  if (d.kind === 'done') { renderResult(d.detail || {}); }
  if (d.kind === 'error') { toast('出片失败: ' + d.message, true); $('#btn-run').disabled = false; }
}

async function finishJob(jobId) {
  try {
    const r = await api('/api/jobs/' + jobId);
    CURRENT_JOB = r.job;
    if (r.job.status === 'error') toast('出片失败: ' + r.job.error, true);
    else if (r.job.status === 'cancelled') toast('已停止（已完成的片段仍已出片）');
    else toast('出片完成');
    if (r.job.video_path && r.job.status !== 'error') {
      show($('#result-card'), true);
      await loadResult(r.job);
    }
  } catch (e) { /* 忽略 */ }
  $('#btn-run').disabled = false;
  refreshJobs();
}

function renderResult(detail) {
  const t = detail.timings || {};
  $('#result-meta').innerHTML = [
    detail.clips ? `<span>${detail.clips} 段</span>` : '',
    detail.duration ? `<span>${detail.duration}s</span>` : '',
    Object.entries(t).map(([k, v]) => `<span>${k} ${v}s</span>`).join(''),
  ].join('');
  if (detail.warnings && detail.warnings.length) {
    $('#result-warnings').innerHTML = detail.warnings.map((w) => `<div class="warn">⚠ ${esc(w)}</div>`).join('');
  }
}

async function loadResult(job) {
  const video = job.video_path;
  $('#player').src = '/api/artifact?path=' + encodeURIComponent(video);
  $('#btn-download').href = '/api/download?path=' + encodeURIComponent(video);
  $('#btn-reveal').onclick = () => showOutputs(job.out_dir);
  await loadTimeline(job.out_dir);
}

/* ================= 时间轴 / 素材 ================= */
async function loadTimeline(outDir) {
  if (!outDir) return;
  try {
    STATE.edl = await api('/api/artifact?path=' + encodeURIComponent(outDir + '/edl.json'));
    STATE.demo = await api('/api/artifact?path=' + encodeURIComponent(outDir + '/demo_analysis.json'));
    STATE.music = await api('/api/artifact?path=' + encodeURIComponent(outDir + '/music_analysis.json'));
  } catch (e) { toast('读取产物 JSON 失败: ' + e.message, true); return; }
  show($('#timeline-card'), true);
  drawTimeline();
  drawSegments();
  drawCards();
}

function drawTimeline() {
  const clips = (STATE.edl && STATE.edl.clips) || [];
  const segs = (STATE.music && STATE.music.segments) || [];
  const total = clips.reduce((a, c) => a + (c.out_end - c.out_start), 0) || 1;
  const maxDur = Math.max(...clips.map((c) => c.out_end - c.out_start), 1);
  const box = $('#timeline');
  box.innerHTML = '';
  clips.forEach((c, i) => {
    const dur = c.out_end - c.out_start;
    const seg = segs[c.music_segment];
    const arou = seg ? seg.arousal : 0.5;
    const el = document.createElement('div');
    el.className = 'tl-clip';
    el.style.height = (34 + 62 * (dur / maxDur)).toFixed(0) + 'px';
    el.style.flexGrow = String(dur / total);
    el.style.background = `hsl(${28 + 14 * arou}, ${45 + 40 * arou}%, ${34 + 26 * arou}%)`;
    el.title = `#${i + 1} ${dur.toFixed(2)}s | ${c.highlight_id} | 段落 ${c.music_segment} ${seg ? seg.label : ''} | speed ${c.speed}`;
    el.innerHTML = `<b>${i + 1}</b>`;
    el.addEventListener('click', () => selectClip(i));
    box.appendChild(el);
  });
}

function selectClip(i) {
  $$('.tl-clip').forEach((el, k) => el.classList.toggle('sel', k === i));
  const clip = STATE.edl.clips[i];
  const id = clip.highlight_id;
  const cards = (STATE.demo && STATE.demo.highlights) || [];
  const ci = cards.findIndex((c) => c.id === id);
  $$('.mini').forEach((el, k) => el.classList.toggle('sel', k === ci));
  const target = $(`.mini[data-i="${ci}"]`);
  if (target) target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  const seg = (STATE.music.segments || [])[clip.music_segment];
  const sel = $(`.seg[data-i="${clip.music_segment}"]`);
  if (sel) sel.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  if (seg) logLine(`#${i + 1} ${clip.highlight_id} → 段落 ${clip.music_segment}「${seg.label}」${seg.emotion}`, 'hl');
}

function drawSegments() {
  const segs = (STATE.music && STATE.music.segments) || [];
  $('#segments').innerHTML = segs.map((s, i) => `
    <div class="seg" data-i="${i}">
      <div><b>${i}</b> ${s.start.toFixed(1)}–${s.end.toFixed(1)}s · <span class="pill">${esc(s.label)}</span>
        <span class="mono">${esc(s.emotion || '')}</span></div>
      <div class="bar2"><i style="width:${(s.arousal * 100).toFixed(0)}%"></i></div>
      <div class="hint">激烈度 ${s.arousal.toFixed(2)} · 亮度 ${s.brightness.toFixed(2)} · 段内 ${s.bpm_local} BPM</div>
    </div>`).join('') || '<div class="hint">没有段落数据</div>';
}

function drawCards() {
  const cards = (STATE.demo && STATE.demo.highlights) || [];
  const used = new Set(((STATE.edl && STATE.edl.clips) || []).map((c) => c.highlight_id));
  $('#cards').innerHTML = cards.map((c, i) => `
    <div class="mini" data-i="${i}">
      <div><b>${esc(c.player)}</b> · R${c.round_num} · ${c.duration}s ·
        <b style="color:var(--accent)">${c.score}</b>
        ${used.has(c.id) ? '<span class="pill">已采用</span>' : ''}</div>
      <div class="tags">${(c.tags || []).map((t) => `<span class="pill${t === 'clutch' ? ' clutch' : ''}">${esc(t)}</span>`).join('')}
        ${c.clutch_enemies ? `<span class="pill clutch">1v${c.clutch_enemies}</span>` : ''}</div>
      <div class="hint">${esc(c.places.join(' / ')) || '-'} · ${esc(c.id)}</div>
    </div>`).join('') || '<div class="hint">没有素材卡片</div>';
}

async function showOutputs(dir) {
  try {
    const r = await api('/api/outputs?dir=' + encodeURIComponent(dir));
    const lines = r.items.map((it) => `${it.is_dir ? '📁' : '📄'} ${it.name} ${it.is_dir ? '' : fmtSize(it.size)}`);
    toast(`${r.dir}（${r.clips} 个片段）`);
    logLine('产物目录 ' + r.dir, 'hl');
    lines.forEach((l) => logLine('  ' + l));
    $('#progress-card').hidden = false;
  } catch (e) { toast('打开产物目录失败: ' + e.message, true); }
}

/* ================= 偏好 / 画像 ================= */
let PREFS_ORIG = {};
let PROFILE_META = null;
function fillObsValues() {
  if (!PROFILE_META) return;
  const dim = $('#obs-dim').value;
  const vals = (PROFILE_META.dimension_values || {})[dim] || [];
  $('#obs-val').outerHTML = `<select id="obs-val">` +
    vals.map((v) => `<option value="${esc(v)}">${esc(v)} · ${esc((PROFILE_META.value_labels || {})[dim + '|' + v] || '')}</option>`).join('') +
    '</select>';
}
$('#obs-dim').addEventListener('change', fillObsValues);
async function loadPrefs() {
  try {
    const r = await api('/api/prefs');
    PREFS_ORIG = r.effective;
    const specs = r.specs;
    $('#prefs-form').innerHTML = Object.keys(specs).map((k) => {
      const s = specs[k];
      const v = r.effective[k];
      let input;
      if (s.choices) {
        input = `<select data-k="${k}">` + s.choices.map((c) =>
          `<option value="${esc(c)}"${c === v ? ' selected' : ''}>${esc(c)}</option>`).join('') + '</select>';
      } else if (s.type === 'bool') {
        input = `<select data-k="${k}"><option value="true"${v ? ' selected' : ''}>开</option>
                 <option value="false"${!v ? ' selected' : ''}>关</option></select>`;
      } else {
        input = `<input data-k="${k}" type="${s.type === 'int' || s.type === 'float' ? 'number' : 'text'}"
                 step="${s.type === 'float' ? '0.05' : '1'}" value="${esc(v)}">`;
      }
      const isDefault = JSON.stringify(v) === JSON.stringify(s.default);
      return `<div class="pref" data-k="${k}">
        <div class="k"><b>${esc(k)}</b><span>${isDefault ? '默认值' : '已自定义'}</span></div>
        ${input}<div class="d">${esc(s.desc)}（默认 ${esc(String(s.default))}）</div></div>`;
    }).join('');
    $$('#prefs-form [data-k]').forEach((el) => el.addEventListener('change', () => {
      const k = el.dataset.k;
      const changed = String(el.value) !== String(PREFS_ORIG[k]);
      el.closest('.pref').classList.toggle('changed', changed);
    }));
  } catch (e) { toast('读取偏好失败: ' + e.message, true); }
}
$('#btn-prefs-reload').addEventListener('click', loadPrefs);
$('#btn-prefs-save').addEventListener('click', async () => {
  const values = {};
  $$('#prefs-form [data-k]').forEach((el) => { values[el.dataset.k] = el.value; });
  try {
    const r = await api('/api/prefs/set', { method: 'POST', body: JSON.stringify({ values }) });
    $('#prefs-msg').textContent = `已写入 ${Object.keys(r.applied).length} 项偏好`;
    toast('偏好已保存');
    loadPrefs();
    if (META) { META.prefs = r.prefs; }
  } catch (e) { toast('保存失败: ' + e.message, true); $('#prefs-msg').textContent = e.message; }
});
$('#btn-prefs-reset').addEventListener('click', async () => {
  if (!confirm('清空全部偏好？之后会回到内置默认值（用户画像与历史记录不受影响）。')) return;
  try {
    const r = await api('/api/prefs/reset', { method: 'POST' });
    toast(`已清空 ${r.cleared} 项偏好`);
    loadPrefs();
  } catch (e) { toast('清空失败: ' + e.message, true); }
});

async function loadProfile() {
  try {
    const r = await api('/api/profile');
    const rows = r.profile || [];
    $('#profile-body').innerHTML = rows.length ? rows.map((row) => {
      const label = (r.value_labels[row.dimension + '|' + row.value]) || row.value;
      return `<div class="prof-row">
        <div class="prof-head"><b>${esc(r.dimensions[row.dimension] || row.dimension)}</b>
          <span class="pill">${esc(label)}</span>
          <span class="hint">置信 ${(row.confidence * 100).toFixed(0)}% · 证据 ${row.evidence} 条</span></div>
        <div class="conf"><i style="width:${(row.confidence * 100).toFixed(0)}%"></i></div>
        <div class="quote">${esc(row.quote || '（无原话）')}</div>
      </div>`;
    }).join('') : '<div class="hint">还没有画像：先在出片后跑几次，或在下方手工补证据。</div>';
    $('#obs-dim').innerHTML = Object.entries(r.dimensions).map(([k, v]) =>
      `<option value="${esc(k)}">${esc(v)}</option>`).join('');
    PROFILE_META = r;
    fillObsValues();
    const obs = (r.observations || []).slice(0, 25);
    $('#profile-body').insertAdjacentHTML('beforeend',
      `<h3>最近证据（共 ${(r.observations || []).length} 条）</h3>` +
      (obs.length ? obs.map((o) => `<div class="quote">#${o.id} ${esc(o.dimension)} = <b>${esc(o.value)}</b>
        <span class="hint">[${esc(o.source)}, w=${o.confidence}]</span> 「${esc((o.quote || '').slice(0, 46))}」</div>`).join('')
        : '<div class="hint">暂无</div>'));
  } catch (e) { toast('读取画像失败: ' + e.message, true); }
}
$('#btn-profile-reload').addEventListener('click', loadProfile);
$('#btn-profile-rebuild').addEventListener('click', async () => {
  try {
    const r = await api('/api/profile/rebuild', { method: 'POST' });
    toast(`已重建画像（${r.rows} 个维度）`);
    loadProfile();
  } catch (e) { toast('重建失败: ' + e.message, true); }
});
$('#btn-observe').addEventListener('click', async () => {
  const body = {
    dimension: $('#obs-dim').value,
    value: $('#obs-val').value.trim(),
    quote: $('#obs-quote').value.trim(),
  };
  if (!body.value) { toast('取值不能为空', true); return; }
  try {
    const r = await api('/api/observe', { method: 'POST', body: JSON.stringify(body) });
    toast(`证据已写入，画像现有 ${r.rows} 个维度`);
    $('#obs-val').value = ''; $('#obs-quote').value = '';
    loadProfile();
  } catch (e) { toast('写入失败: ' + e.message, true); }
});

/* ================= 历史 ================= */
async function loadRuns() {
  try {
    const r = await api('/api/runs?limit=50');
    const s = r.stats;
    $('#run-stats').innerHTML = `
      <div class="stat"><b>${s.runs}</b>次出片</div>
      <div class="stat"><b>${s.clips}</b>段镜头</div>
      <div class="stat"><b>${fmtSec(s.avg_elapsed_sec)}</b>平均耗时</div>
      <div class="stat"><b>${fmtSize(s.total_video_bytes)}</b>成片总量</div>
      <div class="stat"><b>${s.prefs}</b>项偏好</div>
      <div class="stat"><b>${s.observations}</b>条证据</div>
      <div class="stat"><b>${s.profile_dims}</b>维画像</div>
      <div class="stat"><b>${fmtSize(s.db_bytes)}</b>数据库</div>`;
    const tb = $('#runs-table tbody');
    tb.innerHTML = r.runs.map((run) => `
      <tr data-id="${run.id}">
        <td>${run.id}</td>
        <td>${esc(fmtTime(run.started_at))}</td>
        <td>${esc(run.map_name || '-')}</td>
        <td>${esc(run.aspect || '-')}@${run.fps || '-'}</td>
        <td>${esc(run.planner || '-')}${run.use_llm ? ' · LLM' : ''}</td>
        <td>${run.clips ?? '-'}</td>
        <td>${run.duration_sec ? run.duration_sec.toFixed(1) + 's' : '-'}</td>
        <td>${run.elapsed_sec ? run.elapsed_sec.toFixed(0) + 's' : '-'}</td>
        <td>${run.video_path ? '<span class="pill">有</span>' : '<span class="hint">无</span>'}</td>
        <td><button class="btn sm" data-act="detail">详情</button></td>
      </tr>`).join('') || '<tr><td colspan="10" class="hint">还没有运行记录</td></tr>';
    $$('#runs-table tbody tr').forEach((tr) => {
      tr.addEventListener('click', (e) => {
        const run = r.runs.find((x) => String(x.id) === tr.dataset.id);
        if (e.target.dataset.act === 'detail' || !e.target.dataset.act) showRun(run);
      });
    });
  } catch (e) { toast('读取历史失败: ' + e.message, true); }
}
$('#btn-runs-reload').addEventListener('click', loadRuns);
$('#btn-run-detail-close').addEventListener('click', () => { show($('#run-detail'), false); });

async function showRun(run) {
  if (!run) return;
  show($('#run-detail'), true);
  $('#run-detail-title').textContent = `运行 #${run.id} · ${run.map_name || ''} ${run.aspect || ''}`;
  let clips = [];
  try { clips = (await api(`/api/runs/${run.id}/clips`)).clips; } catch (e) { /* 忽略 */ }
  const vp = run.video_path || '';
  const inRoot = vp && !vp.includes('..');
  $('#run-detail-body').innerHTML = `
    <div class="row wrap" style="margin-bottom:10px">
      <span class="pill">${esc(run.aspect || '-')}</span>
      <span class="pill">${run.fps || '-'} fps</span>
      <span class="pill">${esc(run.planner || '-')}</span>
      <span class="pill">${run.clips ?? 0} 段</span>
      <span class="pill">${run.duration_sec ? run.duration_sec.toFixed(1) + 's' : '-'}</span>
      <span class="pill">耗时 ${run.elapsed_sec ? run.elapsed_sec.toFixed(0) + 's' : '-'}</span>
      ${vp && inRoot ? `<a class="btn sm" href="/api/download?path=${encodeURIComponent(vp)}" download>下载成片</a>` : ''}
    </div>
    <div class="hint">${esc(vp || '（这次运行没有留下成片文件）')}</div>
    ${clips.length ? `<div class="table-wrap" style="margin-top:10px"><table>
      <thead><tr><th>#</th><th>出点</th><th>时长</th><th>素材</th><th>玩家</th><th>回合</th>
        <th>评分</th><th>speed</th><th>标签</th></tr></thead><tbody>
      ${clips.map((c) => `<tr><td>${c.idx ?? c.index ?? ''}</td>
        <td>${c.out_start != null ? c.out_start.toFixed(2) : '-'}</td>
        <td>${c.out_end != null && c.out_start != null ? (c.out_end - c.out_start).toFixed(2) : '-'}</td>
        <td>${esc(c.highlight_id || '-')}</td><td>${esc(c.player || '-')}</td>
        <td>${c.round_num ?? '-'}</td><td>${c.score ?? '-'}</td>
        <td>${c.speed ?? '-'}</td>
        <td>${(c.tags || []).map((t) => `<span class="pill">${esc(t)}</span>`).join('')}</td></tr>`).join('')}
      </tbody></table></div>` : '<div class="hint">没有片段明细</div>'}`;
}

/* ================= 其它 ================= */
async function refreshJobs() {
  try {
    const r = await api('/api/jobs');
    const active = r.jobs.find((j) => j.status === 'running' || j.status === 'queued');
    const busy = !!active;
    $('#health').textContent = busy ? (active.status === 'queued' ? '排队中' : `进行中 ${active.percent.toFixed(0)}%`) : '就绪';
    $('#health').className = 'status-chip ' + (busy ? '' : 'ok');
    if (active && !EVENT_SOURCE) {
      CURRENT_JOB = active;
      $('#progress-card').hidden = false;
      attachStream(active.id);
    }
    if (active) $('#btn-cancel').disabled = false;
  } catch (e) { /* 忽略 */ }
}
setInterval(refreshJobs, 5000);

async function loadHealth() {
  try {
    const h = await api('/api/health');
    $('#about-info').textContent = JSON.stringify(h, null, 2);
  } catch (e) { /* 忽略 */ }
}

boot();
