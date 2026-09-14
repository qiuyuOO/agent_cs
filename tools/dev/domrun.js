'use strict';
/* 在无头 DOM 里真跑一遍 app.js, 定位"关闭按钮点了没反应"。
 *
 * 为什么必须真跑: index.html 用的是普通 <script> (没有 defer)。
 * 普通脚本一旦顶层抛出未捕获异常, 就从那一行起**整体中断**, 后面所有
 * addEventListener 都不会绑定 —— 症状正是"某个按钮怎么点都没反应",
 * 而页面看起来一切正常。静态检查选择器/接口是抓不到这种问题的。
 *
 * 用法: node tools/dev/domrun.js
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WEB = path.join(__dirname, '..', '..', 'cs2clipper', 'web');
const html = fs.readFileSync(path.join(WEB, 'index.html'), 'utf8');
const js = fs.readFileSync(path.join(WEB, 'app.js'), 'utf8');

/* ---------------- 极简 DOM ---------------- */
class El {
  constructor(tag, attrs = {}) {
    this.tagName = (tag || 'div').toUpperCase();
    this.attrs = attrs;
    this.children = [];
    this.parentNode = null;
    this._classes = new Set((attrs.class || '').split(/\s+/).filter(Boolean));
    this._listeners = Object.create(null);
    this._value = attrs.value !== undefined ? attrs.value : '';
    this._checked = 'checked' in attrs;
    this._hidden = 'hidden' in attrs;
    this._html = '';
    this._text = '';
    this.src = attrs.src || '';
    this.id = attrs.id || '';
  }
  get className() { return [...this._classes].join(' '); }
  set className(v) { this._classes = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get classList() {
    const self = this;
    return {
      add: (...c) => c.forEach((x) => self._classes.add(x)),
      remove: (...c) => c.forEach((x) => self._classes.delete(x)),
      contains: (c) => self._classes.has(c),
      toggle: (c, force) => {
        const on = force === undefined ? !self._classes.has(c) : !!force;
        if (on) self._classes.add(c); else self._classes.delete(c);
        return on;
      },
    };
  }
  get hidden() { return this._hidden; }
  set hidden(v) {
    this._hidden = !!v;
    if (v) this.attrs.hidden = ''; else delete this.attrs.hidden;
  }
  get value() { return this._value; }
  set value(v) { this._value = v === undefined || v === null ? '' : String(v); }
  get checked() { return this._checked; }
  set checked(v) { this._checked = !!v; }
  get dataset() {
    const out = {};
    for (const [k, v] of Object.entries(this.attrs)) {
      if (k.startsWith('data-')) out[k.slice(5).replace(/-([a-z])/g, (m, c) => c.toUpperCase())] = v;
    }
    return out;
  }
  get innerHTML() { return this._html; }
  set innerHTML(v) {
    this._html = String(v);
    this.children = parseFragment(this._html, this);
  }
  get textContent() { return this._text; }
  set textContent(v) { this._text = v === null || v === undefined ? '' : String(v); this.children = []; }
  get firstChild() { return this.children[0] || null; }
  get childElementCount() { return this.children.length; }
  addEventListener(kind, fn) { (this._listeners[kind] = this._listeners[kind] || []).push(fn); }
  removeEventListener(kind, fn) {
    const l = this._listeners[kind] || [];
    const i = l.indexOf(fn); if (i >= 0) l.splice(i, 1);
  }
  /** 触发监听器, 返回触发次数 —— 0 表示"根本没绑上" */
  dispatchEvent(ev) {
    const kind = ev && ev.type;
    let n = 0;
    const l = this._listeners[kind] || [];
    if (!ev.target) ev.target = this;
    if (!ev.preventDefault) ev.preventDefault = () => { ev.defaultPrevented = true; };
    for (const fn of l.slice()) { fn.call(this, ev); n++; }
    return n;
  }
  click() { return this.dispatchEvent({ type: 'click', target: this }); }
  focus() {}
  scrollIntoView() {}
  closest(sel) {
    let n = this;
    while (n) {
      if (sel.startsWith('.') && n._classes.has(sel.slice(1))) return n;
      if (sel.startsWith('#') && n.id === sel.slice(1)) return n;
      n = n.parentNode;
    }
    return null;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  querySelectorAll(sel) {
    const out = [];
    const walk = (n) => { for (const c of n.children) { if (match(c, sel)) out.push(c); walk(c); } };
    walk(this);
    return out;
  }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); }
  setAttribute(k, v) { this.attrs[k] = v; if (k === 'id') this.id = v; }
  hasAttribute(k) { return k in this.attrs; }
  getBoundingClientRect() { return { top: 0, left: 0, width: 100, height: 20 }; }
}

function match(node, sel) {
  sel = sel.trim();
  if (sel.startsWith('#')) return node.id === sel.slice(1);
  if (sel.startsWith('.')) return node._classes.has(sel.slice(1));
  if (sel.startsWith('[')) return node.hasAttribute(sel.slice(1, -1).split('=')[0]);
  return node.tagName === sel.toUpperCase();
}

const VOID = new Set(['br', 'hr', 'img', 'input', 'meta', 'link', 'source']);
function parseFragment(src, parent) {
  const out = [];
  const stack = [];
  const re = /<!--[\s\S]*?-->|<\/?([a-zA-Z][a-zA-Z0-9]*)((?:\s+[^<>]*?)?)(\/?)>/g;
  let m;
  while ((m = re.exec(src))) {
    if (m[0].startsWith('<!--')) continue;
    const tag = m[1].toLowerCase();
    if (m[0].startsWith('</')) { stack.pop(); continue; }
    const attrs = {};
    const ar = /([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?:="([^"]*)")?/g;
    let a;
    while ((a = ar.exec(m[2] || ''))) attrs[a[1]] = a[2] === undefined ? '' : a[2];
    const node = new El(tag, attrs);
    node.parentNode = stack.length ? stack[stack.length - 1] : parent;
    (stack.length ? stack[stack.length - 1].children : out).push(node);
    if (!(m[3] === '/' || VOID.has(tag))) stack.push(node);
  }
  return out;
}

/* ---------------- 顶层容器 ---------------- */
const roots = parseFragment(html, null);
const byId = new Map();
(function indexAll(n) {
  for (const c of n.children) {
    if (c.id) byId.set(c.id, c);
    indexAll(c);
  }
})({ children: roots });

function all(sel) {
  const out = [];
  const walk = (n) => {
    for (const c of n.children) {
      if (match(c, sel)) out.push(c);
      walk(c);
    }
  };
  for (const r of roots) {
    if (match(r, sel)) out.push(r);
    walk(r);
  }
  return out;
}
const doc = {
  body: roots.find((r) => r.tagName === 'BODY') || new El('body'),
  documentElement: roots.find((r) => r.tagName === 'HTML') || new El('html'),
  querySelector: (sel) => all(sel)[0] || null,
  querySelectorAll: (sel) => all(sel),
  getElementById: (id) => byId.get(id) || null,
  createElement: (t) => new El(t),
  addEventListener: () => {},
  removeEventListener: () => {},
};

/* ---------------- 其余宿主 API 桩 ---------------- */
const errors = [];
const sandbox = {
  document: doc,
  window: { document: doc, location: { href: 'http://127.0.0.1:8760/', reload() {} } },
  console: { log: () => {}, warn: () => {}, error: (...a) => errors.push(a.join(' ')) },
  setTimeout: (fn) => 0,
  clearTimeout: () => {},
  setInterval: () => 0,
  clearInterval: () => {},
  localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
  fetch: (url) => {
    const p = String(url);
    let payload = { ok: true };
    if (p.includes('/api/meta')) {
      const specs = {};
      for (let i = 0; i < 18; i++) specs['p' + i] = { type: 'int', desc: 'd', choices: null, default: 1 };
      payload = {
        ok: true, pref_specs: specs, prefs: { fps: 30, default_demo: 'D:\\x.dem' },
        defaults: { fps: 30, max_clips: 22, max_cards: 40 },
        aspects: ['square', 'tall', 'wide'], default_demo: 'D:\\x.dem',
        default_demo_exists: true, music_dir: 'D:\\CloudMusic', busy: false, stats: {},
      };
    } else if (p.includes('/api/demos')) {
      payload = { ok: true, files: [{ name: 'a.dem', path: 'D:\\a.dem', size: 1 }] };
    } else if (p.includes('/api/jobs')) {
      payload = { ok: true, jobs: [] };
    }
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(payload) });
  },
  EventSource: class { constructor() { this.readyState = 0; } close() {} addEventListener() {} },
  confirm: () => true,
  alert: () => {},
};
sandbox.window.window = sandbox.window;
sandbox.globalThis = sandbox;

/* ---------------- 真跑 app.js ---------------- */
console.log('='.repeat(72));
console.log('无头执行 app.js（普通 <script> 语义：顶层抛错即整体中断）');
console.log('='.repeat(72));
let topLevelError = null;
try {
  vm.createContext(sandbox);
  vm.runInContext(js, sandbox, { filename: 'app.js' });
  console.log('  顶层执行无异常 [OK]');
} catch (e) {
  topLevelError = e;
  console.log(`  [ERR] 顶层抛错: ${e.name}: ${e.message}`);
  console.log(String(e.stack).split('\n').slice(1, 5).map((l) => '     ' + l.trim()).join('\n'));
}

/* ---------------- 关键控件是否绑上 ---------------- */
// 这三个是**故意**不在顶层绑的:
//   #btn-download / #btn-reveal: 出片完成时才在 loadResult() 里绑
//   #demo-path: 用内联 onchange? 不, 它只在选择时读值, 不需要监听器
const DYNAMIC = new Set(['btn-download', 'btn-reveal', 'demo-path']);
const CONTROLS = [
  'btn-run', 'btn-cancel', 'btn-browse-music', 'modal-close', 'browse-go', 'browse-up',
  'btn-demo-refresh', 'btn-prefs-save', 'btn-prefs-reload', 'btn-prefs-reset',
  'btn-profile-reload', 'btn-profile-rebuild', 'btn-observe', 'btn-runs-reload',
  'btn-run-detail-close', 'music-path',
].filter((id) => !DYNAMIC.has(id));
const unbound = [];
for (const id of CONTROLS) {
  const el = byId.get(id);
  if (!el) { unbound.push(`#${id}(不存在)`); continue; }
  const kinds = Object.keys(el._listeners);
  if (!kinds.length) unbound.push(`#${id}(没有监听器)`);
}
console.log(`\n  未绑定监听器的控件: ${unbound.length ? unbound.join(', ') : '无 [OK]'}`);

/* ---------------- 真点一下关闭按钮 ---------------- */
const modal = byId.get('modal-music');
const closeBtn = byId.get('modal-close');
console.log('\n  ---- 模拟点击"关闭" ----');
console.log(`  #modal-music.hidden(初始) = ${modal ? modal.hidden : 'n/a'}`);
if (modal) { modal.hidden = false; modal.classList.remove('is-hidden'); }
let fired = 0;
if (closeBtn) fired = closeBtn.click();
console.log(`  #modal-close 触发监听器数量 = ${fired}`);
console.log(`  点击后 hidden = ${modal ? modal.hidden : 'n/a'}`);
console.log(`  点击后 class  = ${modal ? modal.className : 'n/a'}`);
const clickOk = !!modal && modal.hidden === true && modal.classList.contains('is-hidden');
console.log(`  => ${clickOk ? '关闭生效 [OK]' : '[!] 关闭无效 —— 复现了用户报的问题'}`);

/* ---------------- Esc 关闭 ---------------- */
let escFired = 0;
if (modal) { modal.hidden = false; }
// document.addEventListener 在桩里是 no-op, 所以这里直接找 keydown 监听器
escFired = doc._keyListeners ? doc._keyListeners.length : 0;
console.log(`\n  Esc 关闭监听器: ${escFired ? '已绑定' : '未绑定(document.addEventListener 是桩, 仅提示)'}`);

console.log('\n' + '='.repeat(72));
const ok = !topLevelError && !unbound.length && clickOk;
console.log(ok ? '结论: 前端事件绑定与关闭逻辑都正常' : '结论: 存在问题, 见上面 [!] 标记');
console.log('='.repeat(72));
process.exit(ok ? 0 : 1);
