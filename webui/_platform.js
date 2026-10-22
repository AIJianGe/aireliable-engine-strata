/* ==========================================================================
 * Strata LLM 引擎 · 平台适配注入层（脚本）
 * --------------------------------------------------------------------------
 * 由 server.py 在反代**原生 Strata WebUI**（index.html）时以 <script defer>
 * 注入（见 /ui/{path} 路由）。它把平台侧能力叠在原生界面之上：
 *
 *   1. 顶部平台条：当前模式 / 引擎状态 / 本机硬件摘要
 *   2. 模型（模式）冷切换：首次 POST /load，切换 POST /mode/switch
 *      —— 这是平台「动态切换模型」的唯一入口
 *   3. /status 轮询：状态胶囊 + 模式同步（引擎被平台侧停止时也能感知）
 *   4. 中文提示（Toast / 二次确认弹窗）—— 原生界面本身保持英文不动
 *
 * 设计约束：
 *   - 与原生 app.js **零耦合**：只操作自己创建的 .pf-* DOM；
 *     唯一改动是隐藏原生「Model settings」卡片（样式层已处理）。
 *   - 平台端点一律走**根绝对路径**（/capabilities、/load…），
 *     因为页面 URL 是 /ui/ —— 写成相对路径会被解析成 /ui/xxx（原生端点）。
 *   - 不暴露原生 /api-monitor 独立监控页（原生 Monitor Tab 已内嵌，够用）。
 * ========================================================================== */

(function () {
  'use strict';

  /* ============================ 小工具 ============================ */

  var $ = function (sel, root) { return (root || document).querySelector(sel); };

  function mk(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  /** 单实例 Toast；错误 Toast 不自动消失（点击关闭）。 */
  function toast(title, msg, kind) {
    kind = kind || 'info';
    var box = $('#pf-toasts');
    if (!box) {
      box = mk('div', 'pf-toasts');
      box.id = 'pf-toasts';
      document.body.appendChild(box);
    }
    var el = mk('div', 'pf-toast pf-toast--' + kind);
    el.appendChild(mk('div', 'pf-toast__title', title));
    if (msg) el.appendChild(mk('div', 'pf-toast__msg', msg));
    box.appendChild(el);
    if (kind === 'error') {
      el.title = '点击关闭';
      el.addEventListener('click', function () { el.remove(); });
    } else {
      setTimeout(function () { el.remove(); }, 6000);
    }
  }

  /** 平台化确认弹窗（Promise<boolean>；不使用原生 confirm）。 */
  var confirmResolver = null;

  function ensureConfirm() {
    if ($('#pf-confirm')) return;
    var modal = mk('div', 'pf-modal');
    modal.id = 'pf-confirm';
    modal.hidden = true;
    var panel = mk('div', 'pf-modal__panel');
    panel.appendChild(mk('div', 'pf-modal__title', '确认'));
    panel.appendChild(mk('div', 'pf-modal__body'));
    var foot = mk('div', 'pf-modal__foot');
    var no = mk('button', 'pf-btn', '取消');
    var yes = mk('button', 'pf-btn pf-btn--primary', '确定');
    foot.appendChild(no); foot.appendChild(yes);
    panel.appendChild(foot);
    modal.appendChild(panel);
    document.body.appendChild(modal);

    var done = function (ok) {
      modal.hidden = true;
      if (confirmResolver) { var r = confirmResolver; confirmResolver = null; r(ok); }
    };
    no.addEventListener('click', function () { done(false); });
    yes.addEventListener('click', function () { done(true); });
  }

  function confirmDialog(title, body, warn) {
    ensureConfirm();
    var modal = $('#pf-confirm');
    $('.pf-modal__title', modal).textContent = title;
    var b = $('.pf-modal__body', modal);
    b.innerHTML = '';
    b.appendChild(mk('p', null, body));
    if (warn) b.appendChild(mk('div', 'pf-modal__warn', warn));
    modal.hidden = false;
    return new Promise(function (resolve) { confirmResolver = resolve; });
  }

  /** 错误详情弹窗（模型缺失等长错误信息） */
  function errorDialog(title, detail) {
    var modal = mk('div', 'pf-modal');
    var panel = mk('div', 'pf-modal__panel');
    panel.appendChild(mk('div', 'pf-modal__title', title));
    panel.appendChild(mk('div', 'pf-modal__err', detail));
    var foot = mk('div', 'pf-modal__foot');
    var ok = mk('button', 'pf-btn pf-btn--primary', '知道了');
    foot.appendChild(ok);
    panel.appendChild(foot);
    modal.appendChild(panel);
    document.body.appendChild(modal);
    ok.addEventListener('click', function () { modal.remove(); });
  }

  /* ============================ 平台 API ============================ */

  /** 平台端点（7010 根路径；页面在 /ui/ 下所以必须用绝对路径）。 */
  function api(path, opts) {
    return fetch(path, opts || { cache: 'no-store' }).then(function (r) {
      if (r.ok) return r.json();
      return r.json().catch(function () { return {}; }).then(function (j) {
        var msg = (j && j.error && j.error.message) || (j && j.detail) || ('HTTP ' + r.status);
        var err = new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
        err.status = r.status;
        throw err;
      });
    });
  }

  /* ============================ 状态 ============================ */

  var S = {
    caps: null,
    status: null,
    hw: null,
    busy: null,        // {label, since, timer, warn60, warn180}
    touched: false,    // 用户已手动改过下拉框 → 轮询 / 重渲染不得回弹其选择
    fullscreen: false,
  };

  var STATE_LABEL = {
    idle: '未加载',
    loading: '加载中',
    ready: '已就绪',
    inferring: '生成中',
    error: '错误',
  };

  /* ============================ 顶部平台条 ============================ */

  var bar = null;

  function buildBar() {
    bar = mk('div', 'pf-bar');
    bar.id = 'pf-bar';

    var brand = mk('div', 'pf-bar__brand');
    brand.appendChild(mk('span', null, 'Strata LLM'));
    brand.appendChild(mk('small', null, '推理引擎'));
    bar.appendChild(brand);

    bar.appendChild(mk('div', 'pf-bar__sep'));

    var select = mk('select', 'pf-select');
    select.id = 'pf-mode';
    select.title = '选择运行模式（不同模式 = 不同模型档位 / 上下文）';
    select.addEventListener('change', onModeChanged);
    bar.appendChild(select);

    var btn = mk('button', 'pf-btn pf-btn--primary');
    btn.id = 'pf-go';
    btn.textContent = '启动引擎';
    btn.addEventListener('click', onGoClicked);
    bar.appendChild(btn);

    bar.appendChild(mk('div', 'pf-bar__sep'));

    var pill = mk('span', 'pf-pill');
    pill.id = 'pf-pill';
    pill.dataset.state = 'idle';
    pill.appendChild(mk('span', 'pf-pill__dot'));
    pill.appendChild(mk('span', null, '未加载'));
    bar.appendChild(pill);

    var elapsed = mk('span', 'pf-bar__elapsed');
    elapsed.id = 'pf-elapsed';
    elapsed.hidden = true;
    bar.appendChild(elapsed);

    bar.appendChild(mk('div', 'pf-bar__spacer'));

    var hw = mk('span', 'pf-bar__hw');
    hw.id = 'pf-hw';
    hw.textContent = '硬件：检测中…';
    bar.appendChild(hw);

    document.body.insertBefore(bar, document.body.firstChild);
  }

  function renderHw() {
    var el = $('#pf-hw');
    if (!el) return;
    var d = (S.hw && S.hw.hardware) || {};
    if (!d.gpu && !d.ram_total_gb) { el.textContent = '硬件：无法检测'; return; }
    var parts = [];
    if (d.gpu) parts.push(d.gpu);
    if (d.vram_total_gb != null) parts.push(d.vram_total_gb + 'GB 显存');
    if (d.ram_total_gb != null) parts.push(d.ram_total_gb + 'GB 内存');
    if (d.compute_cap) parts.push('算力 ' + d.compute_cap);
    el.textContent = '硬件：' + parts.join(' · ');
    var rec = S.hw && S.hw.recommended_mode;
    if (rec) el.title = '本机推荐模式：' + rec + '（设备：' + (S.hw.device || '—') +
      '，MTP spec：' + (S.hw.spec != null ? S.hw.spec : '—') + '）';
  }

  function renderModes() {
    var select = $('#pf-mode');
    if (!select || !S.caps) return;
    var variants = S.caps.variants || [];
    var current = S.status && S.status.extra && S.status.extra.mode;
    var recommended = (S.hw && S.hw.recommended_mode) || S.caps.default_variant;

    select.innerHTML = '';
    variants.forEach(function (v) {
      var opt = mk('option');
      opt.value = v.name;
      var tags = [];
      if (v.mode_label) tags.push(v.mode_label);
      var quant = v.quant || v.precision;      // 能力声明里量化档字段是 precision
      if (quant) tags.push(quant);
      if (v.ctx_recommended) tags.push((v.ctx_recommended / 1024) + 'K');
      if (v.mtp) tags.push('MTP');
      if (v.vision) tags.push('vision');
      var label = tags.join(' · ') || v.name;
      if (v.name === recommended) label = '★ ' + label + '（推荐）';
      if (!v.available) label += '（未下载）';
      opt.textContent = label;
      opt.disabled = !v.available;
      select.appendChild(opt);
    });

    // 优先显示当前已加载模式，否则默认落在推荐档。
    // ⚠️ 用户已手动选择时**不得覆盖**：否则重渲染 / 轮询会把它弹回运行模式，
    // 导致点击「切换模式」实际变成用旧模式「重新加载」（真实浏览器实测踩中）。
    if (!S.touched) {
      var want = current || recommended || (variants[0] && variants[0].name);
      if (want) select.value = want;
      // 当前模式若不在列表（异常），补一项
      if (current && select.value !== current) {
        var opt2 = mk('option');
        opt2.value = current; opt2.textContent = current;
        select.appendChild(opt2);
        select.value = current;
      }
    }
  }

  function renderStatus() {
    var pill = $('#pf-pill');
    var select = $('#pf-mode');
    var go = $('#pf-go');
    if (!pill) return;

    var st = (S.status && S.status.state) || 'idle';
    var extra = (S.status && S.status.extra) || {};
    var running = !!extra.is_running;

    pill.dataset.state = st === 'ready' ? 'ready' : st === 'loading' ? 'loading'
      : st === 'inferring' ? 'inferring' : st === 'error' ? 'error' : 'idle';
    pill.lastChild.textContent = STATE_LABEL[st] || st;

    if (select) select.disabled = !!S.busy;
    if (go) {
      go.disabled = !!S.busy;
      var target = select ? select.value : null;
      go.textContent = running ? '切换模式' : '启动引擎';
      if (running && target && target === extra.mode) {
        go.textContent = '重新加载';
      }
    }
    if (bar) bar.dataset.mode = extra.mode || '';
    document.title = (extra.mode ? extra.mode + ' · ' : '') + 'Strata LLM';
  }

  /* ============================ 进度（加载 / 切换）============================ */

  function startBusy(label) {
    S.busy = { label: label, since: Date.now(), last: 0 };
    var el = $('#pf-elapsed');
    if (el) el.hidden = false;
    S.busy.timer = setInterval(function () {
      if (!S.busy) return;
      var sec = Math.round((Date.now() - S.busy.since) / 1000);
      var el2 = $('#pf-elapsed');
      if (el2) el2.textContent = S.busy.label + ' ' + sec + 's';
      if (sec >= 60 && !S.busy.warn60) {
        S.busy.warn60 = true;
        toast('启动较慢', '已用时 60s，Strata 需要把 35–55GB 权重载入显存/内存，请继续等待。', 'warn');
      }
      if (sec >= 180 && !S.busy.warn180) {
        S.busy.warn180 = true;
        toast('启动很慢', '已用时 180s。首次启动会建立权重索引缓存，属正常现象；若长时间无进展请查看平台「日志」页。', 'warn');
      }
    }, 1000);
  }

  function stopBusy() {
    if (S.busy && S.busy.timer) clearInterval(S.busy.timer);
    S.busy = null;
    var el = $('#pf-elapsed');
    if (el) { el.hidden = true; el.textContent = ''; }
  }

  /** 轮询 /status，直到 state 离开 loading/切换结束（或超时）。 */
  function waitSettled(timeoutMs) {
    var deadline = Date.now() + (timeoutMs || 600000);
    return new Promise(function (resolve, reject) {
      var tick = function () {
        api('/status').then(function (s) {
          S.status = s;
          renderStatus();
          if (s.state === 'ready' || s.state === 'inferring') return resolve(s);
          if (s.state === 'error') return reject(new Error('引擎进入 error 状态'));
          if (Date.now() > deadline) return reject(new Error('等待超时'));
          setTimeout(tick, 2000);
        }).catch(function () { setTimeout(tick, 2000); });
      };
      setTimeout(tick, 1500);
    });
  }

  /* ============================ 交互 ============================ */

  function onModeChanged() {
    S.touched = true;      // 标记用户已手动选择，阻止轮询把下拉框弹回运行模式
    renderStatus();
  }

  function onGoClicked() {
    var select = $('#pf-mode');
    var target = select && select.value;
    if (!target) { toast('无可选模式', '请先确认模型文件已下载。', 'warn'); return; }
    if (S.busy) return;

    var extra = (S.status && S.status.extra) || {};
    var running = !!extra.is_running;

    if (running && target === extra.mode) {
      doLoad(target, '重新加载');
      return;
    }
    if (!running) {
      doLoad(target, '启动引擎');
      return;
    }

    // 运行时换模型 = 冷切换（停 → 改写配置 → 冷启动）
    var v = findVariant(target);
    confirmDialog(
      '切换运行模式',
      '将把引擎切换到「' + ((v && v.mode_label) || target) + '」。Switching restarts the engine.',
      '⚠️ 冷切换需要：停止当前引擎 → 改写运行配置 → 重新冷启动。' +
      '期间需重新载入 35–55GB 权重，约 60–90s，界面暂时不可用。' +
      (v && v.expected_cold_start_s ? '本档预期冷启动约 ' + v.expected_cold_start_s + 's。' : '') +
      '切换失败会自动回滚到原模式。'
    ).then(function (ok) {
      if (!ok) return;
      doSwitch(target);
    });
  }

  function findVariant(name) {
    var list = (S.caps && S.caps.variants) || [];
    for (var i = 0; i < list.length; i++) if (list[i].name === name) return list[i];
    return null;
  }

  function doLoad(variant, label) {
    startBusy(label);
    renderStatus();
    toast(label, '正在冷启动 Strata（' + variant + '）…', 'info');
    api('/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ variant: variant }),
    }).then(function () {
      return waitSettled();
    }).then(function () {
      stopBusy();
      toast('启动完成', '模式 ' + variant + ' 已就绪。', 'success');
      setTimeout(function () { location.reload(); }, 600);
    }).catch(function (e) {
      stopBusy();
      renderStatus();
      toast('启动失败', e.message || String(e), 'error');
      if (e.status === 404 || e.status === 500) errorDialog('启动失败：模型或引擎文件不可用', e.message || String(e));
    });
  }

  function doSwitch(mode) {
    startBusy('切换中');
    renderStatus();
    toast('切换模式', '正在冷切换到 ' + mode + '…', 'info');
    api('/mode/switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: mode }),
    }).then(function () {
      return waitSettled();
    }).then(function () {
      stopBusy();
      toast('切换完成', '已切换到 ' + mode + '。', 'success');
      setTimeout(function () { location.reload(); }, 800);
    }).catch(function (e) {
      stopBusy();
      renderStatus();
      toast('切换失败', (e.message || String(e)) + '（已尝试回滚到原模式）', 'error');
    });
  }

  /* ============================ 轮询 ============================ */

  function poll() {
    if (S.busy) return;                     // 忙时由 waitSettled 接管
    api('/status').then(function (s) {
      S.status = s;
      renderStatus();
      // 模式变化（如平台侧切换）：同步下拉框。
      // 仅当用户**尚未**手动选择时才同步，否则会覆盖其待提交的切换意图。
      var select = $('#pf-mode');
      var cur = s.extra && s.extra.mode;
      if (!S.touched && select && cur && select.value !== cur && findVariant(cur)) {
        select.value = cur;
      }
    }).catch(function () { /* 引擎重启中，静默 */ });
  }

  /* ============================ 启动 ============================ */

  function init() {
    if (document.getElementById('pf-bar')) return;

    buildBar();
    renderStatus();

    api('/capabilities').then(function (c) {
      S.caps = c;
      renderModes();
      renderStatus();
    }).catch(function (e) {
      toast('能力声明获取失败', e.message || String(e), 'error');
    });

    api('/hardware').then(function (h) {
      S.hw = h;
      renderHw();
      renderModes();
    }).catch(function () { renderHw(); });

    poll();
    setInterval(poll, 3000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
