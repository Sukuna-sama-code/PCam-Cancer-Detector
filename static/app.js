import {
  initLocalEngine,
  analyzeSlideLocal,
  reRenderLastOptics,
  runBatchLocal,
  initBiomarkerLocal,
  predictBiomarkerLocal,
} from './onnx-engine.js';

const USE_API_BACKEND = new URLSearchParams(window.location.search).get('backend') === 'api';

const THEME_KEYS = ['gold-espresso', 'espresso-marble', 'mocha-copper', 'cedar-slats', 'chestnut-amber'];
const THEME_META = {
  'gold-espresso': { name: 'Liquid Gold & Espresso', img: './themes/theme-1-gold-espresso.jpg' },
  'espresso-marble': { name: 'Espresso & Cream Marble', img: './themes/theme-2-espresso-marble.jpg' },
  'mocha-copper': { name: 'Mocha & Copper Swirl', img: './themes/theme-3-mocha-copper.jpg' },
  'cedar-slats': { name: 'Warm Cedar Wood Slats', img: './themes/theme-4-cedar-slats.jpg' },
  'chestnut-amber': { name: 'Chestnut & Molten Amber', img: './themes/theme-5-chestnut-amber.jpg' },
};

let defaultThreshold = 0.1261;
let uploadedFileObj = null;
let lastAnalysis = null;
let latestBatchData = null;
let batchFilter = 'all';
let bioMeta = null;
let bioInitialized = false;
let insightsRendered = false;
const bioValues = {};

function getThemeTokens() {
  const cs = getComputedStyle(document.documentElement);
  const read = (v, fb) => cs.getPropertyValue(v).trim() || fb;
  return {
    accent: read('--accent', '#FF8A00'),
    accentHi: read('--accent-hi', '#FFB800'),
    heading: read('--heading', '#FFFFFF'),
    ink: read('--ink', '#F4F4F6'),
    muted: read('--muted', '#A1A1AA'),
    surface: read('--surface', '#0B0B0E'),
    surface2: read('--surface-2', '#111116'),
  };
}

function setThemeMenuOpen(open) {
  const menu = document.getElementById('bgthemes-menu');
  const btn = document.getElementById('bgthemes-btn');
  if (!menu || !btn) return;
  menu.classList.toggle('hidden', !open);
  btn.setAttribute('aria-expanded', String(open));
}

function applyTheme(themeKey, notify = false) {
  const safeKey = THEME_KEYS.includes(themeKey) ? themeKey : 'gold-espresso';
  document.documentElement.dataset.theme = safeKey;
  try { localStorage.setItem('aetheris_theme', safeKey); } catch (_) { /* ignore storage restrictions */ }

  const thumb = document.getElementById('bgthemes-active-thumb');
  if (thumb && THEME_META[safeKey]) thumb.src = THEME_META[safeKey].img;

  document.querySelectorAll('.theme-option').forEach((btn) => {
    const active = btn.dataset.themePick === safeKey;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-checked', String(active));
  });

  if (lastAnalysis?.attribution) renderSpatialAttribution(lastAnalysis.attribution);
  if (bioInitialized && bioMeta) predictBiomarker();
  if (insightsRendered) renderInsightsMatrix();
  if (notify) showToast(`BGThemes: ${THEME_META[safeKey].name}`);
}

function initThemeSwitcher() {
  let saved = 'gold-espresso';
  try {
    const stored = localStorage.getItem('aetheris_theme');
    if (stored && THEME_KEYS.includes(stored)) saved = stored;
  } catch (_) { /* ignore */ }
  applyTheme(saved, false);

  const btn = document.getElementById('bgthemes-btn');
  const menu = document.getElementById('bgthemes-menu');
  const wrap = document.getElementById('bgthemes-dropdown');

  btn?.addEventListener('click', (e) => {
    e.stopPropagation();
    setThemeMenuOpen(menu?.classList.contains('hidden'));
  });

  document.querySelectorAll('.theme-option').forEach((opt) => {
    opt.addEventListener('click', () => {
      applyTheme(opt.dataset.themePick, true);
      setThemeMenuOpen(false);
    });
  });

  document.addEventListener('click', (e) => {
    if (wrap && !wrap.contains(e.target)) setThemeMenuOpen(false);
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') setThemeMenuOpen(false);
  });
}

function initHelpAndFeedback() {
  // 1. Help Modal
  const helpDlg = document.getElementById('help-dialog');
  document.getElementById('help-btn')?.addEventListener('click', () => helpDlg?.showModal());
  document.getElementById('close-help-btn')?.addEventListener('click', () => helpDlg?.close());
  document.getElementById('done-help-btn')?.addEventListener('click', () => helpDlg?.close());
  helpDlg?.addEventListener('click', (e) => { if (e.target === helpDlg) helpDlg.close(); });

  // 2. Moving Outer-Corridor Feedback Heart Button
  const railBtn = document.getElementById('feedback-rail-btn');
  const fbDlg = document.getElementById('feedback-dialog');
  if (railBtn) {
    let rafId = 0;
    const syncRailPos = () => {
      rafId = 0;
      const maxScroll = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      const ratio = Math.min(1, Math.max(0, window.scrollY / maxScroll));
      const minTop = 120;
      const maxTop = Math.max(minTop + 40, window.innerHeight - 96);
      const y = Math.round(minTop + ratio * (maxTop - minTop));
      railBtn.style.setProperty('--rail-y', `${y}px`);
    };
    const onScrollOrResize = () => {
      if (!rafId) rafId = requestAnimationFrame(syncRailPos);
    };
    window.addEventListener('scroll', onScrollOrResize, { passive: true });
    window.addEventListener('resize', onScrollOrResize, { passive: true });
    syncRailPos();
    railBtn.addEventListener('click', () => fbDlg?.showModal());
  }

  // 3. Feedback Modal Form
  document.getElementById('close-feedback-btn')?.addEventListener('click', () => fbDlg?.close());
  fbDlg?.addEventListener('click', (e) => { if (e.target === fbDlg) fbDlg.close(); });

  let rating = 5;
  let category = 'UI & Themes';
  const heartBtns = document.querySelectorAll('.fb-heart-btn');
  const rateText = document.getElementById('feedback-rate-text');

  heartBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      rating = Number(btn.dataset.rate || 5);
      heartBtns.forEach((b) => b.classList.toggle('active', Number(b.dataset.rate) <= rating));
      if (rateText) rateText.textContent = `${rating} / 5`;
    });
  });

  document.querySelectorAll('#feedback-cats .micro-chip-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      category = btn.dataset.cat || 'UI & Themes';
      document.querySelectorAll('#feedback-cats .micro-chip-btn').forEach((b) => b.classList.toggle('active', b === btn));
    });
  });

  document.getElementById('feedback-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const msgEl = document.getElementById('feedback-msg');
    const payload = {
      timestamp: new Date().toISOString(),
      rating,
      category,
      message: msgEl?.value || '',
      theme: document.documentElement.dataset.theme || 'gold-espresso',
    };
    try {
      if (USE_API_BACKEND) {
        const res = await fetch('/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error('Feedback submission failed');
        showToast('Thank you for your feedback!');
      } else {
        const key = 'aetheris_feedback_log';
        const prev = JSON.parse(localStorage.getItem(key) || '[]');
        prev.push(payload);
        localStorage.setItem(key, JSON.stringify(prev));
        showToast('Thank you! Feedback saved in your workstation session.');
      }
      if (msgEl) msgEl.value = '';
      fbDlg?.close();
    } catch (err) {
      showToast(err.message);
    }
  });
}

function hasPlotlyTrace(traceType) {
  if (!window.Plotly) return false;
  if (traceType === 'scatter3d') return typeof window.Plotly.newPlot === 'function';
  try {
    return Boolean(window.Plotly.PlotSchema?.get?.().traces?.[traceType]);
  } catch (_) {
    return false;
  }
}

function showToast(msg) {
  const el = document.getElementById('toast-banner');
  if (!el) return;
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => el.classList.add('hidden'), 3200);
}

function animateNumber(el, target, suffix = '%', decimals = 2, duration = 420) {
  if (!el) return;
  const startVal = parseFloat(el.dataset.val || '0') || 0;
  const startTime = performance.now();
  function step(now) {
    const p = Math.min(1, (now - startTime) / duration);
    const eased = 1 - Math.pow(1 - p, 3);
    const cur = startVal + (target - startVal) * eased;
    el.textContent = `${cur.toFixed(decimals)}${suffix}`;
    if (p < 1) requestAnimationFrame(step);
    else el.dataset.val = String(target);
  }
  requestAnimationFrame(step);
}

// ============================================================================
// 1. WORKSPACE NAVIGATION & MODAL
// ============================================================================
function switchWorkspace(viewName) {
  const tabs = document.querySelectorAll('.nav-tab');
  const views = document.querySelectorAll('.view-section');
  const hero = document.getElementById('hero-pavilion');

  tabs.forEach((t) => {
    const active = t.dataset.view === viewName;
    t.classList.toggle('active', active);
    t.setAttribute('aria-selected', String(active));
  });
  views.forEach((v) => v.classList.toggle('active', v.id === `view-${viewName}`));
  hero?.classList.toggle('is-compact', viewName !== 'single');

  if (viewName === 'insights' && !insightsRendered) renderInsightsMatrix();
  if (viewName === 'biomarker' && !bioInitialized) initBiomarkerView();

  requestAnimationFrame(() => {
    ['bio-3d-plot', 'bio-imp-plot', 'insights-cm-plot'].forEach((id) => {
      const node = document.getElementById(id);
      if (node && node.data && window.Plotly?.Plots?.resize) {
        try { window.Plotly.Plots.resize(node); } catch (_) { /* ignore hidden resize */ }
      }
    });
  });
}

function initNavigation() {
  document.querySelectorAll('.nav-tab').forEach((tab) => {
    tab.addEventListener('click', () => switchWorkspace(tab.dataset.view));
  });
  document.getElementById('brand-home')?.addEventListener('click', (e) => {
    e.preventDefault();
    switchWorkspace('single');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  // Hero CTAs
  document.getElementById('hero-analyze-cta')?.addEventListener('click', () => {
    switchWorkspace('single');
    document.getElementById('view-single')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    runSingleAnalysis();
  });
  document.getElementById('hero-batch-cta')?.addEventListener('click', () => {
    switchWorkspace('batch');
    runBatchScreening(true);
  });

  // Optical Curtain Controls
  const splitBtn = document.getElementById('mode-split-btn');
  const curtainBtn = document.getElementById('mode-curtain-btn');
  const dualWrap = document.getElementById('dual-viewport-wrap');
  const curtainWrap = document.getElementById('curtain-viewport-wrap');

  splitBtn?.addEventListener('click', () => {
    splitBtn.classList.add('active');
    curtainBtn.classList.remove('active');
    dualWrap.classList.remove('hidden');
    curtainWrap.classList.add('hidden');
  });
  curtainBtn?.addEventListener('click', () => {
    curtainBtn.classList.add('active');
    splitBtn.classList.remove('active');
    curtainWrap.classList.remove('hidden');
    dualWrap.classList.add('hidden');
  });

  const cSlider = document.getElementById('curtain-slider');
  const cClip = document.getElementById('curtain-clip');
  const cHandle = document.getElementById('curtain-handle');
  const syncCurtain = () => {
    const pct = Number(cSlider.value);
    cClip.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    cHandle.style.left = `${pct}%`;
  };
  cSlider?.addEventListener('input', syncCurtain);
  syncCurtain();

  // Synchronized crosshair coordinate tracking on dual viewports
  const roiFrame = document.getElementById('frame-roi');
  const camFrame = document.getElementById('frame-cam');
  const coordPill = document.getElementById('meta-cursor-coords');
  const frames = [roiFrame, camFrame].filter(Boolean);

  frames.forEach((frame) => {
    frame.addEventListener('mousemove', (e) => {
      const r = frame.getBoundingClientRect();
      const nx = Math.min(1, Math.max(0, (e.clientX - r.left) / Math.max(1, r.width)));
      const ny = Math.min(1, Math.max(0, (e.clientY - r.top) / Math.max(1, r.height)));
      const px = Math.round(nx * 96);
      const py = Math.round(ny * 96);
      frames.forEach((f) => {
        f.classList.add('has-crosshair');
        f.style.setProperty('--cx', `${(nx * 100).toFixed(1)}%`);
        f.style.setProperty('--cy', `${(ny * 100).toFixed(1)}%`);
      });
      if (coordPill) {
        coordPill.classList.remove('hidden');
        coordPill.textContent = `X ${String(px).padStart(2, '0')} · Y ${String(py).padStart(2, '0')} px`;
      }
    });
    frame.addEventListener('mouseleave', () => {
      frames.forEach((f) => f.classList.remove('has-crosshair'));
      coordPill?.classList.add('hidden');
    });
  });

  // Methodology Modal
  const dialog = document.getElementById('about-dialog');
  document.getElementById('about-btn')?.addEventListener('click', () => dialog?.showModal());
  document.getElementById('close-about-btn')?.addEventListener('click', () => dialog?.close());
  document.getElementById('done-about-btn')?.addEventListener('click', () => dialog?.close());
  dialog?.addEventListener('click', (e) => {
    if (e.target === dialog) dialog.close();
  });

  // Keyboard shortcuts 1-4
  window.addEventListener('keydown', (e) => {
    if (e.target instanceof Element && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    const map = { '1': 'single', '2': 'batch', '3': 'biomarker', '4': 'insights' };
    if (map[e.key]) switchWorkspace(map[e.key]);
  });
}

// ============================================================================
// 2. SINGLE-SLIDE DIAGNOSTIC WORKFLOW (REAL PYTORCH API)
// ============================================================================
function renderSpatialAttribution(attr) {
  const host = document.getElementById('chart-attribution');
  if (!host) return;
  const tok = getThemeTokens();
  const rows = [
    { label: 'Center 32×32 Peak', val: attr.center_peak, grad: 'linear-gradient(90deg, #E11D48, #FB7185)' },
    { label: 'Center 32×32 Mean', val: attr.center_mean, grad: `linear-gradient(90deg, ${tok.accent}, ${tok.accentHi})` },
    { label: 'Outer Context Mean', val: attr.outer_mean, grad: `linear-gradient(90deg, #52525B, ${tok.muted})` },
    { label: '4× TTA Stability', val: attr.tta_stability, grad: 'linear-gradient(90deg, #059669, #34D399)' },
  ];
  host.innerHTML = `<div class="attr-bars-stack">${rows.map((r) => `
    <div class="attr-row">
      <div class="attr-row-head">
        <span>${r.label}</span>
        <span class="mono">${r.val.toFixed(1)}%</span>
      </div>
      <div class="attr-track">
        <div class="attr-fill" style="width:${Math.min(100, Math.max(2, r.val))}%; background:${r.grad};"></div>
      </div>
    </div>
  `).join('')}</div>`;
}

function updateVerdictUI(data, cutoffRatio) {
  const isCancer = data.raw_prob >= cutoffRatio;
  const thresholdPct = cutoffRatio * 100.0;

  const vBanner = document.getElementById('verdict-banner');
  vBanner.className = `verdict-pill ${isCancer ? 'cancer' : 'benign'}`;
  document.getElementById('verdict-title').textContent = isCancer
    ? 'Flagged for metastasis review'
    : 'Benign / normal lymph node tissue';
  document.getElementById('verdict-sub').textContent = isCancer
    ? 'Positive malignant signal detected inside central 32×32 ROI'
    : "No metastatic foci above calibrated Youden's J cutoff";

  animateNumber(document.getElementById('prob-big'), data.mean_prob, '%', 2);
  document.getElementById('cutoff-readout').textContent = `${thresholdPct.toFixed(2)}%`;

  const fill = document.getElementById('prob-bar-fill');
  fill.style.width = `${Math.min(100, Math.max(2, data.mean_prob))}%`;
  fill.style.background = isCancer
    ? 'linear-gradient(90deg, #E11D48, #FB7185)'
    : 'linear-gradient(90deg, #059669, #34D399)';

  const needle = document.getElementById('cutoff-needle');
  if (needle) needle.style.left = `${Math.min(99, Math.max(1, thresholdPct))}%`;

  const rotGrid = document.getElementById('rot-cards-grid');
  rotGrid.innerHTML = Object.entries(data.rot_probs).map(([ang, val]) => {
    const over = val >= thresholdPct;
    return `
      <div class="rot-card">
        <div class="rot-card-head">
          <span>ROT ${ang}</span>
          <span class="kind-dot ${over ? 'cancer' : 'normal'}" title="${over ? 'Above cutoff' : 'Below cutoff'}"></span>
        </div>
        <strong>${val.toFixed(1)}%</strong>
        <div class="rot-mini-track">
          <div class="rot-mini-fill" style="width:${Math.min(100, Math.max(3, val))}%; background:${over ? '#F43F5E' : '#10B981'};"></div>
        </div>
      </div>
    `;
  }).join('');
}

async function runSingleAnalysis() {
  const stage = document.getElementById('slide-stage');
  const badge = document.getElementById('tta-angle-badge');
  const sampleSelect = document.getElementById('sample-select');
  const threshold = Number(document.getElementById('threshold-slider').value);
  const sharpen = Number(document.getElementById('sharpen-slider').value);
  const enhance = document.getElementById('enhance-toggle').checked;

  if (!uploadedFileObj && !sampleSelect.value) return;

  stage.classList.add('is-scanning');
  stage.classList.remove('slide-entering');
  badge.textContent = '4× TTA • ROT 0°';

  const angles = ['0°', '90°', '180°', '270°'];
  let idx = 0;
  let spinTimer = null;
  if (USE_API_BACKEND) {
    spinTimer = setInterval(() => {
      idx = (idx + 1) % angles.length;
      badge.textContent = `4× TTA • ROT ${angles[idx]}`;
    }, 210);
  }

  const minChoreography = new Promise((res) => setTimeout(res, USE_API_BACKEND ? 820 : 560));

  try {
    let data;
    if (USE_API_BACKEND) {
      const form = new FormData();
      form.append('threshold', String(threshold));
      form.append('sharpen', String(sharpen));
      form.append('enhance', String(enhance));
      if (uploadedFileObj) form.append('file', uploadedFileObj);
      else form.append('sample_name', sampleSelect.value);

      const [res] = await Promise.all([fetch('/api/analyze', { method: 'POST', body: form }), minChoreography]);
      if (!res.ok) throw new Error((await res.json()).detail || 'Analysis failed');
      data = await res.json();
    } else {
      const [localRes] = await Promise.all([
        analyzeSlideLocal({
          sampleName: uploadedFileObj ? null : sampleSelect.value,
          file: uploadedFileObj || null,
          threshold,
          enhance,
          sharpen,
          onAngle: (ang) => { badge.textContent = `4× TTA • ROT ${ang}°`; },
        }),
        minChoreography,
      ]);
      data = localRes;
    }

    lastAnalysis = data;

    document.getElementById('meta-specimen').textContent = data.specimen;
    document.getElementById('meta-source').textContent = data.source;
    document.getElementById('meta-fov').textContent = data.native_size;
    document.getElementById('meta-intensity').textContent = `${data.mean_intensity} / 255`;
    document.getElementById('meta-elapsed').textContent = `${data.elapsed_ms} ms`;

    const latBadge = document.getElementById('nav-latency-badge');
    if (latBadge) {
      latBadge.textContent = `${data.elapsed_ms} ms`;
      latBadge.classList.remove('hidden');
    }

    document.getElementById('badge-hd-mode').textContent = enhance ? '6× LANCZOS-4 HD' : 'NATIVE 96×96 NN';
    document.getElementById('tta-spread-chip').textContent = `STD ±${data.tta_std.toFixed(1)}%`;

    document.getElementById('img-roi').src = data.roi_image;
    document.getElementById('img-cam').src = data.cam_image;
    document.getElementById('curtain-base-img').src = data.roi_image;
    document.getElementById('curtain-top-img').src = data.cam_image;

    void stage.offsetWidth;
    stage.classList.add('slide-entering');

    updateVerdictUI(data, threshold);
    renderSpatialAttribution(data.attribution);
  } catch (err) {
    showToast(`Analysis error: ${err.message}`);
  } finally {
    if (spinTimer) clearInterval(spinTimer);
    stage.classList.remove('is-scanning');
  }
}

// ============================================================================
// 3. BATCH SCREENING & CSV DOWNLOAD
// ============================================================================
function renderBatchTable() {
  if (!latestBatchData) return;
  const rows = latestBatchData.rows.filter((r) => batchFilter === 'all' || r.verdict === batchFilter);
  const tbody = document.getElementById('batch-table-body');
  if (rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="12" class="empty-row">No slides match the selected verdict filter.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map((r, i) => `
    <tr>
      <td class="mono">${i + 1}</td>
      <td><img class="table-thumb" src="${r.thumb_b64}" alt="${r.specimen}" /></td>
      <td><b>${r.specimen}</b></td>
      <td><span class="verdict-chip ${r.verdict === 'METASTATIC' ? 'cancer' : 'benign'}">${r.verdict}</span></td>
      <td class="mono"><b>${r.prob_pct.toFixed(2)}%</b></td>
      <td>${r.risk_tier}</td>
      <td>${r.ground_truth}</td>
      <td>${r.audit}</td>
      <td class="mono">${r.rot_0}%</td>
      <td class="mono">${r.rot_90}%</td>
      <td class="mono">${r.rot_180}%</td>
      <td class="mono">${r.rot_270}%</td>
    </tr>
  `).join('');
}

async function runBatchScreening(forceBuiltin = false) {
  const statusEl = document.getElementById('batch-status');
  const filesInput = document.getElementById('batch-files-input');
  const gtInput = document.getElementById('batch-gt-input');
  const useBuiltin = forceBuiltin ? true : document.getElementById('batch-builtin-check').checked;
  const threshold = Number(document.getElementById('threshold-slider').value);

  statusEl.textContent = 'RUNNING 4× TTA…';

  try {
    let data;
    if (USE_API_BACKEND) {
      const form = new FormData();
      form.append('threshold', String(threshold));
      form.append('use_builtin', String(useBuiltin));
      if (!forceBuiltin) {
        for (const f of filesInput.files) form.append('files', f);
      }
      if (gtInput.files[0]) form.append('gt_csv', gtInput.files[0]);

      const res = await fetch('/api/batch', { method: 'POST', body: form });
      if (!res.ok) throw new Error((await res.json()).detail || 'Batch run failed');
      data = await res.json();
    } else {
      data = await runBatchLocal({
        files: forceBuiltin ? [] : Array.from(filesInput.files || []),
        useBuiltin,
        gtCsvFile: gtInput.files?.[0] || null,
        threshold,
        onProgress: (done, total) => { statusEl.textContent = `SCREENING ${done} / ${total} SLIDES…`; },
      });
    }
    latestBatchData = data;

    document.getElementById('download-csv-btn').classList.remove('hidden');
    document.getElementById('batch-filters').classList.remove('hidden');

    document.getElementById('batch-kpis').innerHTML = `
      <div class="bento-card kpi-card"><small>Slides Screened</small><strong class="mono">${data.total}</strong><span>${data.elapsed_ms} ms total</span></div>
      <div class="bento-card kpi-card"><small>Metastatic Flagged</small><strong class="mono" style="color:#FB7185">${data.positive}</strong><span>Above cutoff</span></div>
      <div class="bento-card kpi-card"><small>Benign Cleared</small><strong class="mono" style="color:#34D399">${data.negative}</strong><span>Below cutoff</span></div>
      <div class="bento-card kpi-card"><small>${data.verified_accuracy !== null ? 'Verified Accuracy' : 'Mean Risk'}</small><strong class="mono">${data.verified_accuracy !== null ? data.verified_accuracy + '%' : data.mean_risk + '%'}</strong><span>Cohort audit</span></div>
    `;

    renderBatchTable();
    statusEl.textContent = `AUDITED ${data.total} SLIDES (${data.elapsed_ms} ms)`;
    showToast(`Batch screening completed across ${data.total} slides`);
  } catch (err) {
    statusEl.textContent = 'ERROR';
    showToast(`Batch error: ${err.message}`);
  } finally {
    // batch finished
  }
}

// ============================================================================
// 4. 3D BIOMARKER EXPLORER (OBSIDIAN WEBGL 3D SCATTER + SVG GAUGE + LIVE RF)
// ============================================================================
function renderSvgGauge(probPct, riskTier) {
  const host = document.getElementById('bio-gauge-plot');
  if (!host) return;
  const tok = getThemeTokens();
  const clamped = Math.min(100, Math.max(0, probPct));
  const color = clamped >= 70 ? '#FB7185' : clamped >= 35 ? tok.accentHi : '#34D399';
  const r = 74;
  const halfCirc = Math.PI * r;
  const dashOffset = halfCirc * (1 - clamped / 100);

  host.innerHTML = `
    <svg viewBox="0 0 200 124" width="230" height="145" aria-label="Malignancy probability gauge ${clamped.toFixed(1)}%">
      <path d="M 26 102 A ${r} ${r} 0 0 1 174 102" fill="none" stroke="rgba(255,255,255,0.09)" stroke-width="14" stroke-linecap="round" />
      <path d="M 26 102 A ${r} ${r} 0 0 1 174 102" fill="none" stroke="${color}" stroke-width="14" stroke-linecap="round"
        stroke-dasharray="${halfCirc.toFixed(1)}" stroke-dashoffset="${dashOffset.toFixed(1)}"
        style="transition: stroke-dashoffset 0.5s cubic-bezier(0.22,1,0.36,1), stroke 0.25s ease;" />
      <text x="100" y="88" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="26" font-weight="700" fill="${tok.heading}">${clamped.toFixed(1)}%</text>
      <text x="100" y="109" text-anchor="middle" font-family="Plus Jakarta Sans, sans-serif" font-size="11" font-weight="700" fill="${color}">${riskTier.toUpperCase()}</text>
    </svg>
  `;
}

function renderBiomarkerImportances(importances) {
  const host = document.getElementById('bio-imp-plot');
  if (!host) return;
  const tok = getThemeTokens();
  if (hasPlotlyTrace('bar')) {
    window.Plotly.newPlot(host, [{
      type: 'bar',
      orientation: 'h',
      x: importances.map((d) => d.value),
      y: importances.map((d) => d.feature),
      marker: { color: tok.accent },
      text: importances.map((d) => `${d.value}%`),
      textposition: 'outside',
      textfont: { family: 'JetBrains Mono', size: 10, color: tok.ink },
    }], {
      margin: { l: 128, r: 44, t: 8, b: 14 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: { visible: false, range: [0, Math.max(...importances.map((d) => d.value)) + 9] },
      yaxis: { tickfont: { size: 10, color: tok.ink } },
    }, { displayModeBar: false, responsive: true });
    return;
  }

  const maxV = Math.max(...importances.map((d) => d.value), 1);
  const ordered = [...importances].reverse();
  host.innerHTML = `<div class="attr-bars-stack" style="overflow-y:auto; max-height:260px;">${ordered.map((d) => `
    <div class="attr-row">
      <div class="attr-row-head"><span>${d.feature}</span><span class="mono">${d.value}%</span></div>
      <div class="attr-track"><div class="attr-fill" style="width:${(d.value / maxV) * 100}%; background:linear-gradient(90deg, ${tok.accent}, ${tok.accentHi});"></div></div>
    </div>
  `).join('')}</div>`;
}

function applyBiomarkerPreset(presetName) {
  if (!bioMeta?.presets?.[presetName]) return;
  const preset = bioMeta.presets[presetName];
  bioMeta.features.forEach((feat, i) => {
    const val = preset[feat];
    bioValues[feat] = val;
    const input = document.querySelector(`input[data-feat="${feat}"]`);
    const label = document.getElementById(`bv-${i}`);
    if (input) input.value = String(val);
    if (label) label.textContent = String(val);
  });
  document.querySelectorAll('.preset-group .mini-btn').forEach((b) => {
    b.classList.toggle('active', b.id === `bio-preset-${presetName}`);
  });
  predictBiomarker();
}

async function initBiomarkerView() {
  bioInitialized = true;
  try {
    if (USE_API_BACKEND) {
      const res = await fetch('/api/biomarker/init');
      bioMeta = await res.json();
    } else {
      bioMeta = await initBiomarkerLocal();
    }
    document.getElementById('bio-acc-pill').textContent = `RF Test Accuracy: ${bioMeta.accuracy}% (${bioMeta.n_samples} cases)`;

    const geomWrap = document.getElementById('biomarker-sliders-geom');
    const shapeWrap = document.getElementById('biomarker-sliders-shape');
    geomWrap.innerHTML = '';
    shapeWrap.innerHTML = '';

    let debounceTimer = null;
    bioMeta.features.forEach((feat, i) => {
      const [min, max, step] = bioMeta.ranges[feat];
      const def = bioMeta.defaults[feat];
      bioValues[feat] = def;
      const row = document.createElement('div');
      row.className = 'bio-slider-row';
      row.innerHTML = `
        <div class="bio-slider-head"><span>${feat}</span><b class="mono" id="bv-${i}">${def}</b></div>
        <input type="range" min="${min}" max="${max}" step="${step}" value="${def}" data-feat="${feat}" data-idx="${i}" />
      `;
      row.querySelector('input').addEventListener('input', (e) => {
        const val = Number(e.target.value);
        bioValues[feat] = val;
        document.getElementById(`bv-${i}`).textContent = String(val);
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(predictBiomarker, 150);
      });
      (i < 5 ? geomWrap : shapeWrap).appendChild(row);
    });

    await predictBiomarker();
  } catch (err) {
    showToast(`Biomarker init failed: ${err.message}`);
  }
}

async function predictBiomarker() {
  if (!bioMeta) return;
  try {
    let data;
    if (USE_API_BACKEND) {
      const res = await fetch('/api/biomarker/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ values: bioValues }),
      });
      data = await res.json();
    } else {
      data = await predictBiomarkerLocal(bioValues);
    }
    const isHigh = data.malignant_prob >= 50;

    document.getElementById('bio-tier-chip').textContent = data.risk_tier.toUpperCase();
    const bv = document.getElementById('bio-verdict');
    bv.className = `verdict-pill ${isHigh ? 'cancer' : 'benign'}`;
    bv.innerHTML = `<strong>${isHigh ? 'High Malignancy Profile' : 'Low Risk / Benign Profile'}</strong><span>Malignant probability: ${data.malignant_prob}% · Benign: ${data.benign_prob}%</span>`;

    if (window.Plotly) {
      const tok = getThemeTokens();
      const axisStyle = (titleText) => ({
        title: { text: titleText, font: { size: 11, color: tok.ink } },
        tickfont: { size: 10, color: tok.muted },
        gridcolor: 'rgba(255,255,255,0.10)',
        zerolinecolor: 'rgba(255,255,255,0.18)',
        backgroundcolor: '#08080B',
        showbackground: true,
      });

      window.Plotly.newPlot('bio-3d-plot', [
        { type: 'scatter3d', mode: 'markers', name: 'Benign', x: bioMeta.benign.x, y: bioMeta.benign.y, z: bioMeta.benign.z, marker: { size: 3.6, color: '#34D399', opacity: 0.68 } },
        { type: 'scatter3d', mode: 'markers', name: 'Malignant', x: bioMeta.malignant.x, y: bioMeta.malignant.y, z: bioMeta.malignant.z, marker: { size: 3.6, color: '#FB7185', opacity: 0.68 } },
        { type: 'scatter3d', mode: 'markers', name: 'Active Patient', x: [bioValues['mean radius']], y: [bioValues['mean concavity']], z: [bioValues['worst radius']], marker: { size: 9.5, color: tok.accentHi, symbol: 'diamond', line: { color: tok.heading, width: 2.2 } } },
      ], {
        margin: { l: 16, r: 16, t: 28, b: 28 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        legend: { orientation: 'h', y: 1.02, x: 0.02, font: { size: 11, color: tok.ink } },
        scene: {
          bgcolor: '#08080B',
          camera: { eye: { x: 1.65, y: 1.60, z: 1.28 }, center: { x: 0, y: 0, z: -0.05 } },
          xaxis: axisStyle('Mean Radius'),
          yaxis: axisStyle('Mean Concavity'),
          zaxis: axisStyle('Worst Radius'),
        },
      }, { responsive: true });
    }

    renderSvgGauge(data.malignant_prob, data.risk_tier);
    renderBiomarkerImportances(data.importances);
  } catch (err) {
    showToast(`Biomarker prediction error: ${err.message}`);
  }
}

// ============================================================================
// 5. MODEL INSIGHTS CONFUSION MATRIX (OBSIDIAN THEME)
// ============================================================================
const CM_DETAILS = {
  tn: 'TN = 290: Benign lymph-node tissue patches correctly cleared below Youden’s J cutoff (Specificity 97.97%).',
  fp: 'FP = 6: False alarms sent for specialist review out of 296 normal patches — deliberate high-sensitivity safety margin.',
  fn: 'FN = 0: Zero missed metastatic patches across all 204 cancer specimens (Sensitivity 100.00%).',
  tp: 'TP = 204: Every metastatic lymph-node patch detected cleanly inside the central 32×32 ROI.',
};

function renderInsightsMatrix() {
  insightsRendered = true;
  const host = document.getElementById('insights-cm-plot');
  if (host && hasPlotlyTrace('heatmap')) {
    window.Plotly.newPlot(host, [{
      type: 'heatmap',
      z: [[290, 6], [0, 204]],
      x: ['Predicted Benign', 'Predicted Metastatic'],
      y: ['Actual Benign', 'Actual Metastatic'],
      text: [['TN: 290', 'FP: 6'], ['FN: 0', 'TP: 204']],
      texttemplate: '%{text}',
      textfont: { family: 'JetBrains Mono', size: 15, color: '#FFFFFF' },
      colorscale: [[0, '#121218'], [0.45, '#174A38'], [1, '#10B981']],
      showscale: false,
    }], {
      margin: { l: 122, r: 22, t: 18, b: 46 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: { tickfont: { color: '#E4E4E7' } },
      yaxis: { autorange: 'reversed', tickfont: { color: '#E4E4E7' } },
    }, { displayModeBar: false, responsive: true });
  } else if (host) {
    host.innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;height:100%;padding:18px 0;">
        <div class="bento-card" style="background:rgba(16,185,129,0.12);border-color:rgba(16,185,129,0.35);display:grid;place-items:center;"><strong class="mono" style="font-size:1.6rem;color:#34D399;">TN: 290</strong><small style="color:#A1A1AA;">Actual Benign · Pred Benign</small></div>
        <div class="bento-card" style="background:rgba(255,138,0,0.12);border-color:rgba(255,138,0,0.35);display:grid;place-items:center;"><strong class="mono" style="font-size:1.6rem;color:#FFB800;">FP: 6</strong><small style="color:#A1A1AA;">Actual Benign · Pred Metastatic</small></div>
        <div class="bento-card" style="background:#111116;display:grid;place-items:center;"><strong class="mono" style="font-size:1.6rem;color:#F4F4F6;">FN: 0</strong><small style="color:#A1A1AA;">Actual Metastatic · Pred Benign</small></div>
        <div class="bento-card" style="background:rgba(16,185,129,0.18);border-color:rgba(16,185,129,0.45);display:grid;place-items:center;"><strong class="mono" style="font-size:1.6rem;color:#34D399;">TP: 204</strong><small style="color:#A1A1AA;">Actual Metastatic · Pred Metastatic</small></div>
      </div>`;
  }

  document.querySelectorAll('#cm-inspector .cm-cell-chip').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#cm-inspector .cm-cell-chip').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('cm-note').textContent = CM_DETAILS[btn.dataset.cm] || '';
    });
  });
}

// ============================================================================
// 6. BOOTSTRAP & EVENT WIRING
// ============================================================================
function wireDropzone(el, onFiles) {
  if (!el) return;
  el.addEventListener('dragover', (e) => {
    e.preventDefault();
    el.classList.add('is-dragover');
  });
  el.addEventListener('dragleave', () => el.classList.remove('is-dragover'));
  el.addEventListener('drop', (e) => {
    e.preventDefault();
    el.classList.remove('is-dragover');
    if (e.dataTransfer?.files?.length) onFiles(e.dataTransfer.files);
  });
}

function formatShortSpecimen(s) {
  const m = s.filename.match(/^(cancer|normal)_patch_(\d+)_idx(\d+)/i);
  if (m) {
    const tag = m[1].toLowerCase() === 'cancer' ? 'Cancer' : 'Normal';
    return `${tag} #${m[2]} · ${m[3]}`;
  }
  return s.short_name;
}

function buildSampleFilmstrip(samples, activeFilename) {
  const strip = document.getElementById('sample-filmstrip');
  const sel = document.getElementById('sample-select');
  if (!strip || !sel) return;

  strip.innerHTML = samples.map((s) => `
    <button type="button" role="option" class="sample-thumb-btn ${s.filename === activeFilename ? 'active' : ''}" data-sample="${s.filename}">
      <img src="${s.thumb_url}" alt="${s.label}" loading="lazy" />
      <div class="sample-thumb-meta">
        <strong>${formatShortSpecimen(s)}</strong>
        <span class="sample-thumb-kind"><i class="kind-dot ${s.kind}"></i>${s.kind.toUpperCase()}</span>
      </div>
    </button>
  `).join('');

  strip.querySelectorAll('.sample-thumb-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      uploadedFileObj = null;
      sel.value = btn.dataset.sample;
      syncFilmstripActive(btn.dataset.sample);
      runSingleAnalysis();
    });
  });
}

function syncFilmstripActive(filename) {
  document.querySelectorAll('.sample-thumb-btn').forEach((btn) => {
    btn.classList.toggle('active', Boolean(filename) && btn.dataset.sample === filename);
  });
}

async function boot() {
  initThemeSwitcher();
  initHelpAndFeedback();
  initNavigation();

  const statusDot = document.getElementById('nav-status-dot');
  const statusText = document.getElementById('nav-status-text');
  let status;

  try {
    if (USE_API_BACKEND) {
      const statusRes = await fetch('/api/status');
      status = await statusRes.json();
    } else {
      statusDot.style.background = '#FFB800';
      statusText.textContent = 'Loading ResNet-34 ONNX (0%)…';
      status = await initLocalEngine((pct) => {
        statusText.textContent = pct >= 100
          ? 'Compiling WASM Engine…'
          : `Loading ResNet-34 ONNX (${pct}%)…`;
      });
    }
  } catch (err) {
    statusDot.style.background = '#F43F5E';
    statusText.textContent = 'Model Init Error';
    showToast(`Engine load failed: ${err.message}`);
    return;
  }

  defaultThreshold = Number(status.default_threshold || 0.1261);

  statusText.textContent = status.model_loaded
    ? `ResNet-34 Online (${status.device})`
    : 'Standby (Weights Missing)';
  statusDot.style.background = status.model_loaded ? '#10B981' : '#FFB800';

  const threshSlider = document.getElementById('threshold-slider');
  const threshVal = document.getElementById('threshold-val');
  threshSlider.value = String(defaultThreshold);
  threshVal.textContent = defaultThreshold.toFixed(3);
  document.getElementById('reset-youden-btn').textContent = `J=${defaultThreshold.toFixed(3)}`;

  document.getElementById('hero-val-auc').textContent = status.val_auc.toFixed(4);
  document.getElementById('ins-val-auc').textContent = status.val_auc.toFixed(4);
  document.getElementById('ins-test-auc').textContent = status.test_auc.toFixed(4);
  document.getElementById('ins-thresh').textContent = defaultThreshold.toFixed(3);
  if (status.checkpoint_metrics?.calibrated_test_acc) {
    document.getElementById('ins-test-acc').textContent = `${(status.checkpoint_metrics.calibrated_test_acc * 100).toFixed(2)}%`;
  }

  const sel = document.getElementById('sample-select');
  sel.innerHTML = status.samples.map((s) => `<option value="${s.filename}">${s.label}</option>`).join('');
  buildSampleFilmstrip(status.samples, sel.value);

  threshSlider.addEventListener('input', (e) => {
    const val = Number(e.target.value);
    threshVal.textContent = val.toFixed(3);
    if (lastAnalysis) updateVerdictUI(lastAnalysis, val);
  });
  document.getElementById('reset-youden-btn').addEventListener('click', () => {
    threshSlider.value = String(defaultThreshold);
    threshVal.textContent = defaultThreshold.toFixed(3);
    if (lastAnalysis) updateVerdictUI(lastAnalysis, defaultThreshold);
  });

  let opticTimer = null;
  const scheduleOpticRefresh = () => {
    clearTimeout(opticTimer);
    opticTimer = setTimeout(() => {
      if (!lastAnalysis) return;
      if (USE_API_BACKEND) {
        runSingleAnalysis();
      } else {
        const enh = document.getElementById('enhance-toggle').checked;
        const shp = Number(document.getElementById('sharpen-slider').value);
        const updated = reRenderLastOptics(enh, shp);
        if (updated) {
          lastAnalysis = updated;
          document.getElementById('badge-hd-mode').textContent = enh ? '6× LANCZOS-4 HD' : 'NATIVE 96×96 NN';
          document.getElementById('img-roi').src = updated.roi_image;
          document.getElementById('img-cam').src = updated.cam_image;
          document.getElementById('curtain-base-img').src = updated.roi_image;
          document.getElementById('curtain-top-img').src = updated.cam_image;
        }
      }
    }, USE_API_BACKEND ? 240 : 40);
  };
  document.getElementById('sharpen-slider').addEventListener('input', (e) => {
    document.getElementById('sharpen-val').textContent = Number(e.target.value).toFixed(2);
    scheduleOpticRefresh();
  });
  document.getElementById('enhance-toggle').addEventListener('change', scheduleOpticRefresh);

  const upIn = document.getElementById('upload-input');
  const dropSingle = document.getElementById('dropzone-single');
  dropSingle.addEventListener('click', () => upIn.click());
  dropSingle.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); upIn.click(); }
  });
  upIn.addEventListener('change', () => {
    uploadedFileObj = upIn.files[0] || null;
    if (uploadedFileObj) {
      syncFilmstripActive(null);
      runSingleAnalysis();
    }
  });
  wireDropzone(dropSingle, (files) => {
    uploadedFileObj = files[0];
    syncFilmstripActive(null);
    runSingleAnalysis();
  });

  sel.addEventListener('change', () => {
    uploadedFileObj = null;
    syncFilmstripActive(sel.value);
    runSingleAnalysis();
  });
  document.getElementById('run-analysis-btn').addEventListener('click', runSingleAnalysis);

  document.getElementById('export-json-btn').addEventListener('click', () => {
    if (!lastAnalysis) return showToast('Run a slide analysis first.');
    const { roi_image, cam_image, ...telemetry } = lastAnalysis;
    const blob = new Blob([JSON.stringify(telemetry, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `aetheris_${lastAnalysis.specimen}.json`;
    a.click();
  });

  const batchFilesIn = document.getElementById('batch-files-input');
  const batchGtIn = document.getElementById('batch-gt-input');
  batchFilesIn.addEventListener('change', () => {
    const n = batchFilesIn.files.length;
    document.getElementById('batch-files-label').textContent = n ? `${n} slide patch${n > 1 ? 'es' : ''} selected` : 'Drop patches here or click to browse multiple slides';
  });
  batchGtIn.addEventListener('change', () => {
    const f = batchGtIn.files[0];
    document.getElementById('batch-gt-label').textContent = f ? `Attached: ${f.name}` : 'Attach train_labels.csv for live batch accuracy';
  });
  wireDropzone(document.getElementById('batch-dropzone'), (files) => {
    batchFilesIn.files = files;
    batchFilesIn.dispatchEvent(new Event('change'));
  });

  document.getElementById('run-batch-btn').addEventListener('click', () => runBatchScreening(false));
  document.getElementById('run-builtin-batch-btn').addEventListener('click', () => runBatchScreening(true));
  document.querySelectorAll('#batch-filters .mini-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      batchFilter = btn.dataset.filter;
      document.querySelectorAll('#batch-filters .mini-btn').forEach((b) => b.classList.toggle('active', b === btn));
      renderBatchTable();
    });
  });
  document.getElementById('download-csv-btn').addEventListener('click', () => {
    if (!latestBatchData?.csv) return;
    const blob = new Blob([latestBatchData.csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'aetheris_pcam_batch_report.csv';
    a.click();
  });

  document.getElementById('run-biomarker-btn').addEventListener('click', predictBiomarker);
  document.getElementById('bio-preset-benign').addEventListener('click', () => applyBiomarkerPreset('benign'));
  document.getElementById('bio-preset-borderline').addEventListener('click', () => applyBiomarkerPreset('borderline'));
  document.getElementById('bio-preset-malignant').addEventListener('click', () => applyBiomarkerPreset('malignant'));

  if (status.model_loaded && status.samples.length > 0) {
    await runSingleAnalysis();
  }
}

boot();
