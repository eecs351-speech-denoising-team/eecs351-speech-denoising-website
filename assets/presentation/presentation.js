// Slide navigation, timer, audio playback, and the interactive denoising demo.
(function () {
  'use strict';
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  // ---------- Clips ----------
  const decode = (b64) => {
    const bin = atob(b64), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  };
  const CLIPS = {};
  for (const [name, c] of Object.entries(window.PRESENTATION_CLIPS || {})) {
    CLIPS[name] = { fs: c.fs, noisy: decode(c.noisy), clean: decode(c.clean) };
  }

  // ---------- Audio ----------
  const Player = {
    ctx: null, src: null, key: null, t0: 0, dur: 0, raf: 0, listeners: [],
    play(key, samples, fs, offset) {
      this.halt();
      const Ctx = window.AudioContext || window.webkitAudioContext;
      const ctx = this.ctx || (this.ctx = new Ctx());
      ctx.resume();
      const buf = ctx.createBuffer(1, samples.length, fs);
      buf.getChannelData(0).set(samples);
      const src = ctx.createBufferSource();
      src.buffer = buf; src.connect(ctx.destination);
      offset = clamp(offset || 0, 0, Math.max(0, buf.duration - 0.02));
      src.onended = () => { if (this.src === src) this.stop(); };
      src.start(0, offset);
      this.src = src; this.key = key; this.dur = buf.duration; this.t0 = ctx.currentTime - offset;
      const tick = () => { this.emit(); this.raf = requestAnimationFrame(tick); };
      tick();
    },
    position() { return this.src ? clamp(this.ctx.currentTime - this.t0, 0, this.dur) : 0; },
    halt() {
      cancelAnimationFrame(this.raf);
      if (!this.src) return;
      const s = this.src; this.src = null; this.key = null; s.onended = null;
      try { s.stop(); } catch (e) { /* already stopped */ }
    },
    stop() { this.halt(); this.emit(); },
    emit() { this.listeners.forEach((f) => f(this.key, this.position(), this.dur)); },
  };

  // ---------- Slides ----------
  const stage = $('#stage'), slides = $$('.slide');
  const notes = $('#notes'), help = $('#help');
  let index = 0;

  function fit() {
    const s = Math.min(innerWidth / 1280, innerHeight / 720);
    stage.style.transform = 'translate(-50%, -50%) scale(' + s + ')';
  }
  function go(i) {
    i = clamp(i, 0, slides.length - 1);
    if (i !== index) Player.stop();
    index = i;
    slides.forEach((el, k) => {
      el.classList.toggle('is-active', k === i);
      el.setAttribute('aria-hidden', k === i ? 'false' : 'true');
      el.inert = k !== i;
    });
    $('#counter').textContent = (i + 1) + ' / ' + slides.length;
    $('#progressBar').style.width = ((i + 1) / slides.length * 100) + '%';
    $('#sectionLabel').textContent = slides[i].dataset.section || '';
    notes.textContent = ($('.notes', slides[i]) || {}).textContent || '';
    history.replaceState(null, '', '#' + (i + 1));
    if (slides[i].id === 'demo') Demo.enter();
  }
  const next = () => go(index + 1), prev = () => go(index - 1);

  // ---------- Timer (10 minutes, warning at 2 minutes left) ----------
  const Timer = {
    el: $('#timer'), limit: 600, elapsed: 0, started: 0, id: 0,
    toggle() {
      if (this.id) { this.elapsed = this.now(); clearInterval(this.id); this.id = 0; }
      else { this.started = performance.now() - this.elapsed * 1000; this.id = setInterval(() => this.draw(), 250); }
      this.draw();
    },
    reset() { clearInterval(this.id); this.id = 0; this.elapsed = 0; this.draw(); },
    now() { return this.id ? (performance.now() - this.started) / 1000 : this.elapsed; },
    draw() {
      const left = this.limit - this.now(), a = Math.abs(Math.ceil(left));
      this.el.textContent = (left < 0 ? '+' : '') + Math.floor(a / 60) + ':' + String(a % 60).padStart(2, '0');
      this.el.classList.toggle('is-running', !!this.id);
      this.el.classList.toggle('is-warning', left <= 120 && left > 0);
      this.el.classList.toggle('is-over', left <= 0);
    },
  };
  Timer.el.addEventListener('click', (e) => (e.shiftKey ? Timer.reset() : Timer.toggle()));

  // ---------- Drawing helpers ----------
  const MAGMA = (function () {
    const stops = [[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]];
    const lut = new Uint8Array(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = i / 255 * (stops.length - 1), a = Math.floor(t), b = Math.min(stops.length - 1, a + 1), f = t - a;
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(stops[a][c] + (stops[b][c] - stops[a][c]) * f);
    }
    return lut;
  })();

  function sizeCanvas(canvas) {
    const w = canvas.offsetWidth * 2, h = canvas.offsetHeight * 2;
    if (w && h && (canvas.width !== w || canvas.height !== h)) { canvas.width = w; canvas.height = h; }
    return canvas.getContext('2d');
  }

  function drawWave(canvas, layers) {
    const ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    let peak = 0;
    for (const l of layers) for (let i = 0; i < l.x.length; i++) peak = Math.max(peak, Math.abs(l.x[i]));
    for (const l of layers) {
      const n = l.x.length;
      ctx.fillStyle = l.color;
      for (let px = 0; px < W; px++) {
        let lo = 1, hi = -1;
        for (let i = Math.floor(px * n / W), end = Math.floor((px + 1) * n / W); i < end; i++) {
          if (l.x[i] < lo) lo = l.x[i];
          if (l.x[i] > hi) hi = l.x[i];
        }
        const y0 = H / 2 - hi / peak * H * 0.48, y1 = H / 2 - lo / peak * H * 0.48;
        ctx.fillRect(px, y0, 1, Math.max(1, y1 - y0));
      }
    }
  }

  // Time–frequency image: frame k is centered at (k*hop - L/2) / fs.
  function drawTF(canvas, tf, value) {
    const ctx = sizeCanvas(canvas), W = canvas.width, H = canvas.height;
    const off = document.createElement('canvas');
    off.width = tf.F; off.height = tf.B;
    const octx = off.getContext('2d'), img = octx.createImageData(tf.F, tf.B);
    for (let k = 0; k < tf.F; k++) {
      for (let b = 0; b < tf.B; b++) {
        const c = clamp(Math.round(value(k, b) * 255), 0, 255) * 3, p = ((tf.B - 1 - b) * tf.F + k) * 4;
        img.data[p] = MAGMA[c]; img.data[p + 1] = MAGMA[c + 1]; img.data[p + 2] = MAGMA[c + 2]; img.data[p + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.fillStyle = '#000004'; ctx.fillRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = true;
    const perFrame = W * tf.hop / tf.n;
    ctx.drawImage(off, -(tf.L / 2 + tf.hop / 2) / tf.n * W, 0, tf.F * perFrame, H);
  }

  const DB_LO = -100, DB_HI = -25; // dB re full scale after window normalization
  function spectrogramValue(S) {
    const norm = 1 / (S.L / 2) / (S.L / 2);
    return (k, b) => (10 * Math.log10(S.power[k * S.B + b] * norm + 1e-14) - DB_LO) / (DB_HI - DB_LO);
  }

  function snrDb(x, s, lo, hi) {
    let sig = 0, err = 0;
    for (let i = lo; i <= hi; i++) { sig += s[i] * s[i]; err += (x[i] - s[i]) * (x[i] - s[i]); }
    return 10 * Math.log10(sig / err);
  }

  // ---------- Title and data slides ----------
  function drawStatic() {
    if (!CLIPS.bus) return;
    drawWave($('#titleWave'), [{ x: CLIPS.bus.noisy, color: '#b9c3cd' }, { x: CLIPS.bus.clean, color: '#1d5987' }]);
    $$('.clip-card').forEach((card) => {
      const c = CLIPS[card.dataset.clip];
      drawWave($('canvas', card), [{ x: c.noisy, color: '#c3ccd5' }, { x: c.clean, color: '#1d5987' }]);
    });
  }
  $$('.play').forEach((btn) => btn.addEventListener('click', () => {
    const key = 'data:' + btn.dataset.clip + ':' + btn.dataset.track, c = CLIPS[btn.dataset.clip];
    if (Player.key === key) Player.stop(); else Player.play(key, c[btn.dataset.track], c.fs, 0);
  }));
  Player.listeners.push((key) => $$('.play').forEach((btn) =>
    btn.classList.toggle('is-playing', key === 'data:' + btn.dataset.clip + ':' + btn.dataset.track)));

  // ---------- Tuning results ----------
  const METHOD_COLOR = { wiener: '#2a78d6', spectral_subtraction: '#eb6834', fir: '#1baf7a', identity: '#6b7785' };
  const METHOD_NAME = { wiener: 'Wiener gain', spectral_subtraction: 'Spectral subtraction', fir: 'FIR filter', identity: 'Noisy input' };

  function describe(c) {
    const o = c.options;
    if (c.method === 'identity') return 'Unprocessed noisy input';
    if (c.method === 'fir') {
      return Array.isArray(o.upperCutoffHz) || !o.upperCutoffHz ? 'High-pass at ' + o.cutoffHz + ' Hz'
        : 'Band-pass ' + o.cutoffHz + '–' + o.upperCutoffHz + ' Hz';
    }
    const parts = [c.method === 'wiener' ? 'gain floor ' + o.gainFloor : 'α = ' + o.alpha + ', β = ' + o.beta,
      'noise from quietest ' + Math.round(o.noiseFraction * 100) + '%'];
    if (o.noiseMode === 'pause_updated') parts.push('pause-updated (λ = ' + o.noiseLambda + ', margin ' + o.pauseThresholdDb + ' dB)');
    else parts.push('fixed estimate');
    return parts.join(' · ');
  }

  function drawTuning() {
    const fig = $('#tuneChart'), rows = TUNING.candidates;
    if (!fig || !rows.length) return;
    const W = 700, H = 452, m = { l: 74, r: 18, t: 46, b: 58 };
    const frozen = new Set(Object.values(TUNING.primary));
    const xs = rows.map((r) => r.dSnr), ys = rows.map((r) => r.dStoi * 1000);
    const x0 = Math.floor(Math.min(...xs)) - 0.5, x1 = Math.ceil(Math.max(...xs)) + 0.5;
    const y0 = Math.floor(Math.min(...ys) / 5) * 5 - 1, y1 = 8; // headroom for the labels
    const X = (v) => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
    const Y = (v) => m.t + (y1 - v) / (y1 - y0) * (H - m.t - m.b);
    const ns = 'http://www.w3.org/2000/svg';
    const el = (tag, attrs, parent, text) => {
      const e = document.createElementNS(ns, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      if (text !== undefined) e.textContent = text;
      (parent || svg).appendChild(e);
      return e;
    };
    fig.innerHTML = '';
    const svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' }, fig);
    el('title', {}, svg, 'Mean change in STOI against mean change in SNR for 41 tuning candidates');

    for (let v = Math.ceil(x0); v <= x1; v++) {
      el('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, class: v === 0 ? 'zero' : 'grid' });
      el('text', { x: X(v), y: H - m.b + 20, class: 'tick', 'text-anchor': 'middle' }, svg, (v > 0 ? '+' : '') + v);
    }
    for (let v = Math.ceil(y0 / 5) * 5; v <= 0; v += 5) {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: v === 0 ? 'zero' : 'grid' });
      el('text', { x: m.l - 10, y: Y(v) + 4, class: 'tick', 'text-anchor': 'end' }, svg, (v > 0 ? '+' : v < 0 ? '−' : '') + (Math.abs(v) / 1000).toFixed(3));
    }
    el('text', { x: (m.l + W - m.r) / 2, y: H - 12, class: 'axis', 'text-anchor': 'middle' }, svg, 'Mean SNR change (dB)  →  more noise removed');
    el('text', { x: 16, y: m.t - 22, class: 'axis' }, svg, '↑ Mean STOI change  (more intelligible)');

    const tip = document.createElement('div');
    tip.className = 'tune-tip'; tip.hidden = true;
    fig.appendChild(tip);
    const order = rows.slice().sort((a, b) => frozen.has(a.id) - frozen.has(b.id));
    for (const r of order) {
      const pause = r.options.noiseMode === 'pause_updated', isFrozen = frozen.has(r.id);
      const g = el('g', { class: 'pt' + (isFrozen ? ' is-frozen' : ''), tabindex: 0 });
      el('circle', { cx: X(r.dSnr), cy: Y(r.dStoi * 1000), r: 14, class: 'hit' }, g);
      if (isFrozen) el('circle', { cx: X(r.dSnr), cy: Y(r.dStoi * 1000), r: 11, class: 'halo' }, g);
      el('circle', { cx: X(r.dSnr), cy: Y(r.dStoi * 1000), r: 6.5,
        fill: pause ? '#fff' : METHOD_COLOR[r.method], stroke: METHOD_COLOR[r.method], class: 'dot' + (pause ? ' is-hollow' : '') }, g);
      const show = () => {
        tip.innerHTML = '<b>' + METHOD_NAME[r.method] + (isFrozen ? ' · frozen' : '') + '</b><span>' + describe(r) + '</span>' +
          '<span class="nums">ΔSNR ' + (r.dSnr >= 0 ? '+' : '−') + Math.abs(r.dSnr).toFixed(2) + ' dB · ΔSTOI ' +
          (r.dStoi >= 0 ? '+' : '−') + Math.abs(r.dStoi).toFixed(4) + ' · STOI ' + r.stoi.toFixed(4) + '</span>';
        tip.hidden = false;
        const px = X(r.dSnr) / W * fig.clientWidth, py = Y(r.dStoi * 1000) / H * fig.clientHeight;
        tip.style.left = Math.min(px, fig.clientWidth - 300) + 'px';
        tip.style.top = (py > fig.clientHeight * 0.55 ? py - tip.offsetHeight - 16 : py + 16) + 'px';
      };
      g.addEventListener('mouseenter', show); g.addEventListener('focus', show);
      g.addEventListener('mouseleave', () => { tip.hidden = true; }); g.addEventListener('blur', () => { tip.hidden = true; });
    }
    // Direct labels: frozen picks sit in the headroom above the cluster, with leader lines.
    const label = (id, text, dx, rowY, anchor) => {
      const r = rows.find((c) => c.id === id);
      if (!r) return;
      const x = X(r.dSnr), y = Y(r.dStoi * 1000);
      if (rowY !== undefined) {
        el('line', { x1: x, y1: y - 12, x2: x, y2: rowY + 6, class: 'leader' }, svg).parentNode.insertBefore(svg.lastChild, svg.querySelector('.pt'));
        el('text', { x: x + dx, y: rowY, class: 'pt-label', 'text-anchor': anchor }, svg, text);
      } else {
        el('text', { x: x + dx, y: y + 5, class: 'pt-label', 'text-anchor': anchor }, svg, text);
      }
    };
    label('identity', 'Noisy input', -16, undefined, 'end');
    label(TUNING.primary.spectral_subtraction, 'Subtraction', 4, m.t + 4, 'end');
    label(TUNING.primary.wiener, 'Wiener', 0, m.t + 30, 'middle');
    label(TUNING.primary.fir, 'FIR', -4, m.t + 4, 'start');
    el('text', { x: X(TUNING.candidates.find((c) => c.id === TUNING.primary.wiener).dSnr), y: m.t - 16, class: 'pt-caption', 'text-anchor': 'middle' }, svg, 'frozen settings');

    const legend = document.createElement('div');
    legend.className = 'tune-legend';
    legend.innerHTML = ['fir', 'spectral_subtraction', 'wiener'].map((k) =>
      '<span><i style="background:' + METHOD_COLOR[k] + '"></i>' + METHOD_NAME[k] + '</span>').join('') +
      '<span><i class="hollow"></i>Pause-updated noise estimate</span><span><i class="ring"></i>Frozen setting</span>';
    fig.appendChild(legend);
  }

  // ---------- Held-out results by condition ----------
  // Diverging fill: blue = better than the noisy input, red = worse, gray midpoint.
  function diverging(v, max) {
    const t = Math.min(1, Math.abs(v) / max), pole = v >= 0 ? [42, 120, 214] : [208, 69, 43], mid = [242, 244, 246];
    const rgb = mid.map((m, i) => Math.round(m + (pole[i] - m) * t));
    return { fill: 'rgb(' + rgb.join(',') + ')', ink: t > 0.55 ? '#fff' : '#142534' };
  }

  function drawConditions() {
    const box = $('#condTable'), cells = (window.PRESENTATION_EVALUATION || {}).cells || [];
    if (!box || !cells.length) return;
    const methods = [['fir', 'FIR filter'], ['spectral_subtraction', 'Spectral subtraction'], ['wiener', 'Wiener gain']];
    const noises = [['bus', 'Bus'], ['office', 'Office'], ['cafe', 'Café']], snrs = [2.5, 7.5, 12.5, 17.5];
    const find = (m, n, s) => cells.find((c) => c.method === m && c.noise === n && c.snr === s);
    const sign = (v, d) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
    let html = '<div class="cond-h cond-span" style="grid-column:3/7">SNR change (dB)</div>' +
      '<div class="cond-h cond-span" style="grid-column:8/12">STOI change (× 0.001)</div>' +
      '<div class="cond-h cond-in" style="grid-column:1/3">Input SNR (dB)</div>' +
      snrs.map((s, i) => '<div class="cond-h" style="grid-column:' + (3 + i) + '">' + s + '</div>').join('') +
      snrs.map((s, i) => '<div class="cond-h" style="grid-column:' + (8 + i) + '">' + s + '</div>').join('');
    methods.forEach(([m, label], mi) => {
      const row0 = 3 + mi * 3;
      html += '<div class="cond-m" style="grid-row:' + row0 + '/' + (row0 + 3) + '">' + label + '</div>';
      noises.forEach(([n, nl], ni) => {
        const r = row0 + ni;
        html += '<div class="cond-n" style="grid-row:' + r + '">' + nl + '</div>';
        snrs.forEach((s, si) => {
          const c = find(m, n, s);
          if (!c) return;
          const a = diverging(c.dSnr, 10), b = diverging(c.dStoi * 1000, 10);
          html += '<div class="cond-c" role="cell" style="grid-row:' + r + ';grid-column:' + (3 + si) + ';background:' + a.fill + ';color:' + a.ink + '">' + sign(c.dSnr, 2) + '</div>';
          html += '<div class="cond-c" role="cell" style="grid-row:' + r + ';grid-column:' + (8 + si) + ';background:' + b.fill + ';color:' + b.ink + '">' + sign(c.dStoi * 1000, 2) + '</div>';
        });
      });
    });
    box.innerHTML = html;
  }

  // ---------- Demo ----------
  // Demo defaults are the frozen training-set selections (config/frozen_denoisers.json).
  const TUNING = window.PRESENTATION_TUNING || { primary: {}, candidates: [] };
  const frozenOptions = (method) => {
    const row = TUNING.candidates.find((c) => c.id === TUNING.primary[method]);
    return row ? row.options : {};
  };
  const DEFAULTS = Object.assign(
    { cutoffHz: 100, upperCutoffHz: 7000, alpha: 1, beta: 0.01, gainFloor: 0.05, noiseFraction: 0.1,
      noiseMode: 'fixed', noiseLambda: 0.9, pauseThresholdDb: 3 },
    frozenOptions('fir'), frozenOptions('spectral_subtraction'), frozenOptions('wiener'));
  if (Array.isArray(DEFAULTS.upperCutoffHz)) DEFAULTS.upperCutoffHz = 0; // MATLAB [] means high-pass
  const SLIDERS = {
    cutoffHz: { label: 'Lower cutoff', min: 50, max: 1500, step: 10, fmt: (v) => v + ' Hz' },
    noiseFraction: { label: 'Noise frames used', min: 0.05, max: 0.5, step: 0.05, fmt: (v) => Math.round(v * 100) + '% quietest' },
    alpha: { label: 'Subtraction strength α', min: 0, max: 5, step: 0.1, fmt: (v) => v.toFixed(1) },
    beta: { label: 'Noise floor β', min: 0, max: 0.3, step: 0.005, fmt: (v) => v.toFixed(3) },
    gainFloor: { label: 'Gain floor', min: 0, max: 0.5, step: 0.01, fmt: (v) => v.toFixed(2) },
    noiseLambda: { label: 'Smoothing λ', min: 0.5, max: 0.99, step: 0.01, fmt: (v) => v.toFixed(2) },
    pauseThresholdDb: { label: 'Pause margin', min: 0, max: 10, step: 0.5, fmt: (v) => v.toFixed(1) + ' dB' },
  };
  const METHOD_LABEL = { identity: 'Unprocessed', fir: 'FIR output', spectral_subtraction: 'Spectral subtraction output', wiener: 'Wiener output' };
  const REASONS = { 'insufficient-energy-contrast': 'updates off: too little energy contrast', 'no-confirmed-pause': 'no confirmed pause: initial estimate kept' };

  const Demo = {
    ready: false, clip: 'bus', method: 'wiener', view: 'output', track: 'output',
    p: Object.assign({}, DEFAULTS), cache: {}, out: null, info: null,
    rows: {},

    enter() {
      if (!this.ready) { this.init(); this.ready = true; }
      this.renderClip(); this.update();
    },
    init() {
      $$('.plot-row[data-track]').forEach((row) => {
        const t = row.dataset.track;
        this.rows[t] = { row, canvas: $('canvas', row), plot: $('.plot', row) };
        $('.track', row).addEventListener('click', () => this.toggle(t));
        $('.plot', row).addEventListener('click', (e) => {
          if (e.target.closest('.view')) return;
          const r = e.currentTarget.getBoundingClientRect();
          this.play(t, (e.clientX - r.left) / r.width * this.duration());
        });
      });
      this.segment('#segClip', 'clip', () => { Player.stop(); this.renderClip(); this.update(); });
      this.segment('#segMethod', 'method', () => { this.buildParams(); this.update(); });
      this.segment('#segView', 'view', () => this.renderOutput());
      this.buildParams();
      Player.listeners.push((key, pos, dur) => {
        const playing = key && key.startsWith('demo:') ? key.slice(5) : null;
        for (const t in this.rows) this.rows[t].row.classList.toggle('is-playing', t === playing);
        const head = $('#playhead'), plot = this.rows.noisy.plot;
        head.style.display = playing ? 'block' : 'none';
        if (playing) head.style.left = (plot.offsetLeft + pos / dur * plot.offsetWidth - 1) + 'px';
      });
    },
    segment(sel, prop, onChange) {
      const buttons = $$(sel + ' button');
      const paint = () => buttons.forEach((b) => b.classList.toggle('is-on', b.dataset.v === this[prop]));
      buttons.forEach((b) => b.addEventListener('click', (e) => {
        e.stopPropagation(); this[prop] = b.dataset.v; paint(); onChange();
      }));
      paint();
    },
    buildParams() {
      const box = $('#params'), m = this.method, spectral = m === 'spectral_subtraction' || m === 'wiener';
      box.innerHTML = '';
      const slider = (name) => {
        const s = SLIDERS[name], el = document.createElement('div');
        el.className = 'param';
        el.innerHTML = '<label>' + s.label + '<output></output></label><input type="range" min="' + s.min + '" max="' + s.max + '" step="' + s.step + '">';
        const input = $('input', el), out = $('output', el);
        input.value = this.p[name]; out.textContent = s.fmt(this.p[name]);
        input.addEventListener('input', () => { this.p[name] = +input.value; out.textContent = s.fmt(this.p[name]); this.update(); this.syncReset(); });
        input.addEventListener('dblclick', () => { input.value = this.p[name] = DEFAULTS[name]; out.textContent = s.fmt(DEFAULTS[name]); this.update(); this.syncReset(); });
        box.appendChild(el);
      };
      if (m === 'identity') box.innerHTML = '<p class="param-note">The noisy recording, unchanged. This is the reference every method has to beat.</p>';
      const choice = (label, name, values) => {
        const el = document.createElement('div');
        el.className = 'param';
        el.innerHTML = '<label>' + label + '</label><div class="seg">' +
          values.map(([v, text]) => '<button data-v="' + v + '">' + text + '</button>').join('') + '</div>';
        box.appendChild(el);
        const buttons = $$('button', el);
        const paint = () => buttons.forEach((b) => b.classList.toggle('is-on', b.dataset.v === String(this.p[name])));
        buttons.forEach((b) => b.addEventListener('click', () => {
          this.p[name] = isNaN(+b.dataset.v) ? b.dataset.v : +b.dataset.v;
          paint(); this.buildParams(); this.update();
        }));
        paint();
      };
      if (m === 'fir') { slider('cutoffHz'); choice('Upper cutoff', 'upperCutoffHz', [[7000, '7 kHz'], [0, 'None']]); }
      if (m === 'spectral_subtraction') { slider('alpha'); slider('beta'); }
      if (m === 'wiener') slider('gainFloor');
      if (spectral) {
        slider('noiseFraction');
        choice('Noise estimate', 'noiseMode', [['fixed', 'Fixed'], ['pause_updated', 'Pause-updated']]);
        if (this.p.noiseMode === 'pause_updated') { slider('noiseLambda'); slider('pauseThresholdDb'); }
      }
      const changed = Object.keys(DEFAULTS).some((k) => this.p[k] !== DEFAULTS[k]);
      if (m !== 'identity' && changed) {
        const reset = document.createElement('button');
        reset.className = 'reset'; reset.textContent = 'Reset to frozen settings';
        reset.addEventListener('click', () => { this.p = Object.assign({}, DEFAULTS); this.buildParams(); this.update(); });
        box.appendChild(reset);
      }
    },
    // Show the reset button once a slider leaves its frozen value (without rebuilding mid-drag).
    syncReset() {
      const changed = Object.keys(DEFAULTS).some((k) => this.p[k] !== DEFAULTS[k]);
      if (changed !== !!$('#params .reset')) {
        const focused = document.activeElement, inputs = $$('#params input'), i = inputs.indexOf(focused);
        this.buildParams();
        if (i >= 0) $$('#params input')[i].focus();
      }
    },
    data() {
      const c = CLIPS[this.clip];
      if (!this.cache[this.clip]) {
        this.cache[this.clip] = { noisy: DSP.stft(c.noisy, c.fs), clean: DSP.stft(c.clean, c.fs) };
      }
      return Object.assign({ spec: this.cache[this.clip] }, c);
    },
    duration() { const c = CLIPS[this.clip]; return c.noisy.length / c.fs; },
    renderClip() {
      const d = this.data();
      drawTF(this.rows.noisy.canvas, d.spec.noisy, spectrogramValue(d.spec.noisy));
      drawTF(this.rows.clean.canvas, d.spec.clean, spectrogramValue(d.spec.clean));
    },
    update() {
      const d = this.data(), p = this.p, m = this.method;
      const noise = { noiseFraction: p.noiseFraction, noiseMode: p.noiseMode, noiseLambda: p.noiseLambda, pauseThresholdDb: p.pauseThresholdDb };
      const options = m === 'fir' ? { cutoffHz: p.cutoffHz, upperCutoffHz: p.upperCutoffHz }
        : m === 'spectral_subtraction' ? Object.assign({ alpha: p.alpha, beta: p.beta }, noise)
        : m === 'wiener' ? Object.assign({ gainFloor: p.gainFloor }, noise) : {};
      const r = DSP.denoise(d.noisy, d.fs, m, options);
      this.out = Float32Array.from(r.output); this.info = r.info;
      this.outSpec = DSP.stft(this.out, d.fs);
      $('#outLabel').textContent = METHOD_LABEL[m];

      // Score every method on the samples the FIR filter fully supports.
      const lo = 64, hi = d.noisy.length - 65;
      const a = snrDb(d.noisy, d.clean, lo, hi), b = snrDb(this.out, d.clean, lo, hi);
      $('#mIn').textContent = a.toFixed(1) + ' dB';
      $('#mOut').textContent = b.toFixed(1) + ' dB';
      $('#mDelta').textContent = (b - a >= 0 ? '+' : '−') + Math.abs(b - a).toFixed(1) + ' dB';

      this.renderOutput(); this.renderStrip();
      if (Player.key === 'demo:output') this.play('output', Player.position());
    },
    renderOutput() {
      const S = this.outSpec, info = this.info, canvas = this.rows.output.canvas;
      if (this.view === 'output') return drawTF(canvas, S, spectrogramValue(S));
      const gain = this.method === 'identity' ? () => 1
        : this.method === 'fir' ? (k, b) => info.response[b]
        : (k, b) => info.gain[k * S.B + b];
      drawTF(canvas, S, (k, b) => Math.min(1, gain(k, b)));
    },
    renderStrip() {
      const canvas = $('#strip'), ctx = sizeCanvas(canvas), W = canvas.width, H = canvas.height;
      const d = this.data(), S = d.spec.noisy, tr = this.info.noiseTracking, note = $('#trackNote');
      const axis = 30, plotH = H - axis, dur = this.duration();
      ctx.clearRect(0, 0, W, H);
      ctx.font = '600 20px -apple-system, "Segoe UI", Arial, sans-serif';
      ctx.textBaseline = 'middle';
      const xOf = (k) => (k * S.hop - S.L / 2) / S.n * W;
      const db = (e) => 10 * Math.log10(e + 1e-12);
      let lo = Infinity, hi = -Infinity;
      for (let k = 0; k < S.F; k++) {
        if (k * S.hop - S.L < 0 || k * S.hop > S.n) continue;
        lo = Math.min(lo, db(S.frameEnergy[k])); hi = Math.max(hi, db(S.frameEnergy[k]));
      }
      lo -= 3; hi += 3;
      const yOf = (v) => plotH - (v - lo) / (hi - lo) * plotH;
      const label = (text, x, y, align, color) => {
        const w = ctx.measureText(text).width + 12;
        ctx.fillStyle = 'rgba(240, 246, 250, .86)';
        ctx.fillRect(align === 'right' ? x - w + 6 : x - 6, y - 13, w, 26);
        ctx.fillStyle = color; ctx.textAlign = align;
        ctx.fillText(text, x, y);
      };
      if (tr) {
        for (let k = 0; k < S.F; k++) {
          if (!tr.pauseCandidates[k]) continue;
          ctx.fillStyle = tr.updateMask[k] ? '#d9822b' : 'rgba(217, 130, 43, .22)';
          const x0 = Math.round((xOf(k) + xOf(k - 1)) / 2), x1 = Math.round((xOf(k) + xOf(k + 1)) / 2);
          ctx.fillRect(x0, 0, x1 - x0, plotH);
        }
        const y = yOf(db(tr.pauseEnergyThreshold));
        ctx.setLineDash([10, 8]); ctx.strokeStyle = '#8a4b0c'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); ctx.setLineDash([]);
        label('pause threshold', W - 10, y < 40 ? y + 16 : y - 16, 'right', '#8a4b0c');
      }
      ctx.strokeStyle = '#142534'; ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let k = 0, first = true; k < S.F; k++) {
        if (k * S.hop - S.L < 0 || k * S.hop > S.n) continue;
        const x = xOf(k), y = yOf(db(S.frameEnergy[k]));
        if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
      label('frame energy of noisy input', 10, 16, 'left', '#142534');
      ctx.fillStyle = '#7d8e9c'; ctx.textAlign = 'center';
      for (let t = 0.5; t < dur; t += 0.5) {
        const x = t / dur * W;
        ctx.fillRect(x, plotH, 1.5, 6);
        ctx.fillText(t.toFixed(1) + ' s', x, plotH + 18);
      }
      if (!tr) note.textContent = this.method === 'fir' ? 'Not used: one fixed filter for the whole clip.' : 'Not used.';
      else if (tr.fallbackReason === 'fixed-mode') note.textContent = 'Fixed. Shaded: frames that would count as pauses.';
      else if (tr.fallbackReason !== 'none') note.textContent = REASONS[tr.fallbackReason];
      else note.textContent = 'Updated on ' + tr.updateCount + ' frames (solid orange).';
    },
    samples(track) { const c = CLIPS[this.clip]; return track === 'output' ? this.out : c[track]; },
    play(track, offset) {
      this.track = track;
      Player.play('demo:' + track, this.samples(track), CLIPS[this.clip].fs, offset || 0);
    },
    toggle(track) {
      if (Player.key === 'demo:' + track) return Player.stop();
      // Switching tracks mid-playback keeps the position for A/B listening.
      this.play(track, Player.key && Player.key.startsWith('demo:') ? Player.position() : 0);
    },
  };

  // ---------- Input ----------
  document.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = e.target.tagName;
    if (e.key === 'Escape') { if (document.activeElement) document.activeElement.blur(); help.hidden = true; return; }
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    const onDemo = slides[index].id === 'demo';
    const button = tag === 'BUTTON' || tag === 'A';
    switch (e.key) {
      case 'ArrowRight': case 'ArrowDown': case 'PageDown': next(); break;
      case 'ArrowLeft': case 'ArrowUp': case 'PageUp': prev(); break;
      case ' ': case 'Enter':
        if (button) return;
        if (onDemo && e.key === ' ') Demo.toggle(Demo.track); else next();
        break;
      case 'Home': go(0); break;
      case 'End': go(slides.length - 1); break;
      case 'f': case 'F':
        if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen();
        break;
      case 'n': case 'N': notes.hidden = !notes.hidden; break;
      case 't': case 'T': if (e.shiftKey) Timer.reset(); else Timer.toggle(); break;
      case 'd': case 'D': go(slides.findIndex((s) => s.id === 'demo')); break;
      case '?': help.hidden = !help.hidden; break;
      case '1': case '2': case '3':
        if (onDemo) Demo.toggle(['noisy', 'output', 'clean'][+e.key - 1]);
        break;
      default: return;
    }
    e.preventDefault();
  });
  $('#next').addEventListener('click', next);
  $('#prev').addEventListener('click', prev);

  let touchX = null;
  document.addEventListener('touchstart', (e) => { touchX = e.target.closest('input, .plot') ? null : e.touches[0].clientX; }, { passive: true });
  document.addEventListener('touchend', (e) => {
    if (touchX === null) return;
    const dx = e.changedTouches[0].clientX - touchX;
    if (Math.abs(dx) > 60) (dx < 0 ? next : prev)();
    touchX = null;
  });

  addEventListener('resize', fit);
  fit(); drawStatic(); drawTuning(); drawConditions(); Timer.draw();
  go((parseInt(location.hash.slice(1), 10) || 1) - 1);
  window.Presentation = { go, Demo, Player };
})();
