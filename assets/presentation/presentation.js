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

  // ---------- Demo ----------
  const DEFAULTS = { cutoffHz: 200, alpha: 1, beta: 0.01, gainFloor: 0.05, noiseMode: 'fixed', noiseLambda: 0.9, pauseThresholdDb: 3 };
  const SLIDERS = {
    cutoffHz: { label: 'Cutoff', min: 50, max: 1500, step: 10, fmt: (v) => v + ' Hz' },
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
        input.addEventListener('input', () => { this.p[name] = +input.value; out.textContent = s.fmt(this.p[name]); this.update(); });
        input.addEventListener('dblclick', () => { input.value = this.p[name] = DEFAULTS[name]; out.textContent = s.fmt(DEFAULTS[name]); this.update(); });
        box.appendChild(el);
      };
      if (m === 'identity') box.innerHTML = '<p class="param-note">The noisy recording, unchanged. This is the reference every method has to beat.</p>';
      if (m === 'fir') slider('cutoffHz');
      if (m === 'spectral_subtraction') { slider('alpha'); slider('beta'); }
      if (m === 'wiener') slider('gainFloor');
      if (spectral) {
        const el = document.createElement('div');
        el.className = 'param';
        el.innerHTML = '<label>Noise estimate</label><div class="seg" id="segNoise"><button data-v="fixed">Fixed</button><button data-v="pause_updated">Pause-updated</button></div>';
        box.appendChild(el);
        const buttons = $$('button', el);
        const paint = () => buttons.forEach((b) => b.classList.toggle('is-on', b.dataset.v === this.p.noiseMode));
        buttons.forEach((b) => b.addEventListener('click', () => { this.p.noiseMode = b.dataset.v; paint(); this.buildParams(); this.update(); }));
        paint();
        if (this.p.noiseMode === 'pause_updated') { slider('noiseLambda'); slider('pauseThresholdDb'); }
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
      const options = m === 'fir' ? { cutoffHz: p.cutoffHz }
        : m === 'spectral_subtraction' ? { alpha: p.alpha, beta: p.beta, noiseMode: p.noiseMode, noiseLambda: p.noiseLambda, pauseThresholdDb: p.pauseThresholdDb }
        : m === 'wiener' ? { gainFloor: p.gainFloor, noiseMode: p.noiseMode, noiseLambda: p.noiseLambda, pauseThresholdDb: p.pauseThresholdDb } : {};
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
  fit(); drawStatic(); Timer.draw();
  go((parseInt(location.hash.slice(1), 10) || 1) - 1);
  window.Presentation = { go, Demo, Player };
})();
