#!/usr/bin/env python3
"""Figures for the presentation (assets/figures/slides/): spectrogram PNGs and SVG charts.

All numbers come from our own code and result files:
  exploratory clips (assets/audio), dsp.py, the trained model (learned_alpha.default_checkpoint()),
  reports/learned_alpha/{tuning_alpha_per_file.csv, test_per_file.csv}.
  .venv/bin/python code/make_slide_figures.py
"""
import csv
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from scipy.signal import firwin, freqz, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsp
import learned_alpha
from make_listening_preview import MAGMA, lut, write_png

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'assets'/'figures'/'slides'
NAVY, MAIZE, GREY, INK, SOFT, GRID = '#00274C', '#FFCB05', '#9aa5b1', '#1d1d1f', '#5f6b76', '#e6e9ec'
COLORS = {'noisy': '#b8c0c8', 'fir': '#7f8c99', 'dd098': '#d08a2e', 'ddtuned': '#c25b3f', 'learned': NAVY, 'clean': INK}
CLIPS = {'bus': 'p257_004', 'office': 'p257_017', 'cafe': 'p257_009'}
FONT = 'font-family="Arial, Helvetica, sans-serif"'


def load(clip):
    s, _ = sf.read(ROOT/'assets'/'audio'/'clean'/f'{CLIPS[clip]}.wav')
    y, _ = sf.read(ROOT/'assets'/'audio'/'noisy'/f'{CLIPS[clip]}.wav')
    return resample_poly(s, 1, 3), resample_poly(y, 1, 3)


def power(x):
    Y, _ = dsp.analysis(x)
    return np.abs(Y)**2


def spec_png(name, x, lo=-100, hi=-25):
    db = 10*np.log10(power(x)/(dsp.WINDOW/2)**2+1e-14)
    write_png(OUT/name, ((db-lo)/(hi-lo)).T[::-1], lut(MAGMA))


def mask_png(name, g):
    blues = lut([(255, 255, 255), (198, 219, 239), (107, 174, 214), (33, 113, 181), (0, 39, 76)])
    write_png(OUT/name, np.asarray(g).T[::-1], blues)


# ---------- minimal SVG charts ----------
def svg(w, h, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{label}"><g {FONT}>{body}</g></svg>')


class Axes:
    def __init__(self, w, h, x0, x1, y0, y1, xlog=False, m=(64, 18, 16, 50)):
        self.w, self.h, self.xlog = w, h, xlog
        self.l, self.r, self.t, self.b = m
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def X(self, v):
        f = (np.log10(v)-np.log10(self.x0))/(np.log10(self.x1)-np.log10(self.x0)) if self.xlog else (v-self.x0)/(self.x1-self.x0)
        return self.l+f*(self.w-self.l-self.r)

    def Y(self, v):
        return self.t+(self.y1-v)/(self.y1-self.y0)*(self.h-self.t-self.b)

    def frame(self, xticks, yticks, xlabel, ylabel):
        out = []
        for v, lab in xticks:
            out.append(f'<line x1="{self.X(v):.1f}" x2="{self.X(v):.1f}" y1="{self.t}" y2="{self.h-self.b}" stroke="{GRID}"/>'
                       f'<text x="{self.X(v):.1f}" y="{self.h-self.b+20}" text-anchor="middle" font-size="14" fill="{SOFT}">{lab}</text>')
        for v, lab in yticks:
            out.append(f'<line x1="{self.l}" x2="{self.w-self.r}" y1="{self.Y(v):.1f}" y2="{self.Y(v):.1f}" stroke="{GRID}"/>'
                       f'<text x="{self.l-8}" y="{self.Y(v)+5:.1f}" text-anchor="end" font-size="14" fill="{SOFT}">{lab}</text>')
        out.append(f'<line x1="{self.l}" x2="{self.w-self.r}" y1="{self.h-self.b}" y2="{self.h-self.b}" stroke="{INK}"/>'
                   f'<line x1="{self.l}" x2="{self.l}" y1="{self.t}" y2="{self.h-self.b}" stroke="{INK}"/>')
        out.append(f'<text x="{(self.l+self.w-self.r)/2:.0f}" y="{self.h-8}" text-anchor="middle" font-size="15" fill="{INK}">{xlabel}</text>')
        out.append(f'<text transform="translate(16 {(self.t+self.h-self.b)/2:.0f}) rotate(-90)" text-anchor="middle" font-size="15" fill="{INK}">{ylabel}</text>')
        return ''.join(out)

    def line(self, xs, ys, color, width=2.4, dash=''):
        pts = ' '.join(f'{self.X(a):.1f},{self.Y(b):.1f}' for a, b in zip(xs, ys))
        d = f' stroke-dasharray="{dash}"' if dash else ''
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{d}/>'


def legend(items, x, y, step=22):
    out = []
    for k, (label, color, dash) in enumerate(items):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        out.append(f'<line x1="{x}" x2="{x+26}" y1="{y+k*step}" y2="{y+k*step}" stroke="{color}" stroke-width="3"{d}/>'
                   f'<text x="{x+34}" y="{y+k*step+5}" font-size="14" fill="{INK}">{label}</text>')
    return ''.join(out)


def save(name, text):
    (OUT/name).write_text(text, encoding='utf-8')


# ---------- figures ----------
def fig_intro():
    s, y = load('bus')
    spec_png('intro_clean.png', s)
    spec_png('intro_noise.png', y-s)
    spec_png('intro_noisy.png', y)


def fig_fir_response():
    b = firwin(513, [50, 7000], pass_zero=False, window='hamming', scale=True, fs=16000)
    f = np.logspace(np.log10(20), np.log10(8000), 500)
    _, H = freqz(b, worN=f, fs=16000)
    db = np.clip(20*np.log10(np.abs(H)+1e-12), -60, 5)
    ax = Axes(620, 300, 20, 8000, -60, 5, xlog=True)
    body = (f'<rect x="{ax.X(50):.1f}" y="{ax.t}" width="{ax.X(7000)-ax.X(50):.1f}" height="{ax.h-ax.t-ax.b}" fill="{MAIZE}" opacity=".18"/>'
            + ax.frame([(v, l) for v, l in [(20, '20'), (50, '50'), (100, '100'), (300, '300'), (1000, '1k'), (3000, '3k'), (7000, '7k')]],
                       [(v, str(v)) for v in [0, -20, -40, -60]], 'Frequency (Hz)', 'Gain (dB)')
            + ax.line(f, db, NAVY, 2.8)
            + f'<text x="{(ax.X(50)+ax.X(7000))/2:.0f}" y="{ax.Y(-8):.0f}" text-anchor="middle" font-size="15" fill="{INK}">speech band kept (50–7,000 Hz)</text>')
    save('fir_response.svg', svg(620, 300, body, 'FIR band-pass magnitude response'))


def avg_spectrum_db(x):
    P = power(x)
    return 10*np.log10(P.mean(0)/(dsp.WINDOW/2)**2+1e-14)


def fig_fir_limit():
    f = np.arange(dsp.NFFT//2+1)*16000/dsp.NFFT
    keep = f >= 31.25
    for clip in ['bus', 'cafe']:
        s, y = load(clip)
        series = [('Noisy input', avg_spectrum_db(y), COLORS['noisy'], ''), ('After FIR', avg_spectrum_db(dsp.fir_bandpass(y)), NAVY, ''),
                  ('Clean speech', avg_spectrum_db(s), INK, '5 4')]
        ax = Axes(520, 300, 31.25, 8000, -100, -30, xlog=True)
        body = ax.frame([(v, l) for v, l in [(50, '50'), (100, '100'), (300, '300'), (1000, '1k'), (3000, '3k'), (7000, '7k')]],
                        [(v, str(v)) for v in [-40, -60, -80, -100]], 'Frequency (Hz)', 'Power (dB)')
        for _, d, c, dash in series:
            body += ax.line(f[keep], np.clip(d[keep], -100, -30), c, 2.4, dash)
        body += legend([(n, c, d) for n, _, c, d in series], ax.X(45), ax.Y(-84))
        save(f'fir_limit_{clip}.svg', svg(520, 300, body, f'Average spectrum, {clip}: noisy, after FIR, clean'))


def fig_wiener_pipeline():
    s, y = load('bus')
    out, g = dsp.wiener_dd(y, alpha=0.98, return_gains=True)
    spec_png('wiener_in.png', y)
    mask_png('wiener_gain.png', g)
    spec_png('wiener_out.png', out)


def fig_gain_curve():
    xi_db = np.linspace(-25, 25, 300)
    xi = 10**(xi_db/10)
    ax = Axes(440, 280, -25, 25, 0, 1, m=(56, 16, 16, 50))
    body = (ax.frame([(v, str(v)) for v in [-20, -10, 0, 10, 20]], [(v, f'{v:g}') for v in [0, 0.5, 1]],
                     'ξ: speech-to-noise ratio (dB)', 'Gain G')
            + ax.line(xi_db, xi/(1+xi), NAVY, 3)
            + f'<text x="{ax.X(-24):.0f}" y="{ax.Y(0.3):.0f}" font-size="15" fill="{INK}">noise → small G</text>'
            + f'<text x="{ax.X(24):.0f}" y="{ax.Y(0.62):.0f}" text-anchor="end" font-size="15" fill="{INK}">speech → G ≈ 1</text>')
    save('wiener_gain_curve.svg', svg(440, 280, body, 'Wiener gain G versus speech-to-noise ratio'))


def fig_alpha_tradeoff():
    s, y = load('bus')
    S, V = power(s), power(y-s)
    P = power(y)
    N = dsp.noise_psd(y, 0.1)
    k = 13  # 406 Hz bin, inside the voiced-speech range
    true_db = 10*np.log10(S[:, k]/(V[:, k]+1e-12)+1e-12)
    est = {}
    for a in [0.98, 0.5]:
        g = dsp.dd_gains(P, N, a, -25)[:, k]
        est[a] = 10*np.log10(g/(1-g+1e-12)+1e-12)
    t = (np.arange(P.shape[0])*dsp.HOP-dsp.WINDOW)/16000
    t0, t1 = 0.55, 1.65
    sel = (t >= t0) & (t <= t1)
    ax = Axes(760, 340, t0, t1, -30, 30, m=(60, 20, 70, 50))
    body = ax.frame([(v, f'{v:.1f}') for v in np.arange(0.6, 1.65, 0.2)], [(v, str(v)) for v in [-30, -15, 0, 15, 30]],
                    'Time (s)', 'ξ at 406 Hz (dB)')
    body += ax.line(t[sel], np.clip(true_db[sel], -30, 30), '#c8ced4', 7)
    body += ax.line(t[sel], np.clip(est[0.5][sel], -30, 30), COLORS['ddtuned'], 2)
    body += ax.line(t[sel], np.clip(est[0.98][sel], -30, 30), NAVY, 2.8)
    items = [('True ξ (from clean speech)', '#c8ced4'), ('α = 0.5', COLORS['ddtuned']), ('α = 0.98', NAVY)]
    x = ax.l
    for label, color in items:
        body += (f'<line x1="{x}" x2="{x+26}" y1="22" y2="22" stroke="{color}" stroke-width="4"/>'
                 f'<text x="{x+34}" y="27" font-size="15" fill="{INK}">{label}</text>')
        x += 46+8.2*len(label)
    idx = np.where(sel)[0]
    inner = [i for i in idx if t0+0.08 <= t[i] <= t1-0.12]
    pause_frames = [i for i in inner if true_db[i] < -10]
    spike = max(pause_frames, key=lambda i: est[0.5][i]-est[0.98][i])
    onsets = [i for i in inner if i >= 2 and true_db[i] > 10 and true_db[i-2] < -5]
    lag = max(onsets, key=lambda i: est[0.5][i]-est[0.98][i])
    def region(i0, i1, fill, text):
        x0, x1 = ax.X(t[i0]-0.005), ax.X(t[i1]+0.005)
        cx = min(max((x0+x1)/2, ax.l+110), ax.w-ax.r-110)
        return (f'<rect x="{x0:.1f}" y="{ax.t}" width="{x1-x0:.1f}" height="{ax.h-ax.t-ax.b}" fill="{fill}" opacity=".16"/>'
                f'<line x1="{(x0+x1)/2:.1f}" x2="{cx:.1f}" y1="{ax.t}" y2="{ax.t-8}" stroke="{fill}" stroke-width="1.5"/>'
                f'<text x="{cx:.1f}" y="{ax.t-12}" text-anchor="middle" font-size="14" font-weight="700" fill="{fill}">{text}</text>')
    a0 = spike
    while a0-1 in pause_frames and a0-1 >= idx[0]:
        a0 -= 1
    a1 = spike
    while a1+1 in pause_frames:
        a1 += 1
    body = region(a0, a1, COLORS['ddtuned'], 'pause: α = 0.5 flags noise as speech') + body
    body = region(lag-2, lag+1, NAVY, 'word onset: α = 0.98 reacts late') + body
    save('alpha_tradeoff.svg', svg(760, 330, body, 'True and estimated speech-to-noise ratio over time for two alpha values'))


def fig_best_alpha():
    noise = {r['File']: r['Noise'] for r in csv.DictReader(open(ROOT/'data'/'tuning_manifest.csv'))}
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(ROOT/'reports'/'learned_alpha'/'tuning_alpha_per_file.csv')):
        if r['system'] != 'noisy':
            acc[noise[r['name']]][float(r['system'][3:])].append(float(r['pesq']))
    best = sorted(((n, max(v, key=lambda a: np.mean(v[a]))) for n, v in acc.items()), key=lambda x: x[1])
    w, h, l, r, t, bm = 520, 300, 96, 40, 12, 40
    x = lambda a: l+(a-0.7)/(1.0-0.7)*(w-l-r)
    bh = (h-t-bm)/len(best)
    body = ''
    for v in [0.7, 0.8, 0.9, 1.0]:
        body += (f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{t}" y2="{h-bm}" stroke="{GRID}"/>'
                 f'<text x="{x(v):.1f}" y="{h-bm+20}" text-anchor="middle" font-size="14" fill="{SOFT}">{v:.1f}</text>')
    body += f'<line x1="{x(0.98):.1f}" x2="{x(0.98):.1f}" y1="{t}" y2="{h-bm}" stroke="{INK}" stroke-dasharray="5 4"/>'
    body += f'<text x="{x(0.98)-4:.1f}" y="{t+12}" text-anchor="end" font-size="13" fill="{INK}">textbook 0.98</text>'
    for i, (n, a) in enumerate(best):
        yy = t+i*bh
        body += (f'<rect x="{x(0.7):.1f}" y="{yy+4:.1f}" width="{x(a)-x(0.7):.1f}" height="{bh-8:.1f}" fill="{NAVY}" opacity=".85"/>'
                 f'<text x="{l-8}" y="{yy+bh/2+5:.1f}" text-anchor="end" font-size="14" fill="{INK}">{n}</text>'
                 f'<text x="{x(a)+6:.1f}" y="{yy+bh/2+5:.1f}" font-size="13" fill="{SOFT}">{a:g}</text>')
    body += f'<text x="{(l+w-r)/2:.0f}" y="{h-6}" text-anchor="middle" font-size="15" fill="{INK}">best α by PESQ (400 training pairs)</text>'
    save('best_alpha.svg', svg(w, h, body, 'Best alpha per training noise type'))


def fig_learned():
    model = learned_alpha.load(learned_alpha.default_checkpoint())
    s, y = load('cafe')
    out, alpha = learned_alpha.denoise(model, y, return_alpha=True)
    spec_png('learned_clean.png', s)
    blues = lut([(247, 251, 255), (198, 219, 239), (107, 174, 214), (33, 113, 181), (0, 39, 76)])
    write_png(OUT/'learned_alpha.png', alpha.T[::-1], blues)


def fig_results():
    rows = list(csv.DictReader(open(ROOT/'reports'/'learned_alpha'/'test_per_file.csv')))
    key = {'noisy': 'noisy', 'fir': 'fir', 'dd:0.98': 'dd098', 'dd:0.9': 'ddtuned', 'learned:outputs/learned_alpha/best.pt': 'learned'}
    names = {'noisy': 'Noisy input', 'fir': 'FIR', 'dd098': 'Wiener α=0.98', 'ddtuned': 'Wiener α=0.9', 'learned': 'Ours'}
    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[key[r['system']]][r['snr_db']].append((float(r['pesq']), float(r['stoi'])))
    order = ['noisy', 'fir', 'dd098', 'ddtuned', 'learned']
    mean = {k: np.mean([v for vs in data[k].values() for v in vs], axis=0) for k in order}

    def bars(metric, idx, lo, hi, fmt, fname, title):
        w, h, l, r, t, bm = 460, 290, 122, 56, 34, 20
        x = lambda v: l+(v-lo)/(hi-lo)*(w-l-r)
        bh = (h-t-bm)/len(order)
        body = f'<text x="{l}" y="20" font-size="17" font-weight="700" fill="{INK}">{title}</text>'
        for i, k in enumerate(order):
            v = mean[k][idx]
            yy = t+i*bh
            body += (f'<rect x="{l}" y="{yy+5:.1f}" width="{x(v)-l:.1f}" height="{bh-10:.1f}" fill="{COLORS[k]}"/>'
                     f'<text x="{l-8}" y="{yy+bh/2+5:.1f}" text-anchor="end" font-size="14" fill="{INK}"'
                     f'{" font-weight=\"700\"" if k == "learned" else ""}>{names[k]}</text>'
                     f'<text x="{x(v)+6:.1f}" y="{yy+bh/2+5:.1f}" font-size="14" fill="{INK}"'
                     f'{" font-weight=\"700\"" if k == "learned" else ""}>{fmt.format(v)}</text>')
        if metric == 'stoi':
            nx = x(mean['noisy'][idx])
            body += f'<line x1="{nx:.1f}" x2="{nx:.1f}" y1="{t}" y2="{h-bm}" stroke="{INK}" stroke-dasharray="4 4"/>'
        save(fname, svg(w, h, body, title))

    bars('pesq', 0, 1.8, 2.35, '{:.2f}', 'results_pesq.svg', 'PESQ (quality) ↑')
    bars('stoi', 1, 0.89, 0.932, '{:.3f}', 'results_stoi.svg', 'STOI (intelligibility) ↑')

    snrs = ['2.5', '7.5', '12.5', '17.5']
    ax = Axes(640, 390, 1.5, 18.5, 0, 0.42, m=(62, 170, 16, 50))
    body = ax.frame([(float(v), v) for v in snrs], [(v, f'+{v:.1f}' if v else '0') for v in [0, 0.1, 0.2, 0.3, 0.4]],
                    'Input SNR (dB)', 'PESQ gain over noisy')
    for k in ['fir', 'dd098', 'ddtuned', 'learned']:
        gain = [np.mean([p for p, _ in data[k][v]])-np.mean([p for p, _ in data['noisy'][v]]) for v in snrs]
        body += ax.line([float(v) for v in snrs], gain, COLORS[k], 3.2 if k == 'learned' else 2.2)
        body += ''.join(f'<circle cx="{ax.X(float(v)):.1f}" cy="{ax.Y(g):.1f}" r="4" fill="{COLORS[k]}"/>' for v, g in zip(snrs, gain))
    body += legend([(names[k], COLORS[k], '') for k in ['learned', 'ddtuned', 'dd098', 'fir']], ax.w-ax.r+18, 40)
    save('results_snr.svg', svg(640, 390, body, 'PESQ gain over the noisy input by SNR'))


def fig_test_noises():
    index = list(csv.DictReader(open(ROOT/'data'/'cache16k'/'test_index.csv')))
    for noise in ['bus', 'cafe', 'living', 'office', 'psquare']:
        row = next(r for r in index if r['noise'] == noise and r['snr_db'] == '2.5')
        with np.load(ROOT/'data'/'cache16k'/'test'/f"{row['name']}.npz") as z:
            v = z['noisy'].astype(float)-z['clean'].astype(float)
        spec_png(f'test_noise_{noise}.png', v[:int(2.5*16000)])


def fig_noise_types():
    import json
    rows = json.loads((ROOT/'reports'/'learned_alpha'/'noise_type_analysis_validation.json').read_text())
    kind = {'station': 'steady', 'ssn': 'steady', 'traffic': 'steady', 'car': 'changing', 'kitchen': 'changing',
            'metro': 'changing', 'cafeteria': 'speech', 'restaurant': 'speech', 'babble': 'speech', 'meeting': 'speech'}
    color = {'steady': NAVY, 'changing': '#d08a2e', 'speech': '#c0392b'}
    label = {'steady': 'Steady noise', 'changing': 'Changing noise', 'speech': 'Background talkers'}
    rows.sort(key=lambda r: r['fixed N_pause_db'])
    w, h, l, r, t, bm = 600, 360, 100, 60, 44, 40
    x = lambda v: l+v/30*(w-l-r)
    bh = (h-t-bm)/len(rows)
    body = ''
    for v in [0, 10, 20, 30]:
        body += (f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{t}" y2="{h-bm}" stroke="{GRID}"/>'
                 f'<text x="{x(v):.1f}" y="{h-bm+18}" text-anchor="middle" font-size="14" fill="{SOFT}">{v}</text>')
    for i, row in enumerate(rows):
        v = -row['fixed N_pause_db']
        yy = t+i*bh
        c = color[kind[row['noise']]]
        body += (f'<rect x="{l}" y="{yy+4:.1f}" width="{x(v)-l:.1f}" height="{bh-8:.1f}" fill="{c}"/>'
                 f'<text x="{l-8}" y="{yy+bh/2+5:.1f}" text-anchor="end" font-size="14" fill="{INK}">{row["noise"]}</text>'
                 f'<text x="{x(v)+6:.1f}" y="{yy+bh/2+5:.1f}" font-size="13" fill="{SOFT}">{v:.0f} dB</text>')
    lx = l
    for k in ['steady', 'changing', 'speech']:
        body += (f'<rect x="{lx}" y="12" width="16" height="12" fill="{color[k]}"/>'
                 f'<text x="{lx+22}" y="23" font-size="14" fill="{INK}">{label[k]}</text>')
        lx += 46+7.6*len(label[k])
    body += f'<text x="{(l+w-r)/2:.0f}" y="{h-6}" text-anchor="middle" font-size="15" fill="{INK}">Noise removed in speech pauses (dB)</text>'
    save('noise_types.svg', svg(w, h, body, 'Noise removed in speech pauses by noise type'))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for f in [fig_intro, fig_fir_response, fig_fir_limit, fig_wiener_pipeline, fig_gain_curve,
              fig_alpha_tradeoff, fig_best_alpha, fig_learned, fig_results, fig_test_noises, fig_noise_types]:
        f()
        print('ok', f.__name__)
