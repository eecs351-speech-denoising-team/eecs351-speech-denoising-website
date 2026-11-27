#!/usr/bin/env python3
"""Listening preview: the three exploratory clips through each Wiener variant.

Writes outputs/listening_preview/{index.html, *.wav, *.png}. One export gain per clip
(shared by every version, so loudness differences are real). The clips are neither
training nor test data.
  .venv/bin/python code/make_listening_preview.py --tuned-alpha 0.9
"""
import argparse
import html
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import zlib

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsp
import learned_alpha
import quality_metrics as qm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'/'listening_preview'
CLIPS = [('bus', 'p257_004'), ('office', 'p257_017'), ('cafe', 'p257_009')]
MAGMA = [(0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129), (181, 54, 122), (229, 80, 100),
         (251, 135, 97), (254, 194, 135), (252, 253, 191)]
BLUES = [(247, 251, 255), (198, 219, 239), (107, 174, 214), (33, 113, 181), (8, 48, 107)]


def lut(stops):
    stops = np.array(stops, float)
    x = np.linspace(0, len(stops)-1, 256)
    a = np.floor(x).astype(int).clip(0, len(stops)-2)
    f = (x-a)[:, None]
    return (stops[a]*(1-f)+stops[a+1]*f).round().astype(np.uint8)


def write_png(path, values, colors, scale_x=2):
    """values [rows, cols] in 0..1, row 0 at the top."""
    rgb = colors[(np.clip(values, 0, 1)*255).astype(int)]
    rgb = np.repeat(rgb, scale_x, axis=1)
    h, w, _ = rgb.shape
    raw = b''.join(b'\x00'+rgb[r].tobytes() for r in range(h))
    chunk = lambda tag, data: struct.pack('>I', len(data))+tag+data+struct.pack('>I', zlib.crc32(tag+data) & 0xffffffff)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(raw, 9))+chunk(b'IEND', b''))


def spectrogram(x):
    Y, _ = dsp.analysis(x)
    db = 10*np.log10(np.abs(Y)**2/(dsp.WINDOW/2)**2+1e-14)
    return ((db+100)/75).T[::-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(learned_alpha.default_checkpoint().relative_to(ROOT)))
    parser.add_argument('--tuned-alpha', type=float, default=0.9)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:  # training may rewrite best.pt meanwhile
        snapshot = Path(tmp)/'model.pt'
        shutil.copy(ROOT/args.checkpoint, snapshot)
        model = learned_alpha.load(snapshot)
        import torch
        epoch = torch.load(snapshot, map_location='cpu', weights_only=False)['epoch']
    OUT.mkdir(parents=True, exist_ok=True)
    magma, blues = lut(MAGMA), lut(BLUES)
    versions = [('noisy', 'Noisy input'), ('fir', 'FIR band-pass, 50–7,000 Hz'), ('dd098', 'Wiener, α = 0.98 (literature default)'),
                ('ddtuned', f'Wiener, α = {args.tuned_alpha:g} (tuned on 400 pairs)'),
                ('learned', f'Wiener, learned α (ours, epoch {epoch})'), ('clean', 'Clean reference')]
    sections, summary = [], {}
    for noise, name in CLIPS:
        s, fs = sf.read(ROOT/'assets'/'audio'/'clean'/f'{name}.wav')
        y, _ = sf.read(ROOT/'assets'/'audio'/'noisy'/f'{name}.wav')
        s, y = resample_poly(s, 1, 3), resample_poly(y, 1, 3)
        learned, alpha = learned_alpha.denoise(model, y, return_alpha=True)
        signals = dict(noisy=y, fir=dsp.fir_bandpass(y), dd098=dsp.wiener_dd(y, 0.98), ddtuned=dsp.wiener_dd(y, args.tuned_alpha),
                       learned=learned, clean=s)
        gain = 0.89/max(np.abs(x).max() for x in signals.values())
        k = slice(256, len(y)-256)
        rows, summary[noise] = [], {}
        for key, label in versions:
            x = signals[key]
            sf.write(OUT/f'{noise}_{key}.wav', (gain*x).astype(np.float32), 16000, subtype='PCM_16')
            write_png(OUT/f'{noise}_{key}.png', spectrogram(x), magma)
            m = qm.all_metrics(s[k], x[k]) if key != 'clean' else None
            summary[noise][key] = m
            metrics = '' if m is None else (f'<span>PESQ <b>{m["pesq"]:.2f}</b></span><span>STOI <b>{m["stoi"]:.3f}</b></span>'
                                            f'<span>SSNR <b>{m["ssnr"]:.1f} dB</b></span>')
            rows.append(f'<div class="row{" ours" if key == "learned" else ""}"><div class="meta"><h3>{html.escape(label)}</h3>'
                        f'<audio controls preload="none" src="{noise}_{key}.wav"></audio><div class="m">{metrics}</div></div>'
                        f'<img src="{noise}_{key}.png" alt="Spectrogram, {html.escape(label)}"></div>')
        write_png(OUT/f'{noise}_alpha.png', alpha.T[::-1], blues)
        rows.append(f'<div class="row alpha"><div class="meta"><h3>Learned α(k, m)</h3><p>Light: small α, follows the current frame. '
                    f'Dark: α near 1, smooths over time. Mean α = {alpha.mean():.3f}.</p>'
                    f'<div class="scale"><span>0</span><i></i><span>1</span></div></div><img src="{noise}_alpha.png" alt="Learned alpha map"></div>')
        title = {'bus': 'Bus', 'office': 'Office', 'cafe': 'Café'}[noise]
        sections.append(f'<section><h2>{title} <small>{name} · nominal SNR 2.5 dB</small></h2>{"".join(rows)}</section>')
    (OUT/'summary.json').write_text(json.dumps(dict(epoch=epoch, tuned_alpha=args.tuned_alpha, metrics=summary), indent=2))
    (OUT/'index.html').write_text(PAGE.replace('{{EPOCH}}', str(epoch)).replace('{{SECTIONS}}', ''.join(sections)), encoding='utf-8')
    print(json.dumps(summary, indent=1))


PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Listening preview · learned α</title><style>
:root{--ink:#142534;--soft:#465b6d;--faint:#7d8e9c;--blue:#1d5987;--line:#dbe3ea;--wash:#f0f6fa}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;color:var(--ink);background:#fff}
main{max-width:1180px;margin:0 auto;padding:36px 28px 60px}h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}
.lead{color:var(--soft);margin:0 0 28px;line-height:1.5;max-width:70ch}section{margin-bottom:44px}
h2{font-size:22px;margin:0 0 12px;border-bottom:2px solid var(--line);padding-bottom:8px}h2 small{font-size:14px;color:var(--faint);font-weight:500;margin-left:8px}
.row{display:grid;grid-template-columns:300px 1fr;gap:18px;align-items:center;padding:10px 12px;border-radius:10px}
.row+.row{margin-top:4px}.row.ours{background:var(--wash)}.row.alpha{margin-top:10px}
h3{font-size:15px;margin:0 0 8px}.ours h3{color:var(--blue)}audio{width:100%;height:34px}
.m{display:flex;gap:14px;font-size:13px;color:var(--soft);margin-top:6px}.m b{color:var(--ink);font-variant-numeric:tabular-nums}
img{width:100%;height:110px;image-rendering:auto;border-radius:6px;background:#000;display:block}
.alpha img{background:#fff;border:1px solid var(--line)}.alpha p{font-size:13px;color:var(--soft);margin:0 0 8px;line-height:1.45}
.scale{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--faint)}.scale i{flex:1;height:8px;border-radius:4px;background:linear-gradient(90deg,#f7fbff,#6baed6,#08306b)}
</style></head><body><main><h1>Listening preview: learned α</h1>
<p class="lead">Three demo clips (not training or test data) at 2.5 dB SNR. Every version of a clip shares one playback gain, so loudness differences are real. Spectrograms: 0–8 kHz, same colour scale. The learned model is the best checkpoint so far (epoch {{EPOCH}}); training is still running.</p>
{{SECTIONS}}</main></body></html>'''

if __name__ == '__main__':
    main()
