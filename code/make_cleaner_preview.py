#!/usr/bin/env python3
"""Listening page: learned-alpha Wiener at several noise over-estimation factors.

Writes outputs/cleaner_preview/{index.html,*.wav,*.png} for the three exploratory clips.
  .venv/bin/python code/make_cleaner_preview.py
"""
import html
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import learned_alpha
import quality_metrics as qm
from make_slide_figures import load, spec_png, OUT as _FIG
import make_slide_figures
from sweep_suppression import enhance, pause_residual_db

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'/'cleaner_preview'
SCALES = [1.0, 2.0, 3.0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_slide_figures.OUT = OUT
    model = learned_alpha.load(learned_alpha.default_checkpoint())
    sections = []
    for clip, title in [('bus', 'Bus'), ('office', 'Office'), ('cafe', 'Café')]:
        s, y = load(clip)
        alpha = learned_alpha.alpha_map(model, y)
        versions = [('noisy', 'Noisy input', y)]
        versions += [(f'x{c:g}', f'Ours, noise × {c:g}' + (' (current)' if c == 1 else ''), enhance(y, alpha, -25, c)) for c in SCALES]
        versions += [('clean', 'Clean reference', s)]
        gain = 0.89/max(np.abs(x).max() for _, _, x in versions)
        k = slice(256, len(y)-256)
        rows = []
        for key, label, x in versions:
            sf.write(OUT/f'{clip}_{key}.wav', (gain*x).astype(np.float32), 16000, subtype='PCM_16')
            spec_png(f'{clip}_{key}.png', x)
            m = '' if key == 'clean' else (lambda r: f'PESQ {r["pesq"]:.2f} · STOI {r["stoi"]:.3f} · pauses {pause_residual_db(s[k], y[k], x[k]):+.1f} dB')(qm.all_metrics(s[k], x[k]))
            rows.append(f'<div class="row"><div><h3>{html.escape(label)}</h3><audio controls preload="none" src="{clip}_{key}.wav"></audio>'
                        f'<p>{m}</p></div><img src="{clip}_{key}.png" alt=""></div>')
        sections.append(f'<section><h2>{title}</h2>{"".join(rows)}</section>')
    (OUT/'index.html').write_text(PAGE.replace('{{S}}', ''.join(sections)), encoding='utf-8')
    print(OUT/'index.html')


PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Cleaner settings · listening</title><style>
body{margin:0;font-family:Arial,Helvetica,sans-serif;color:#1d1d1f}main{max-width:1100px;margin:0 auto;padding:30px 24px 60px}
h1{color:#00274C;margin:0 0 6px}.lead{color:#555;margin:0 0 24px;line-height:1.5}h2{color:#00274C;border-bottom:2px solid #FFCB05;padding-bottom:6px}
.row{display:grid;grid-template-columns:300px 1fr;gap:18px;align-items:center;padding:8px 0}h3{font-size:15px;margin:0 0 6px}
audio{width:100%;height:34px}p{font-size:13px;color:#555;margin:6px 0 0}img{width:100%;height:100px;border-radius:5px;display:block}
</style></head><body><main><h1>How clean should it be?</h1><p class="lead">Our method at three noise over-estimation factors. Higher factor = cleaner pauses, but more risk of thin or cut speech.
"pauses" = noise left in speech pauses relative to the noisy input (more negative = cleaner). All versions of a clip share one playback level.</p>{{S}}</main></body></html>'''

if __name__ == '__main__':
    main()
