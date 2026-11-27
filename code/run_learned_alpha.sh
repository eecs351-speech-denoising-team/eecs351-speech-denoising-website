#!/usr/bin/env bash
# Full learned-alpha study, in order. Every selection rule is fixed before it runs, so the
# 824-pair test set is scored once, at the end. Needs the archives in data/archives/
# (see code/prepare_corpus.py) and the venv: python3.13 -m venv .venv &&
# .venv/bin/pip install -r requirements.txt pesq soundfile torch
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY code/prepare_corpus.py --split test
$PY code/prepare_corpus.py --split train
# 1. One global alpha for the standard DD Wiener, chosen on the 400 tuning pairs.
$PY code/run_study.py tune-alpha
ALPHA=$($PY -c "import json;print(json.load(open('reports/learned_alpha/tuned_alpha.json'))['alpha'])")
# 2. Learned per time-frequency alpha; best epoch by validation PESQ (speakers p226, p287).
$PY code/learned_alpha.py train --epochs 20 --batch 16 --device cpu --workers 3
# 3. Final comparison on the full test set.
SYSTEMS="noisy fir dd:0.98"
[ "$ALPHA" != "0.98" ] && SYSTEMS="$SYSTEMS dd:$ALPHA"
$PY code/run_study.py evaluate --systems $SYSTEMS learned:outputs/learned_alpha/best.pt --tag test
