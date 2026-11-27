#!/usr/bin/env python3
"""Learned decision-directed smoothing: a small causal GRU predicts alpha(k, m).

Everything except alpha is the standard Wiener of dsp.wiener_dd: same STFT, the same fixed
noise estimate (quietest 10% frames), xi_min = -25 dB and the gain G = xi / (1 + xi).

  xi(k,m) = max(alpha(k,m) * |S_hat(k,m-1)|^2 / N(k) + (1 - alpha(k,m)) * max(gamma - 1, 0), xi_min)

Inputs per frame: log noisy power and log a posteriori SNR (514 features).
Training target: the oracle Wiener gain |S|^2 / (|S|^2 + |V|^2) from the clean/noise pair;
the loss is back-propagated through the DD recursion. The output layer starts at zero
with alpha = 0.98 everywhere, so training starts from the standard DD Wiener.
Model selection: mean PESQ on held-out training speakers (default p226, p287).

  .venv/bin/python code/learned_alpha.py train --epochs 20
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsp
import quality_metrics as qm

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'data'/'cache16k'
OUT = ROOT/'outputs'/'learned_alpha'
COMMITTED_MODEL = ROOT/'models'/'learned_alpha.pt'  # copy of the best checkpoint (epoch 18) kept in git


def default_checkpoint():
    """A freshly trained outputs/learned_alpha/best.pt if present, else the committed model."""
    fresh = OUT/'best.pt'
    return fresh if fresh.exists() else COMMITTED_MODEL
ALPHA_MAX = 0.999
ALPHA_INIT = 0.98
XI_MIN_DB = -25.0
NOISE_FRACTION = 0.1
EPS = 1e-10


def spectra(noisy, clean=None):
    """Noisy power P [T,F], noise PSD N [F], features [T,2F]; oracle gain if clean is given."""
    Y, _ = dsp.analysis(noisy)
    P = np.abs(Y)**2
    N = np.maximum(dsp.noise_psd(noisy, NOISE_FRACTION), np.finfo(float).tiny)
    features = np.concatenate([np.log10(P+EPS), np.log10(P/N+EPS)], axis=1)
    if clean is None:
        return P, N, features
    S, _ = dsp.analysis(clean)
    V, _ = dsp.analysis(noisy-clean)
    s2, v2 = np.abs(S)**2, np.abs(V)**2
    return P, N, features, s2/(s2+v2+EPS)


class AlphaNet(nn.Module):
    def __init__(self, bins=dsp.NFFT//2+1, hidden=128, layers=2):
        super().__init__()
        self.register_buffer('mean', torch.zeros(2*bins))
        self.register_buffer('std', torch.ones(2*bins))
        self.inp = nn.Sequential(nn.Linear(2*bins, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, bins)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, math.log(ALPHA_INIT/ALPHA_MAX/(1-ALPHA_INIT/ALPHA_MAX)))

    def forward(self, features):  # [B,T,2F] -> alpha [B,T,F]
        h, _ = self.gru(self.inp((features-self.mean)/self.std))
        return ALPHA_MAX*torch.sigmoid(self.out(h))


def dd_torch(alpha, P, N, mask=None):
    """Differentiable DD recursion; alpha, P [B,T,F], N [B,F]. Returns gains [B,T,F]."""
    xi_min = 10**(XI_MIN_DB/10)
    previous = torch.zeros_like(N)
    gains = []
    for m in range(P.shape[1]):
        gamma = P[:, m]/N
        xi = torch.clamp(alpha[:, m]*previous/N + (1-alpha[:, m])*torch.clamp(gamma-1, min=0), min=xi_min)
        g = xi/(1+xi)
        gains.append(g)
        previous = g*g*P[:, m]
    return torch.stack(gains, 1)


def load(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = AlphaNet(**checkpoint['config'])
    model.load_state_dict(checkpoint['state'])
    return model.eval()


@torch.no_grad()
def alpha_map(model, noisy):
    _, _, features = spectra(np.asarray(noisy, float))
    return model(torch.from_numpy(features).float()[None])[0].double().numpy()


def denoise(model, noisy, return_alpha=False):
    alpha = alpha_map(model, noisy)
    out = dsp.wiener_dd(noisy, alpha=alpha, xi_min_db=XI_MIN_DB, noise_fraction=NOISE_FRACTION)
    return (out, alpha) if return_alpha else out


class Pairs(torch.utils.data.Dataset):
    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        with np.load(CACHE/'train'/f'{self.names[i]}.npz') as d:
            clean, noisy = d['clean'].astype(np.float64), d['noisy'].astype(np.float64)
        P, N, features, target = spectra(noisy, clean)
        return [torch.from_numpy(x).float() for x in (P, N, features, target)]


def collate(items):
    T = max(x[0].shape[0] for x in items)
    pad = lambda x: torch.nn.functional.pad(x, (0, 0, 0, T-x.shape[0]))
    mask = torch.stack([torch.arange(T) < x[0].shape[0] for x in items]).float()
    return (torch.stack([pad(x[0]) for x in items]), torch.stack([x[1] for x in items]),
            torch.stack([pad(x[2]) for x in items]), torch.stack([pad(x[3]) for x in items]), mask)


def length_batches(names, lengths, batch, rng):
    """Shuffle, then group similar lengths to limit padding in the sequential recursion."""
    order = sorted(names, key=lambda n: lengths[n] + rng.uniform(0, 8000))
    batches = [order[i:i+batch] for i in range(0, len(order), batch)]
    rng.shuffle(batches)
    return batches


def validate_pesq(model, names):
    scores = []
    for name in names:
        with np.load(CACHE/'train'/f'{name}.npz') as d:
            clean, noisy = d['clean'].astype(np.float64), d['noisy'].astype(np.float64)
        out = denoise(model, noisy)
        k = slice(256, len(noisy)-256)
        scores.append((qm.pesq_wb(clean[k], out[k]), qm.stoi(clean[k], out[k])))
    return [float(v) for v in np.mean(scores, axis=0)]


def train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    with (CACHE/'train_index.csv').open() as f:
        index = list(csv.DictReader(f))
    lengths = {r['name']: int(r['samples']) for r in index}
    val_speakers = set(args.val_speakers)
    train_names = [r['name'] for r in index if r['speaker'] not in val_speakers]
    val_names = sorted(r['name'] for r in index if r['speaker'] in val_speakers)
    val_pesq_names = random.Random(351).sample(val_names, min(args.val_pesq, len(val_names)))
    print(f'train {len(train_names)} pairs, validation {len(val_names)} pairs '
          f'({", ".join(sorted(val_speakers))}); PESQ on {len(val_pesq_names)}', flush=True)

    config = dict(hidden=args.hidden, layers=args.layers)
    model = AlphaNet(**config)
    stats = np.concatenate([spectra(*[np.load(CACHE/'train'/f'{n}.npz')[k].astype(np.float64)
                                      for k in ('noisy',)])[2] for n in rng.sample(train_names, 300)])
    model.mean.copy_(torch.from_numpy(stats.mean(0)).float())
    model.std.copy_(torch.from_numpy(stats.std(0)+1e-3).float())
    device = torch.device(args.device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader_kw = dict(collate_fn=collate, num_workers=args.workers, persistent_workers=args.workers > 0)
    OUT.mkdir(parents=True, exist_ok=True)
    history, best = [], None

    def evaluate_epoch(epoch, train_loss):
        nonlocal best
        model.eval()
        cpu_model = AlphaNet(**config)
        cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
        pesq, stoi = validate_pesq(cpu_model.eval(), val_pesq_names)
        record = dict(epoch=epoch, train_loss=train_loss, val_pesq=pesq, val_stoi=stoi)
        history.append(record)
        checkpoint = dict(config=config, state=cpu_model.state_dict(), epoch=epoch, record=record,
                          val_speakers=sorted(val_speakers), seed=args.seed)
        torch.save(checkpoint, OUT/f'epoch{epoch:02d}.pt')
        if best is None or pesq > best['val_pesq']:
            best = record
            torch.save(checkpoint, OUT/'best.pt')
        (OUT/'history.json').write_text(json.dumps(dict(history=history, best=best, args=vars(args)), indent=2))
        print(f'epoch {epoch:2d}  train loss {train_loss:.5f}  val PESQ {pesq:.3f}  val STOI {stoi:.4f}'
              f'{"  *best" if best is record else ""}', flush=True)

    evaluate_epoch(0, float('nan'))  # epoch 0 = standard DD Wiener (alpha = 0.98 everywhere)
    for epoch in range(1, args.epochs+1):
        model.train()
        batches = length_batches(train_names, lengths, args.batch, rng)
        if args.max_batches:
            batches = batches[:args.max_batches]
        loader = torch.utils.data.DataLoader(Pairs([n for b in batches for n in b]),
                                             batch_sampler=_Fixed(batches, train_names), **loader_kw)
        total, count, start = 0.0, 0, time.time()
        for k, (P, N, features, target, mask) in enumerate(loader, 1):
            P, N, features, target, mask = (x.to(device) for x in (P, N, features, target, mask))
            gains = dd_torch(model(features), P, N)
            loss = (((gains-target)**2).mean(-1)*mask).sum()/mask.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            count += 1
            if k % 100 == 0:
                print(f'  epoch {epoch} batch {k}/{len(batches)}  loss {total/count:.5f}  '
                      f'{(time.time()-start)/k:.2f} s/batch', flush=True)
        evaluate_epoch(epoch, total/count)
    print('best', json.dumps(best))


class _Fixed(torch.utils.data.Sampler):
    """Yield the precomputed batches as dataset indices of the concatenated order."""
    def __init__(self, batches, _names):
        self.batches, k = [], 0
        for b in batches:
            self.batches.append(list(range(k, k+len(b))))
            k += len(b)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--epochs', type=int, default=20)
    t.add_argument('--batch', type=int, default=16)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--hidden', type=int, default=128)
    t.add_argument('--layers', type=int, default=2)
    t.add_argument('--val-speakers', nargs='+', default=['p226', 'p287'])
    t.add_argument('--val-pesq', type=int, default=200)
    t.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    t.add_argument('--workers', type=int, default=4)
    t.add_argument('--max-batches', type=int, default=0, help='limit batches per epoch (smoke tests)')
    t.add_argument('--seed', type=int, default=351)
    args = parser.parse_args()
    train(args)
