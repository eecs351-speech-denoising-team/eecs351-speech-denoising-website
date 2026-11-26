"""Predeclared bounded search: core strengths plus one-factor noise variations."""
import json
from pathlib import Path


def build_candidates():
    candidates = []

    def add(id, method, family, options):
        candidates.append(dict(id=id, method=method, family=family, options=options))

    add('identity', 'identity', 'baseline', {})
    for cutoff in [100, 200, 300]:
        for upper in [None, 7000]:
            add(f'fir_{cutoff}_{upper or "highpass"}', 'fir', 'fir',
                dict(order=128, cutoffHz=cutoff, upperCutoffHz=[] if upper is None else upper))
    common = dict(windowLength=400, hop=160, nfft=512, noiseFraction=0.2,
                  noiseLambda=0.9, pauseThresholdDb=3, minPauseFrames=3,
                  minEnergyContrastDb=3, storeNoiseHistory=False)
    for method, prefix in [('spectral_subtraction', 'ss'), ('wiener', 'wiener')]:
        strengths = [dict(alpha=a, beta=b) for a in [1, 2, 4] for b in [0.01, 0.05]] \
            if method == 'spectral_subtraction' else [dict(gainFloor=g) for g in [0.01, 0.05, 0.1]]
        base = dict(alpha=1, beta=0.01) if method == 'spectral_subtraction' else dict(gainFloor=0.05)
        for mode in ['fixed', 'pause_updated']:
            for k, strength in enumerate(strengths):
                add(f'{prefix}_{mode}_strength{k+1}', method, f'{prefix}_{mode}',
                    dict(common, noiseMode=mode, **strength))
            for fraction in [0.1, 0.3]:
                options = dict(common, noiseMode=mode, **base)
                options['noiseFraction'] = fraction
                add(f'{prefix}_{mode}_fraction{int(fraction*100)}', method, f'{prefix}_{mode}', options)
            if mode == 'pause_updated':
                for key, values in [('noiseLambda', [0.8, 0.98]), ('pauseThresholdDb', [0, 6])]:
                    for value in values:
                        options = dict(common, noiseMode=mode, **base)
                        options[key] = value
                        add(f'{prefix}_{mode}_{key}{value}', method, f'{prefix}_{mode}', options)
    assert len(candidates) == 41 and len({c['id'] for c in candidates}) == 41
    return candidates


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    path = root/'config/tuning_candidates.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(dict(schema_version=1, candidates=build_candidates()), indent=2)+'\n')
    print(path)
