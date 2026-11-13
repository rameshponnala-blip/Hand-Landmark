"""Step 6: handedness. ASL Alphabet shows mirror-image hands relative to Surrey.

Two remedies are compared with the plain normalized descriptor:
  canon  - reflect every hand that MediaPipe labels 'Right' (x -> -x about the wrist);
  mirror - train on every sample plus its reflected copy (handedness-invariant training).
Evaluated on Protocol B, ASL Alphabet Test and Surrey (cross-dataset), as in s3.

Usage:  .venv\\Scripts\\python.exe experiments\\s6_handedness.py
"""
import os
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from common import (FEATURES, LETTERS, MLPClassifier, Results, as_L, labels, letters_only, load_landmarks,  # noqa: E402
                    pct, protocol_b, scores)

ALL = ('default', 'pad', 'flip', 'upscale', 'lowconf')


def reflect(L):
    out = L.copy()
    out[..., 0] = 2 * L[:, :1, 0] - L[..., 0]
    return out


def canon(L, hand):
    out = L.copy()
    r = hand == 'Right'
    out[r] = reflect(L[r])
    return out


def run(tag, feat, Ltr, ytr, tests):
    clf = MLPClassifier().fit(FEATURES[feat](Ltr), ytr)
    res = {}
    for name, (Lt, yt) in tests.items():
        z = clf.logits(FEATURES[feat](Lt))
        z[:, ~np.isin(np.arange(len(LETTERS)), np.unique(yt))] = -np.inf
        res[name] = scores(yt, z.argmax(1))['acc']
    return tag, res


def main():
    R = Results('s6_handedness')
    A = letters_only(load_landmarks('alphabet'))
    y, L, h = labels(A), as_L(A), A['handedness'].to_numpy()
    tr, te = protocol_b(y)
    ext = {}
    for name in ('test', 'surrey'):
        d = letters_only(load_landmarks(name), ALL)
        ext[name] = (as_L(d), labels(d), d['handedness'].to_numpy())
    for name, d in [('Alphabet', A)] + [(n.capitalize(), letters_only(load_landmarks(n), ALL)) for n in ext]:
        R[f'Hand{name}Left'] = pct((d['handedness'] == 'Left').mean(), 1)

    jobs = []
    for feat in ('norm', 'hybrid', 'angles'):
        # (1) protocol B: train on the training split, test on the hold-out split
        jobs.append((f'B{feat}Plain', feat, L[tr], y[tr], {'B': (L[te], y[te])}))
        jobs.append((f'B{feat}Canon', feat, canon(L[tr], h[tr]), y[tr], {'B': (canon(L[te], h[te]), y[te])}))
        jobs.append((f'B{feat}Mirror', feat, np.concatenate([L[tr], reflect(L[tr])]), np.r_[y[tr], y[tr]],
                     {'B': (L[te], y[te])}))
        # (2) cross-dataset: train on all of ASL Alphabet
        jobs.append((f'X{feat}Plain', feat, L, y, {n: (e[0], e[1]) for n, e in ext.items()}))
        jobs.append((f'X{feat}Canon', feat, canon(L, h), y, {n: (canon(e[0], e[2]), e[1]) for n, e in ext.items()}))
        jobs.append((f'X{feat}Mirror', feat, np.concatenate([L, reflect(L)]), np.r_[y, y],
                     {n: (e[0], e[1]) for n, e in ext.items()}))
    out = Parallel(n_jobs=12, verbose=5)(delayed(run)(*j) for j in jobs)
    for tag, res in out:
        for name, acc in res.items():
            R[f'Hand{tag}{name.capitalize()}'] = pct(acc)
    R.save()
    print('saved', len(R.R), 'values', flush=True)


if __name__ == '__main__':
    main()
