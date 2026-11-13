"""Step 8 (review M3): seed variance of the cross-dataset and handedness results.

Every MLP in the cross-dataset (s3) and handedness (s6) experiments is retrained
with five seeds (42-46). The values written here replace the single-run values of
s3/s6 in results.tex (build_results.py merges files in name order), as
"mean $\\pm$ std over seeds". Also fills the 'Seen' SVM cell on Surrey.

Usage:  .venv\\Scripts\\python.exe experiments\\s8_seeds.py
"""
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from common import (FEATURES, LETTERS, MLPClassifier, SEED, Results, as_L, labels, letters_only,  # noqa: E402
                    load_landmarks, pct, pm, protocol_b, scores)
from s6_handedness import canon, reflect  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
ALL = ('default', 'pad', 'flip', 'upscale', 'lowconf')


def run(tag, feat, seed, Ltr, ytr, tests):
    clf = MLPClassifier(seed=seed).fit(FEATURES[feat](Ltr), ytr)
    res = {}
    for name, (Lt, yt) in tests.items():
        z = clf.logits(FEATURES[feat](Lt))
        z[:, ~np.isin(np.arange(len(LETTERS)), np.unique(yt))] = -np.inf
        res[name] = scores(yt, z.argmax(1))['acc']
    return tag, seed, res


def svm_seen(X, y, a, b):
    clf = make_pipeline(StandardScaler(), SVC(C=10, gamma='scale')).fit(X[a], y[a])
    return scores(y[b], clf.predict(X[b]))['acc']


def main():
    R = Results('s8_seeds')
    A = letters_only(load_landmarks('alphabet'))
    y, L, h = labels(A), as_L(A), A['handedness'].to_numpy()
    tr, te = protocol_b(y)
    ext = {}
    for name in ('test', 'surrey'):
        d = letters_only(load_landmarks(name), ALL)
        ext[name] = (as_L(d), labels(d), d['handedness'].to_numpy())
    plain = {n: (e[0], e[1]) for n, e in ext.items()}

    jobs = []
    for seed in SEEDS:
        for feat in ('raw', 'rot'):                          # cross-dataset only
            jobs.append((f'X{feat}Plain', feat, seed, L, y, plain))
        for feat in ('norm', 'hybrid', 'angles'):            # cross-dataset and handedness
            jobs.append((f'B{feat}Plain', feat, seed, L[tr], y[tr], {'B': (L[te], y[te])}))
            jobs.append((f'B{feat}Canon', feat, seed, canon(L[tr], h[tr]), y[tr], {'B': (canon(L[te], h[te]), y[te])}))
            jobs.append((f'B{feat}Mirror', feat, seed, np.concatenate([L[tr], reflect(L[tr])]), np.r_[y[tr], y[tr]],
                         {'B': (L[te], y[te])}))
            jobs.append((f'X{feat}Plain', feat, seed, L, y, plain))
            jobs.append((f'X{feat}Canon', feat, seed, canon(L, h), y,
                         {n: (canon(e[0], e[2]), e[1]) for n, e in ext.items()}))
            jobs.append((f'X{feat}Mirror', feat, seed, np.concatenate([L, reflect(L)]), np.r_[y, y], plain))
    out = Parallel(n_jobs=10, verbose=5)(delayed(run)(*j) for j in jobs)
    acc = {}
    for tag, seed, res in out:
        for name, a in res.items():
            acc.setdefault((tag, name), []).append(a)
    for (tag, name), v in acc.items():
        R[f'Hand{tag}{name.capitalize()}'] = pm(v)
        R[f'Hand{tag}{name.capitalize()}Std'] = f'{100 * np.std(v):.2f}'
        feat = tag[1:].replace('Plain', '')
        if tag.startswith('X') and tag.endswith('Plain'):        # same runs as the cross-dataset table
            R[f'Cross{name.capitalize()}MLP{feat}'] = pm(v)
            R[f'Cross{name.capitalize()}MLP{feat}Mean'] = pct(np.mean(v))
    stds = [100 * np.std(v) for (tag, name), v in acc.items() if tag.startswith('X')]
    R['SeedStdMax'] = f'{max(stds):.1f}'
    R['SeedStdMedian'] = f'{np.median(stds):.1f}'
    ang = [np.mean(acc[(f'Xangles{c}', 'surrey')]) for c in ('Plain', 'Canon', 'Mirror')]
    R['AnglesSurreySpread'] = f'{100 * (max(ang) - min(ang)):.2f}'
    R['NumSeeds'] = str(len(SEEDS))
    print('seed runs done', flush=True)

    # 'Seen' SVM on Surrey (stratified 5-fold CV), for Table VII
    S = letters_only(load_landmarks('surrey'), ALL)
    Xs, ys = FEATURES['norm'](as_L(S)), labels(S)
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xs, ys)
    v = Parallel(n_jobs=5)(delayed(svm_seen)(Xs, ys, a, b) for a, b in folds)
    R['SurreySeenSVMnorm'] = pm(v)
    R.save()
    print('saved', len(R.R), 'values', flush=True)


if __name__ == '__main__':
    main()
