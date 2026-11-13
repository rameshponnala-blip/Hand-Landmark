"""Step 10 (review minor 6): do the grid-searched baselines keep their edge under strict protocols?

The 3-fold grid search of s9 selects an RBF SVM with C=100 and 1-nearest-neighbour.
Both are compared with the MLP on Protocol B (exact McNemar), under frame-blocked
5-fold CV, leave-one-signer-out on Surrey and the two cross-dataset tests.

Usage:  .venv\\Scripts\\python.exe experiments\\s10_tuned.py
"""
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from common import (LETTERS, RES, SEED, Results, as_L, feat_norm, labels, letters_only, load_landmarks,  # noqa: E402
                    mcnemar, pct, pfmt, pm, protocol_b, scores)

ALL = ('default', 'pad', 'flip', 'upscale', 'lowconf')
MODELS = {'SVM': lambda: make_pipeline(StandardScaler(), SVC(C=100, gamma='scale')),
          'KNN': lambda: KNeighborsClassifier(n_neighbors=1)}


def fit_eval(m, Xa, ya, Xb, yb):
    p = MODELS[m]().fit(Xa, ya).predict(Xb)
    return p, scores(yb, p)['acc']


def main():
    R = Results('s10_tuned')
    D = letters_only(load_landmarks('alphabet'))
    y, X = labels(D), feat_norm(as_L(D))
    tr, te = protocol_b(y)
    pb = np.load(os.path.join(RES, 'protoB_preds.npz'))
    assert np.array_equal(pb['te'], te)
    groups = (D['label'] + '_' + (D['frame'] // 100).astype(str)).to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))
    S = letters_only(load_landmarks('surrey'), ALL)
    ys, Xs, sg = labels(S), feat_norm(as_L(S)), S['signer'].to_numpy()
    E = letters_only(load_landmarks('test'), ALL)

    jobs = []
    for m in MODELS:
        jobs.append(('B', m, 0, X[tr], y[tr], X[te], y[te]))
        for k, (a, b) in enumerate(folds):
            jobs.append(('Blocked', m, k, X[a], y[a], X[b], y[b]))
        for s in sorted(np.unique(sg)):
            a, b = np.flatnonzero(sg != s), np.flatnonzero(sg == s)
            jobs.append(('LOSO', m, s, Xs[a], ys[a], Xs[b], ys[b]))
    out = Parallel(n_jobs=12, verbose=5)(delayed(fit_eval)(m, a, b, c, d) for _, m, _, a, b, c, d in jobs)
    res = {}
    for (proto, m, *_), (p, acc) in zip(jobs, out):
        res.setdefault((proto, m), []).append(acc)
        if proto == 'B':
            b_, c_, pv = mcnemar(pb['MLP'] == y[te], p == y[te])
            R[f'TunedB{m}'] = pct(acc)
            R[f'TunedMcNemar{m}'], R[f'TunedMcNemar{m}b'], R[f'TunedMcNemar{m}c'] = pfmt(pv), str(b_), str(c_)
            R[f'TunedMcNemar{m}Rel'] = '<0.001' if pv < 1e-3 else f'={pv:.3f}'
    for (proto, m), v in res.items():
        if proto != 'B':
            R[f'Tuned{proto}{m}'] = pm(v)

    # cross-dataset (24 letters on Surrey: outputs restricted by predicting with all, masking not possible
    # for SVC.predict, so restrict the training set to the test letters instead)
    for name, df in (('Test', E), ('Surrey', S)):
        ye, Xe = labels(df), feat_norm(as_L(df))
        keep = np.isin(y, np.unique(ye))
        for m in MODELS:
            _, acc = fit_eval(m, X[keep], y[keep], Xe, ye)
            R[f'TunedCross{name}{m}'] = pct(acc)
    R.save()
    for k, v in sorted(R.R.items()):
        print(k, v)


if __name__ == '__main__':
    main()
