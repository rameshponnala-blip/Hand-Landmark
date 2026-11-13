"""Step 3 (gap G1): how far does the recognizer generalize?

(a) near-duplicate images between training and test sets of a random split;
(b) random vs. frame-blocked 5-fold CV on the Kaggle ASL Alphabet;
(c) cross-dataset tests: train on Kaggle, test on the varied-background test set and Surrey;
(d) leave-one-signer-out (LOSO) on the 5-signer Surrey dataset for every feature set (G2).

Usage:  .venv\\Scripts\\python.exe experiments\\s3_generalization.py
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from common import (FEATURES, LETTERS, MLPClassifier, SEED, Results, as_L, labels, letters_only,  # noqa: E402
                    load_landmarks, pct, pm, protocol_b, scores, write_dat)

ALL_METHODS = ('default', 'pad', 'flip', 'upscale', 'lowconf')
MODELS = {
    'KNN': lambda: KNeighborsClassifier(n_neighbors=5),
    'SVM': lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma='scale')),
    'MLP': lambda: MLPClassifier(),
}


def popcount64(a):
    """Number of set bits of each uint64 (Hamming distance after XOR)."""
    b = np.ascontiguousarray(a, dtype=np.uint64).view(np.uint8).reshape(*a.shape, 8)
    return np.unpackbits(b, axis=-1).sum(-1).astype(np.int64)


def fit_eval(make, Xtr, ytr, Xte, yte, mask=None):
    clf = make().fit(Xtr, ytr)
    if mask is not None and hasattr(clf, 'logits'):           # restrict output to the test label set
        z = clf.logits(Xte)
        z[:, ~mask] = -np.inf
        p = z.argmax(1)
    else:
        p = clf.predict(Xte)
    return scores(yte, p)['acc'], scores(yte, p)['f1']


def main():
    R = Results('s3_generalization')
    alpha = load_landmarks('alphabet')
    D = letters_only(alpha)
    y, L = labels(D), as_L(D)
    Xn = FEATURES['norm'](L)

    # (a) near duplicates in a random split
    tr, te = protocol_b(y)
    hashes = D['dhash'].astype('uint64').to_numpy()
    mind = np.empty(len(te), np.int64)
    for c in range(len(LETTERS)):
        sel = y[te] == c
        if not sel.any():
            continue
        x = hashes[te[sel]][:, None] ^ hashes[tr[y[tr] == c]][None, :]
        mind[sel] = popcount64(x).min(1)
    for t in (0, 2, 5):
        R[f'NearDup{t}'] = pct((mind <= t).mean(), 1)
    hist = np.bincount(np.minimum(mind, 16), minlength=17)
    write_dat('neardup_hist', pd.DataFrame({'d': np.arange(17), 'pct': 100 * hist / hist.sum()}))
    R['NearDupMedian'] = str(int(np.median(mind)))

    # (b) random vs frame-blocked cross-validation
    groups = (D['label'] + '_' + (D['frame'] // 100).astype(str)).to_numpy()
    R['NumBlocks'] = str(len(np.unique(groups)))
    splitters = {'Random': StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xn, y),
                 'Blocked': StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xn, y, groups)}
    jobs = []
    for sname, sp in splitters.items():
        for k, (a, b) in enumerate(sp):
            for m in MODELS:
                jobs.append((sname, m, a, b))
    out = Parallel(n_jobs=12, verbose=5)(delayed(fit_eval)(MODELS[m], Xn[a], y[a], Xn[b], y[b])
                                         for _, m, a, b in jobs)
    res = {}
    for (sname, m, _, _), (acc, f1) in zip(jobs, out):
        res.setdefault((sname, m), []).append(acc)
    for (sname, m), v in res.items():
        R[f'Split{sname}{m}'] = pm(v)
    for m in MODELS:
        R[f'SplitGap{m}'] = f'{100 * (np.mean(res[("Random", m)]) - np.mean(res[("Blocked", m)])):.2f}'
    print('split comparison done', flush=True)

    # (c) cross-dataset generalisation
    ext = {}
    for name in ('test', 'surrey'):
        try:
            df = letters_only(load_landmarks(name), ALL_METHODS)
        except FileNotFoundError:
            continue
        ext[name] = df
    feats = ['raw', 'norm', 'rot', 'angles', 'hybrid']
    jobs = []
    for name, df in ext.items():
        ye, Le = labels(df), as_L(df)
        mask = np.isin(np.arange(len(LETTERS)), np.unique(ye))
        for f in feats:
            jobs.append((name, f'MLP{f}', MODELS['MLP'], FEATURES[f](L), y, FEATURES[f](Le), ye, mask))
        jobs.append((name, 'SVMnorm', MODELS['SVM'], Xn, y, FEATURES['norm'](Le), ye, None))
    out = Parallel(n_jobs=12)(delayed(fit_eval)(mk, a, b, c, d, msk) for _, _, mk, a, b, c, d, msk in jobs)
    for (name, tag, *_), (acc, f1) in zip(jobs, out):
        R[f'Cross{name.capitalize()}{tag}'] = pct(acc)
        R[f'Cross{name.capitalize()}{tag}F1'] = pct(f1)
    for name, df in ext.items():
        R[f'Cross{name.capitalize()}N'] = f'{len(df):,}'
        R[f'Cross{name.capitalize()}Classes'] = str(df['label'].nunique())
        full = load_landmarks(name)
        full = full[full['label'].isin(LETTERS)]
        vc, n, tag = full['method'].value_counts(), len(full), name.capitalize()
        R[f'Det{tag}N'] = f'{n:,}'
        R[f'Det{tag}Default'] = pct(vc.get('default', 0) / n)
        R[f'Det{tag}All'] = pct((n - vc.get('none', 0) - vc.get('unreadable', 0)) / n)
        for m in ['pad', 'flip', 'upscale', 'lowconf']:
            R[f'Det{tag}{m.capitalize()}'] = pct(vc.get(m, 0) / n)
    print('cross-dataset done', flush=True)

    # (d) Surrey: LOSO vs seen signers
    if 'surrey' in ext and ext['surrey']['signer'].nunique() > 1:
        S = ext['surrey']
        ys, Ls, sg = labels(S), as_L(S), S['signer'].to_numpy()
        signers = sorted(np.unique(sg))
        R['SurreySigners'] = str(len(signers))
        R['SurreyN'] = f'{len(S):,}'
        jobs = []
        for f in feats:
            Xs = FEATURES[f](Ls)
            for s in signers:
                a, b = np.flatnonzero(sg != s), np.flatnonzero(sg == s)
                jobs.append(('LOSO', f'MLP{f}', s, MODELS['MLP'], Xs[a], ys[a], Xs[b], ys[b]))
            for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xs, ys)):
                jobs.append(('Seen', f'MLP{f}', k, MODELS['MLP'], Xs[a], ys[a], Xs[b], ys[b]))
        Xs = FEATURES['norm'](Ls)
        for s in signers:
            a, b = np.flatnonzero(sg != s), np.flatnonzero(sg == s)
            jobs.append(('LOSO', 'SVMnorm', s, MODELS['SVM'], Xs[a], ys[a], Xs[b], ys[b]))
        out = Parallel(n_jobs=12, verbose=5)(delayed(fit_eval)(mk, a, b, c, d)
                                             for _, _, _, mk, a, b, c, d in jobs)
        tab = {}
        rows = []
        for (proto, tag, fold, *_), (acc, f1) in zip(jobs, out):
            tab.setdefault((proto, tag), []).append(acc)
            if proto == 'LOSO':
                rows.append({'model': tag, 'signer': fold, 'acc': 100 * acc})
        for (proto, tag), v in tab.items():
            R[f'Surrey{proto}{tag}'] = pm(v)
            R[f'Surrey{proto}{tag}Min'] = pct(min(v))
        per = pd.DataFrame(rows).pivot(index='signer', columns='model', values='acc').reset_index()
        per['idx'] = np.arange(1, len(per) + 1)
        write_dat('surrey_loso', per)
        for f in feats:
            R[f'SurreyGap{f}'] = f'{100 * (np.mean(tab[("Seen", f"MLP{f}")]) - np.mean(tab[("LOSO", f"MLP{f}")])):.2f}'
    R.save()
    print('saved', len(R.R), 'values', flush=True)


if __name__ == '__main__':
    main()
