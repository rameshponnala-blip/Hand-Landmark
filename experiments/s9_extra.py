"""Step 9 (review minor 5, 6 and M7, M8): supporting analyses.

(a) Near-duplicates: pixel-level SSIM of identical-hash pairs, their frame distance,
    and the near-duplicate rate that remains under blocked splits of several block sizes;
(b) baseline hyper-parameters chosen by 3-fold grid search on the Protocol-B training split;
(c) rejection of synthetic transition hand shapes (interpolated between two letters);
(d) simulated fingerspelling: words spelled with hand shapes of the unseen ASL Alphabet Test
    signer, decoded with Algorithm 1 and the spelling corrector.

Usage:  .venv\\Scripts\\python.exe experiments\\s9_extra.py [abc]
        (the letters select the parts; each part writes its own results file)
"""
import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from common import (LETTERS, MLPClassifier, RES, SEED, Results, as_L, feat_norm, labels, letters_only,  # noqa: E402
                    load_landmarks, pct, protocol_b, rotate_xy, scores)
from s3_generalization import popcount64  # noqa: E402

ALL = ('default', 'pad', 'flip', 'upscale', 'lowconf')
TAU, N_RUN = 0.8, 10


def ssim(a, b):
    """Single-scale SSIM of two grayscale images in [0, 255] (Gaussian window, sigma 1.5)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = gaussian_filter(a, 1.5), gaussian_filter(b, 1.5)
    va = gaussian_filter(a * a, 1.5) - mu_a ** 2
    vb = gaussian_filter(b * b, 1.5) - mu_b ** 2
    cov = gaussian_filter(a * b, 1.5) - mu_a * mu_b
    return float(np.mean((2 * mu_a * mu_b + c1) * (2 * cov + c2) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))))


def gray(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)


def nearest(hashes, y, te, tr):
    """For each test index: min Hamming distance and index of the closest same-letter training image."""
    d, j = np.empty(len(te), np.int64), np.empty(len(te), np.int64)
    for c in np.unique(y[te]):
        sel = y[te] == c
        cand = tr[y[tr] == c]
        x = popcount64(hashes[te[sel]][:, None] ^ hashes[cand][None, :])
        d[sel], j[sel] = x.min(1), cand[x.argmin(1)]
    return d, j


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def decode(frames, mlp):
    """Algorithm 1 on a sequence of landmark arrays (None = no hand)."""
    w, prev, r = '', None, 0
    X = [f for f in frames if f is not None]
    p = mlp.predict_proba(feat_norm(np.stack(X))) if X else np.zeros((0, 26))
    k = 0
    for f in frames:
        if f is None:
            continue
        c, conf = LETTERS[p[k].argmax()], p[k].max()
        k += 1
        if conf < TAU:
            continue
        if c == prev:
            r += 1
        else:
            prev, r = c, 0
        if r == N_RUN and (not w or w[-1] != c):
            w += c
    return w


def spell_word(word, bank, rng):
    """Frames for one word: hold each letter 12-24 frames with jitter, 6-frame transitions, 5% dropouts."""
    frames, prev = [], None
    for ch in word.upper():
        L = bank[ch][rng.integers(len(bank[ch]))]
        if prev is not None:
            for a in np.linspace(0, 1, 8)[1:-1]:
                frames.append((1 - a) * prev + a * L)
        for _ in range(rng.integers(12, 25)):
            f = rotate_xy(L[None], np.radians(rng.normal(0, 3)))[0] + rng.normal(0, 0.003, L.shape).astype(np.float32)
            frames.append(f)
        prev = L
    return [None if rng.random() < 0.05 else f for f in frames]


def main(parts):
    D = letters_only(load_landmarks('alphabet'))
    y, L = labels(D), as_L(D)
    tr, te = protocol_b(y)
    hashes = D['dhash'].astype('uint64').to_numpy()
    paths, frame = D['path'].to_numpy(), D['frame'].to_numpy()
    rng = np.random.default_rng(SEED)
    X = feat_norm(L)
    if 'a' in parts:
        part_a(D, y, L, tr, te, hashes, paths, frame, rng, X)
    if 'b' in parts:
        part_b(y, tr, te, X)
    if 'c' in parts:
        part_c(y, te, X, rng)


def part_a(D, y, L, tr, te, hashes, paths, frame, rng, X):
    """(a) near-duplicate validation."""
    R = Results('s9a_neardup')
    d, j = nearest(hashes, y, te, tr)
    zero = np.flatnonzero(d == 0)
    pick = rng.choice(zero, min(1000, len(zero)), replace=False)
    s_dup = Parallel(n_jobs=8)(delayed(ssim)(gray(paths[te[i]]), gray(paths[j[i]])) for i in pick)
    rnd = [(rng.choice(np.flatnonzero(y[tr] == y[te[i]]))) for i in pick]
    s_rnd = Parallel(n_jobs=8)(delayed(ssim)(gray(paths[te[i]]), gray(paths[tr[k]])) for i, k in zip(pick, rnd))
    R['DupSSIMMedian'] = f'{np.median(s_dup):.3f}'
    R['DupSSIMLow'] = f'{np.percentile(s_dup, 5):.3f}'
    R['RandSSIMMedian'] = f'{np.median(s_rnd):.3f}'
    R['DupFrameGapMedian'] = str(int(np.median(np.abs(frame[te[pick]] - frame[j[pick]]))))
    R['DupFrameGapP90'] = str(int(np.percentile(np.abs(frame[te[pick]] - frame[j[pick]]), 90)))
    R['NearDupRandom2'] = pct((d <= 2).mean(), 1)
    rows = []
    for bs in (25, 50, 100, 200, 400):
        g = (D['label'] + '_' + (D['frame'] // bs).astype(str)).to_numpy()
        a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g))
        db, _ = nearest(hashes, y, b, a)
        acc = scores(y[b], MLPClassifier(threads=4).fit(X[a], y[a]).predict(X[b]))['acc']
        rows.append({'block': bs, 'dup2': 100 * (db <= 2).mean(), 'dup0': 100 * (db == 0).mean(), 'acc': 100 * acc})
        R[f'BlockDup{bs}'] = pct((db <= 2).mean(), 1)
        R[f'BlockAcc{bs}'] = pct(acc)
        print('block', rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'block_sizes.csv'), index=False)
    R.save()


def part_b(y, tr, te, X):
    """(b) tuned classical baselines."""
    R = Results('s9b_tuning')
    grids = {
        'LR': (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
               {'logisticregression__C': [0.1, 1, 10, 100]}),
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [1, 3, 5, 9, 15], 'weights': ['uniform', 'distance']}),
        'SVM': (make_pipeline(StandardScaler(), SVC()),
                {'svc__C': [1, 10, 100], 'svc__gamma': ['scale', 0.005, 0.05]}),
        'RF': (RandomForestClassifier(random_state=SEED), {'n_estimators': [100, 300, 600],
                                                           'max_features': ['sqrt', 0.3]}),
    }
    for name, (est, grid) in grids.items():
        gs = GridSearchCV(est, grid, cv=3, n_jobs=8).fit(X[tr], y[tr])
        acc = scores(y[te], gs.predict(X[te]))['acc']
        best = ', '.join(f'{k.split("__")[-1]}={v}' for k, v in gs.best_params_.items())
        R[f'Tuned{name}Acc'] = pct(acc)
        R[f'Tuned{name}Params'] = best.replace('_', ' ')
        print(name, best, pct(acc), flush=True)
    R.save()


def part_c(y, te, X, rng):
    """(c) transition hand shapes and (d) video simulation."""
    R = Results('s9c_video')
    mlp = pickle.load(open(os.path.join(RES, 'mlp_protoB.pkl'), 'rb'))
    Xte = X[te]
    ia, ib = rng.integers(len(te), size=(2, 20000))
    keep = y[te][ia] != y[te][ib]
    ia, ib = ia[keep][:5000], ib[keep][:5000]
    for a in (0.25, 0.5):
        Xm = (1 - a) * Xte[ia] + a * Xte[ib]
        conf = mlp.predict_proba(Xm).max(1)
        R[f'TransReject{int(a * 100)}'] = pct((conf < TAU).mean())
    R['TransN'] = f'{len(ia):,}'

    from spellchecker import SpellChecker
    spell = SpellChecker()
    E = letters_only(load_landmarks('test'), ALL)
    bank = {c: as_L(E[E['label'] == c]) for c in LETTERS}
    freq = spell.word_frequency.dictionary
    ranked = sorted(freq, key=freq.get, reverse=True)[:20000]
    common = [w for w in ranked if w.isalpha() and w.isascii() and 3 <= len(w) <= 8][:3000]
    single = [w for w in common if all(a != b for a, b in zip(w, w[1:]))]
    double = [w for w in common if any(a == b for a, b in zip(w, w[1:]))]
    names = ['ramesh', 'kaleem', 'vitheesha', 'ponnala', 'hyderabad', 'telangana', 'priya', 'arjun', 'deepika',
             'suresh', 'lakshmi', 'venkat', 'anjali', 'kiran', 'rahul', 'sneha', 'vikram', 'kavya', 'manoj',
             'divya', 'srinivas', 'harsha', 'pavani', 'sandeep', 'swathi', 'naveen', 'bhavana', 'charan',
             'yamini', 'gopal']
    sets = {'Words': list(rng.choice(single, 300, replace=False)), 'Double': list(rng.choice(double, 100, replace=False)),
            'Names': names}
    for tag, words in sets.items():
        cer_raw, cer_cor, ok_raw, ok_cor, changed_ok = [], [], [], [], 0
        for w in words:
            raw = decode(spell_word(w, bank, rng), mlp).lower()
            cor = (spell.correction(raw) or raw) if raw else raw
            cer_raw.append(levenshtein(raw, w) / len(w))
            cer_cor.append(levenshtein(cor, w) / len(w))
            ok_raw.append(raw == w)
            ok_cor.append(cor == w)
            changed_ok += (raw == w) and (cor != w)
        R[f'Sim{tag}N'] = str(len(words))
        R[f'Sim{tag}CERRaw'] = pct(np.mean(cer_raw), 1)
        R[f'Sim{tag}CERCor'] = pct(np.mean(cer_cor), 1)
        R[f'Sim{tag}AccRaw'] = pct(np.mean(ok_raw), 1)
        R[f'Sim{tag}AccCor'] = pct(np.mean(ok_cor), 1)
        R[f'Sim{tag}Broken'] = str(changed_ok)
        print(tag, R[f'Sim{tag}CERRaw'], R[f'Sim{tag}CERCor'], R[f'Sim{tag}AccRaw'], R[f'Sim{tag}AccCor'],
              changed_ok, flush=True)
    R['SimTau'] = str(TAU)
    R.save()
    print('saved', len(R.R), 'values', flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'abc')
