"""Step 11: Surrey results per signer (Table VII) and the threshold test with non-letters (Table XI).

a: leave-one-signer-out on Surrey. For every held-out signer: how many hands MediaPipe finds (default
   detector / with the fallback steps), accuracy on the detected hands and end-to-end accuracy where a
   missed hand counts as an error. Also the same with default detections only, and the worst letters.
d: pick tau on 'del' (plus validation letters) by Youden's J and test it on 'space', then the other way
   round. Needs the model saved by s2_main.py.

Usage:  python experiments/s11_signers_threshold.py [ad]
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from common import (FEATURES, LETTERS, MLPClassifier, RES, SEED, Results, as_L, labels,  # noqa: E402
                    letters_only, load_landmarks, pct, protocol_b, write_dat)

ALL = ('default', 'pad', 'flip', 'upscale', 'lowconf')
TAU = 0.8


# (a)
def loso_fit(seed, Xtr, ytr, Xte):
    return MLPClassifier(seed=seed).fit(Xtr, ytr).predict(Xte)


def part_a():
    R = Results('s11a_signers')
    full = load_landmarks('surrey')
    full = full[full['label'].isin(LETTERS)].reset_index(drop=True)
    S = letters_only(full, ALL)
    Sd = letters_only(full, ('default',))
    signers = sorted(S['signer'].unique())

    jobs = []
    for s in signers:                                         # recovery-cascade detections (as in s3)
        a, b = (S['signer'] != s).to_numpy(), (S['signer'] == s).to_numpy()
        jobs.append(('all', s, S[a], S[b]))
    for s in signers:                                         # default detector only
        a, b = (Sd['signer'] != s).to_numpy(), (Sd['signer'] == s).to_numpy()
        jobs.append(('default', s, Sd[a], Sd[b]))
    out = Parallel(n_jobs=10, verbose=5)(delayed(loso_fit)(SEED, FEATURES['norm'](as_L(tr)), labels(tr),
                                                           FEATURES['norm'](as_L(te))) for _, _, tr, te in jobs)
    rows, letter_ok = [], []
    acc_def = {}
    for (kind, s, _, te), p in zip(jobs, out):
        ok = p == labels(te)
        if kind == 'default':
            acc_def[s] = ok.mean()
            continue
        n_all = int((full['signer'] == s).sum())
        n_def = int(((full['signer'] == s) & (full['method'] == 'default')).sum())
        rows.append({'signer': s, 'n': n_all, 'det_default': n_def / n_all, 'det_all': len(te) / n_all,
                     'acc': ok.mean(), 'e2e': ok.sum() / n_all})
        letter_ok.append(pd.DataFrame({'label': te['label'].to_numpy(), 'ok': ok}))
    T = pd.DataFrame(rows)
    T['acc_default'] = T['signer'].map(acc_def)
    T['e2e_default'] = T['acc_default'] * T['det_default']
    print(T.to_string(), flush=True)
    for i, r in T.iterrows():
        tag = 'ABCDE'[i] if len(T) <= 5 else str(i)
        R[f'Sig{tag}Name'] = str(r['signer'])
        R[f'Sig{tag}N'] = f'{r["n"]:,}'
        for k in ('det_default', 'det_all', 'acc', 'e2e', 'acc_default', 'e2e_default'):
            R[f'Sig{tag}{k.replace("_", "").capitalize()}'] = pct(r[k], 1)
    for k in ('det_default', 'det_all', 'acc', 'e2e', 'acc_default', 'e2e_default'):
        key = k.replace('_', '').capitalize()
        R[f'SigMean{key}'] = pct(T[k].mean(), 1)
        R[f'SigMin{key}'] = pct(T[k].min(), 1)
        R[f'SigMax{key}'] = pct(T[k].max(), 1)
    R['SigPooledEtoe'] = pct(T['e2e'].mul(T['n']).sum() / T['n'].sum(), 2)
    LO = pd.concat(letter_ok).groupby('label')['ok'].mean().sort_values()
    R['SigWorstLetters'] = ', '.join(f'{c} ({100 * v:.1f}\\%)' for c, v in LO.head(5).items())
    write_dat('surrey_letters_loso', pd.DataFrame({'label': LO.index, 'acc': 100 * LO.values}))
    write_dat('surrey_signers', T.assign(idx=np.arange(1, len(T) + 1)))
    R.save()
    print('saved', len(R.R), 'values', flush=True)


# (d)
def part_d():
    R = Results('s11d_threshold')
    alpha = load_landmarks('alphabet')
    D = letters_only(alpha)
    y, X = labels(D), FEATURES['norm'](as_L(D))
    tr, te = protocol_b(y)
    mlp = pickle.load(open(os.path.join(RES, 'mlp_protoB.pkl'), 'rb'))
    val = tr[mlp.val_idx]
    conf_val = mlp.predict_proba(X[val]).max(1)                # uncalibrated MSP, as used by the decoder
    pt = mlp.predict_proba(X[te])
    conf_te, ok_te = pt.max(1), pt.argmax(1) == y[te]
    unk = {}
    for c in ('del', 'space'):
        u = alpha[(alpha['label'] == c) & (alpha['method'] == 'default')]
        unk[c] = mlp.predict_proba(FEATURES['norm'](as_L(u))).max(1)
        R[f'Thr{c.capitalize()}N'] = f'{len(u):,}'
    rng = np.random.default_rng(SEED)
    ia, ib = rng.integers(len(te), size=(2, 20000))
    keep = y[te][ia] != y[te][ib]
    ia, ib = ia[keep][:5000], ib[keep][:5000]
    trans = {a: mlp.predict_proba((1 - a) * X[te][ia] + a * X[te][ib]).max(1) for a in (0.25, 0.5)}

    taus = np.unique(np.r_[np.linspace(0.30, 0.999, 1400), 0.9999])
    for sel, ev in (('del', 'space'), ('space', 'del')):
        J = [(conf_val >= t).mean() - (unk[sel] >= t).mean() for t in taus]   # Youden's J
        t = float(taus[int(np.argmax(J))])
        tag = f'{sel.capitalize()}To{ev.capitalize()}'
        R[f'Thr{tag}Tau'] = f'{t:.3f}'
        R[f'Thr{tag}SelReject'] = pct((unk[sel] < t).mean())
        R[f'Thr{tag}EvalReject'] = pct((unk[ev] < t).mean())
        R[f'Thr{tag}Coverage'] = pct((conf_te >= t).mean())
        R[f'Thr{tag}SelAcc'] = pct(ok_te[conf_te >= t].mean())
        for a, c in trans.items():
            R[f'Thr{tag}Trans{int(a * 100)}'] = pct((c < t).mean())
        print(tag, f'tau={t:.3f}', 'rej sel', R[f'Thr{tag}SelReject'], 'rej eval', R[f'Thr{tag}EvalReject'],
              'coverage', R[f'Thr{tag}Coverage'], 'selacc', R[f'Thr{tag}SelAcc'],
              'trans50', R[f'Thr{tag}Trans50'], flush=True)
    # reference: the decoder's fixed tau = 0.8, per non-letter class
    for c in unk:
        R[f'ThrFixed{c.capitalize()}Reject'] = pct((unk[c] < TAU).mean())
    for a, c in trans.items():
        R[f'ThrFixedTrans{int(a * 100)}'] = pct((c < TAU).mean())
    R.save()
    print('saved', len(R.R), 'values', flush=True)


if __name__ == '__main__':
    parts = sys.argv[1] if len(sys.argv) > 1 else 'ad'
    if 'a' in parts:
        part_a()
    if 'd' in parts:
        part_d()
