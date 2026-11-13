"""Step 4 (gap G3): calibration, selective prediction and rejection of non-letter hand postures.

Uses the Protocol-B MLP saved by s2_main.py. Temperature scaling is fitted on the
MLP's own validation split. Unknown postures are the 'del' and 'space' classes of
the Kaggle ASL Alphabet, which are deliberate hand shapes that are not letters.

Usage:  .venv\\Scripts\\python.exe experiments\\s4_calibration.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import (RES, Results, as_L, feat_norm, labels, letters_only, load_landmarks, pct,  # noqa: E402
                    protocol_b, write_dat)

TAU = 0.8


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(z, y, T):
    p = softmax(z, T)
    return -np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)))


def ece(conf, ok, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
            rows.append({'mid': (lo + hi) / 2, 'conf': conf[m].mean(), 'acc': ok[m].mean(), 'n': int(m.sum())})
    return e, pd.DataFrame(rows)


def fpr_at_tpr(known, unknown, tpr=0.95):
    t = np.quantile(known, 1 - tpr)
    return (unknown >= t).mean()


def main():
    R = Results('s4_calibration')
    alpha = load_landmarks('alphabet')
    D = letters_only(alpha)
    y, X = labels(D), feat_norm(as_L(D))
    tr, te = protocol_b(y)
    mlp = pickle.load(open(os.path.join(RES, 'mlp_protoB.pkl'), 'rb'))
    val = tr[mlp.val_idx]                                    # same validation split used for early stopping
    zv, zt = mlp.logits(X[val]), mlp.logits(X[te])
    T = minimize_scalar(lambda t: nll(zv, y[val], t), bounds=(0.05, 10), method='bounded').x
    R['Temperature'] = f'{T:.2f}'

    ok = zt.argmax(1) == y[te]
    for tag, t in (('Raw', 1.0), ('Cal', T)):
        p = softmax(zt, t)
        conf = p.max(1)
        e, rel = ece(conf, ok)
        R[f'ECE{tag}'] = pct(e)
        R[f'NLL{tag}'] = f'{nll(zt, y[te], t):.4f}'
        onehot = np.eye(p.shape[1])[y[te]]
        R[f'Brier{tag}'] = f'{np.mean(np.sum((p - onehot) ** 2, 1)):.4f}'
        write_dat(f'reliability_{tag.lower()}', rel.assign(conf=100 * rel['conf'], acc=100 * rel['acc'],
                                                           mid=100 * rel['mid']))

    # risk-coverage curve on known letters (calibrated confidence)
    conf_k = softmax(zt, T).max(1)
    order = np.argsort(-conf_k)
    cov = np.arange(1, len(order) + 1) / len(order)
    sel_acc = np.cumsum(ok[order]) / np.arange(1, len(order) + 1)
    idx = np.unique(np.linspace(0, len(order) - 1, 200).astype(int))
    write_dat('risk_coverage', pd.DataFrame({'coverage': 100 * cov[idx], 'risk': 100 * (1 - sel_acc[idx])}))
    for c in (0.90, 0.95, 0.99):
        R[f'SelAcc{int(c * 100)}'] = pct(sel_acc[int(c * len(order)) - 1])

    # unknown hand postures: 'del' and 'space'
    unk = alpha[alpha['label'].isin(['del', 'space']) & (alpha['method'] == 'default')]
    zu = mlp.logits(feat_norm(as_L(unk)))
    R['UnknownN'] = f'{len(unk):,}'
    scores_k, scores_u = {}, {}
    for tag, t in (('MSP', 1.0), ('Cal', T)):
        scores_k[tag], scores_u[tag] = softmax(zt, t).max(1), softmax(zu, t).max(1)
    scores_k['Energy'] = np.log(np.exp(zt - zt.max(1, keepdims=True)).sum(1)) + zt.max(1)
    scores_u['Energy'] = np.log(np.exp(zu - zu.max(1, keepdims=True)).sum(1)) + zu.max(1)
    for tag in scores_k:
        lab = np.r_[np.ones(len(scores_k[tag])), np.zeros(len(scores_u[tag]))]
        R[f'AUROC{tag}'] = pct(roc_auc_score(lab, np.r_[scores_k[tag], scores_u[tag]]))
        R[f'FPR95{tag}'] = pct(fpr_at_tpr(scores_k[tag], scores_u[tag]))

    # operating points: the decoder's tau = 0.8 and a tau tuned on validation for 99.5% selective accuracy
    pv = softmax(zv, T)
    cv, okv = pv.max(1), pv.argmax(1) == y[val]
    taus = np.linspace(0.3, 0.999, 500)
    good = [t for t in taus if (cv >= t).any() and okv[cv >= t].mean() >= 0.995]
    tau_star = good[0] if good else 0.99
    R['TauStar'] = f'{tau_star:.2f}'
    for tag, t_op, Tuse in (('Fixed', TAU, 1.0), ('Tuned', tau_star, T)):
        ck, cu = softmax(zt, Tuse).max(1), softmax(zu, Tuse).max(1)
        acc_m = ck >= t_op
        R[f'Op{tag}Coverage'] = pct(acc_m.mean())
        R[f'Op{tag}SelAcc'] = pct(ok[acc_m].mean())
        R[f'Op{tag}UnkReject'] = pct((cu < t_op).mean())

    # does rejection help under dataset shift? (varied-background test set)
    try:
        ext = letters_only(load_landmarks('test'), ('default', 'pad', 'flip', 'upscale', 'lowconf'))
        ze = mlp.logits(feat_norm(as_L(ext)))
        oke = ze.argmax(1) == labels(ext)
        ce = softmax(ze, T).max(1)
        R['ShiftAcc'] = pct(oke.mean())
        R['ShiftECE'] = pct(ece(ce, oke)[0])
        m = ce >= tau_star
        R['ShiftCoverage'], R['ShiftSelAcc'] = pct(m.mean()), pct(oke[m].mean() if m.any() else 0)
    except FileNotFoundError:
        pass
    R.save()
    print('saved', len(R.R), 'values', flush=True)


if __name__ == '__main__':
    main()
