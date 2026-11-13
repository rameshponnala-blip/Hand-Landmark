"""Step 2: recognition experiments on the Kaggle ASL Alphabet.

Protocol A (5-fold CV) and Protocol B (80/20 hold-out) for all landmark models,
ablations, the feature-set comparison (G2), robustness, joint importance (G4),
detection recovery (G5), latency and int8 quantisation.

Usage:  .venv\\Scripts\\python.exe experiments\\s2_main.py
"""
import os
import pickle
import shutil
import sys
import time

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from common import (FEATURES, FINGER_OF, K, LETTERS, MLPClassifier, OUT, RES, SEED, Results, as_L,  # noqa: E402
                    cpu_name, feat_norm, ints, labels, letters_only, load_landmarks, mcnemar, pct, pm,
                    pfmt, protocol_b, rotate_xy, scores, write_dat)

SPECS = {   # name: (feature set, constructor)
    'SVM':        ('norm',   lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma='scale'))),
    'RF':         ('norm',   lambda: RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=SEED)),
    'LR':         ('norm',   lambda: make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=3000))),
    'KNN':        ('norm',   lambda: KNeighborsClassifier(n_neighbors=5)),
    'MLPRaw':     ('raw',    lambda: MLPClassifier()),
    'MLP':        ('norm',   lambda: MLPClassifier()),                       # proposed
    'AblCenter':  ('center', lambda: MLPClassifier()),
    'AblNoDrop':  ('norm',   lambda: MLPClassifier(dropout=0.0)),
    'AblSmall':   ('norm',   lambda: MLPClassifier(hidden=(64, 32))),
    'AblLarge':   ('norm',   lambda: MLPClassifier(hidden=(256, 128))),
    'FeatRot':    ('rot',    lambda: MLPClassifier()),
    'FeatAngles': ('angles', lambda: MLPClassifier()),
    'FeatHybrid': ('hybrid', lambda: MLPClassifier()),
}


def fit_predict(make, Xtr, ytr, Xte):
    clf = make().fit(Xtr, ytr)
    return clf.predict(Xte)


def fit_keep(make, Xtr, ytr):
    return make().fit(Xtr, ytr)


def augment(L, rng, max_deg=25, sigma=0.005):
    """One rotated (|theta| <= max_deg) and jittered copy of every sample."""
    rot = rotate_xy(L, np.deg2rad(rng.uniform(-max_deg, max_deg, len(L))))
    return rot + rng.normal(0, sigma, rot.shape).astype(np.float32)


def main():
    R = Results('s2_main')
    t_start = time.time()
    alpha = load_landmarks('alphabet')
    allL = alpha[alpha['label'].isin(LETTERS)]
    D = letters_only(alpha)
    y, L = labels(D), as_L(D)
    F = {m: f(L) for m, f in FEATURES.items()}

    # dataset stats
    per = allL.groupby('label').agg(n=('path', 'size'), det=('method', lambda s: (s == 'default').sum()))
    per['rate'] = per['det'] / per['n']
    R['NumImages'], R['NumDetected'] = ints(len(allL)), ints(len(D))
    R['DetRate'] = pct(len(D) / len(allL))
    R['MinPerClass'], R['MaxPerClass'] = ints(per['det'].min()), ints(per['det'].max())
    R['LowDetClasses'] = ', '.join(f'{c} ({100 * r:.1f}\\%)' for c, r in per['rate'].nsmallest(3).items())
    R['HighDetClasses'] = ', '.join(f'{c} ({100 * r:.1f}\\%)' for c, r in per['rate'].nlargest(3).items())
    R['ImgRes'] = '{}$\\times${}'.format(*allL[['w', 'h']].mode().iloc[0])
    write_dat('detrate', per.reset_index()[['label', 'rate']].assign(rate=lambda d: 100 * d['rate']))

    # Protocol A: 5-fold CV
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(skf.split(L, y))
    jobs = [(n, k) for n in SPECS for k in range(5)]
    out = Parallel(n_jobs=12, verbose=5)(
        delayed(fit_predict)(SPECS[n][1], F[SPECS[n][0]][folds[k][0]], y[folds[k][0]], F[SPECS[n][0]][folds[k][1]])
        for n, k in jobs)
    cv = {n: {'acc': [], 'f1': []} for n in SPECS}
    for (n, k), p in zip(jobs, out):
        s = scores(y[folds[k][1]], p)
        cv[n]['acc'].append(s['acc'])
        cv[n]['f1'].append(s['f1'])
    for n in SPECS:
        R[f'{n}CVAcc'], R[f'{n}CVF1'] = pm(cv[n]['acc']), pm(cv[n]['f1'])
    pickle.dump(cv, open(os.path.join(RES, 'cv_scores.pkl'), 'wb'))
    print('Protocol A done', flush=True)

    # Protocol B: hold-out
    tr, te = protocol_b(y)
    R['NumTrainB'], R['NumTestB'] = ints(len(tr)), ints(len(te))
    rng = np.random.default_rng(SEED)
    Laug = np.concatenate([L[tr], augment(L[tr], rng)])
    yaug = np.concatenate([y[tr], y[tr]])
    names = list(SPECS)
    fitted = Parallel(n_jobs=12)(delayed(fit_keep)(SPECS[n][1], F[SPECS[n][0]][tr], y[tr]) for n in names)
    M = dict(zip(names, fitted))
    M['MLPAug'] = MLPClassifier().fit(feat_norm(Laug), yaug)
    mode = {n: SPECS[n][0] for n in names} | {'MLPAug': 'norm'}
    preds = {n: M[n].predict(F[mode[n]][te]) for n in M}
    for n, p in preds.items():
        s = scores(y[te], p)
        R[f'{n}TestAcc'], R[f'{n}TestF1'] = pct(s['acc']), pct(s['f1'])
        if n == 'MLP':
            R['MLPTestPrec'], R['MLPTestRec'] = pct(s['prec']), pct(s['rec'])
    np.savez(os.path.join(RES, 'protoB_preds.npz'), te=te, y=y[te], **{n: preds[n] for n in preds})
    pickle.dump(M['MLP'], open(os.path.join(RES, 'mlp_protoB.pkl'), 'wb'))
    for n, hid in [('MLP', (128, 64)), ('AblSmall', (64, 32)), ('AblLarge', (256, 128))]:
        R[f'{n}Params'] = ints(MLPClassifier(hidden=hid).n_params(63))
    R['LRParams'] = ints(63 * K + K)
    mlp = M['MLP']
    R['MLPEpochs'] = str(mlp.best_epoch)
    R['MLPEpochsRun'] = str(len(mlp.history['loss']))
    h = pd.DataFrame(mlp.history)
    write_dat('training_curves', h.assign(epoch=np.arange(1, len(h) + 1),
                                          acc=100 * h['acc'], val_acc=100 * h['val_acc']))
    for other in ['SVM', 'RF', 'KNN', 'LR', 'MLPRaw', 'FeatHybrid']:
        b, c, p = mcnemar(preds['MLP'] == y[te], preds[other] == y[te])
        R[f'McNemar{other}'] = pfmt(p)
        R[f'McNemar{other}b'], R[f'McNemar{other}c'] = str(b), str(c)

    # confusion matrix and per-class analysis
    cm = confusion_matrix(y[te], preds['MLP'], labels=range(K))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    write_dat('confusion', pd.DataFrame([(j, i, 100 * cmn[i, j]) for i in range(K) for j in range(K)],
                                        columns=['x', 'y', 'c']))
    _, _, f1c, _ = precision_recall_fscore_support(y[te], preds['MLP'], labels=range(K), zero_division=0)
    order = np.argsort(f1c)
    R['WorstClasses'] = ', '.join(f'{LETTERS[i]} ({100 * f1c[i]:.2f}\\%)' for i in order[:3])
    R['NumPerfectClasses'] = str(int((f1c >= 0.99995).sum()))
    R['NumAbove99Classes'] = str(int((f1c >= 0.99).sum()))
    off = cmn.copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:5], off.shape))[0]
    R['ConfusedRows'] = ' \\\\ '.join(f'{LETTERS[i]} & {LETTERS[j]} & {cm[i, j]} & {100 * cmn[i, j]:.2f}'
                                      for i, j in top)
    R['NumTestErrors'] = str(int((preds['MLP'] != y[te]).sum()))
    write_dat('perclass_f1', pd.DataFrame({'label': LETTERS, 'f1': 100 * f1c}))

    # robustness: noise and rotation
    rob_models = {'Raw': 'MLPRaw', 'Norm': 'MLP', 'Aug': 'MLPAug', 'Rot': 'FeatRot',
                  'Hybrid': 'FeatHybrid', 'Angles': 'FeatAngles'}
    Lte = L[te]

    def acc_under(Lp):
        return {k: (M[m].predict(FEATURES[mode[m]](Lp)) == y[te]).mean() for k, m in rob_models.items()}

    conds = {'Clean': Lte}
    for i, sg in enumerate([0.005, 0.01, 0.02], 1):
        r2 = np.random.default_rng(100 + i)
        conds[f'Noise{i}'] = Lte + r2.normal(0, sg, Lte.shape).astype(np.float32)
    curve = []
    for deg in range(0, 50, 5):
        r2 = np.random.default_rng(200 + deg)
        sign = r2.choice([-1, 1], len(Lte))
        a = acc_under(rotate_xy(Lte, np.deg2rad(deg) * sign))
        curve.append({'deg': deg, **{k: 100 * v for k, v in a.items()}})
        if deg in (10, 20, 30, 45):
            for k, v in a.items():
                R[f'RobRot{deg}{k}'] = pct(v)
    write_dat('rotation_curve', pd.DataFrame(curve))
    for c, Lp in conds.items():
        for k, v in acc_under(Lp).items():
            R[f'Rob{c}{k}'] = pct(v)
    print('robustness done', flush=True)

    # orientation-pair analysis (G2): errors within H/U, K/P, G/Q per feature set
    pairs = [('H', 'U'), ('K', 'P'), ('G', 'Q')]
    for k, m in rob_models.items():
        p = preds[m]
        tot = 0
        for a, b in pairs:
            ia, ib = LETTERS.index(a), LETTERS.index(b)
            tot += np.sum((y[te] == ia) & (p == ib)) + np.sum((y[te] == ib) & (p == ia))
        n = sum(np.sum(y[te] == LETTERS.index(c)) for pr in pairs for c in pr)
        R[f'PairErr{k}'] = pct(tot / n)
        R[f'PairErrN{k}'] = str(int(tot))

    # G4: grouped permutation importance
    Xte = F['norm'][te]
    base = (mlp.predict(Xte) == y[te])
    base_rec = np.array([base[y[te] == c].mean() for c in range(K)])
    rng = np.random.default_rng(SEED)
    drop_j = np.zeros((21, K))
    drop_all = np.zeros(21)
    for j in range(21):
        accs, recs = [], []
        for _ in range(5):
            Xp = Xte.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, 3 * j:3 * j + 3] = Xp[perm, 3 * j:3 * j + 3]
            ok = mlp.predict(Xp) == y[te]
            accs.append(base.mean() - ok.mean())
            recs.append(base_rec - np.array([ok[y[te] == c].mean() for c in range(K)]))
        drop_all[j], drop_j[j] = np.mean(accs), np.mean(recs, 0)
    write_dat('importance', pd.DataFrame([(j, c, 100 * drop_j[j, c]) for c in range(K) for j in range(21)],
                                         columns=['x', 'y', 'c']))
    write_dat('importance_joint', pd.DataFrame({'joint': range(21), 'drop': 100 * drop_all,
                                                'finger': [FINGER_OF[j] for j in range(21)]}))
    top3 = np.argsort(-drop_all)[:3]
    R['TopJoints'] = ', '.join(f'{j} ({FINGER_OF[j].lower()}, {100 * drop_all[j]:.1f}~pp)' for j in top3)
    fingers = {}
    for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Little']:
        cols = [c for j in range(21) if FINGER_OF[j] == f for c in range(3 * j, 3 * j + 3)]
        d = []
        for _ in range(5):
            Xp = Xte.copy()
            Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
            d.append(base.mean() - (mlp.predict(Xp) == y[te]).mean())
        fingers[f] = np.mean(d)
        R[f'FingerDrop{f}'] = pct(fingers[f], 1)
    fist = [LETTERS.index(c) for c in 'AEMNST']
    rest = [c for c in range(K) if c not in fist]
    thumb = [j for j in range(21) if FINGER_OF[j] == 'Thumb']
    R['ThumbDropFist'] = pct(drop_j[thumb][:, fist].mean(), 1)
    R['ThumbDropRest'] = pct(drop_j[thumb][:, rest].mean(), 1)
    print('importance done', flush=True)

    # G5: detection recovery
    for name in ['alphabet', 'test', 'surrey']:
        try:
            df = load_landmarks(name)
        except FileNotFoundError:
            continue
        df = df[df['label'].isin(LETTERS)]
        n = len(df)
        vc = df['method'].value_counts()
        tag = name.capitalize()
        R[f'Det{tag}N'] = ints(n)
        R[f'Det{tag}Default'] = pct(vc.get('default', 0) / n)
        R[f'Det{tag}All'] = pct((n - vc.get('none', 0) - vc.get('unreadable', 0)) / n)
        for m in ['pad', 'flip', 'upscale', 'lowconf']:
            R[f'Det{tag}{m.capitalize()}'] = pct(vc.get(m, 0) / n)
    rec = alpha[alpha['label'].isin(LETTERS) & alpha['method'].isin(['pad', 'flip', 'upscale', 'lowconf'])]
    if len(rec):
        okr = mlp.predict(feat_norm(as_L(rec))) == labels(rec)
        R['RecoveredN'], R['RecoveredAcc'] = ints(len(rec)), pct(okr.mean())
        acc_te = (preds['MLP'] == y[te]).mean()
        R['EndToEndBefore'] = pct(len(D) / len(allL) * acc_te)
        R['EndToEndAfter'] = pct((len(D) * acc_te + okr.sum()) / len(allL))
    else:
        R['EndToEndBefore'] = pct(len(D) / len(allL) * (preds['MLP'] == y[te]).mean())

    # latency and int8
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    onnx32, onnx8 = os.path.join(RES, 'mlp_fp32.onnx'), os.path.join(RES, 'mlp_int8.onnx')
    torch.onnx.export(mlp.model, torch.zeros(1, 63), onnx32, input_names=['x'], output_names=['logits'],
                      dynamic_axes={'x': {0: 'n'}}, dynamo=False)
    quantize_dynamic(onnx32, onnx8, weight_type=QuantType.QInt8)

    def ort_session(path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])

    def ort_latency(sess, X, n=2000):
        ts = []
        for i in range(n):
            x = X[i % len(X)][None]
            t0 = time.perf_counter()
            sess.run(None, {'x': x})
            ts.append(time.perf_counter() - t0)
        return 1000 * np.median(ts[100:])

    s32, s8 = ort_session(onnx32), ort_session(onnx8)
    R['MLPLat'] = f'{ort_latency(s32, Xte):.3f}'
    R['MLPInt8Lat'] = f'{ort_latency(s8, Xte):.3f}'
    R['MLPSize'] = f'{os.path.getsize(onnx32) / 1024:.1f}'
    R['MLPInt8Size'] = f'{os.path.getsize(onnx8) / 1024:.1f}'
    p8 = s8.run(None, {'x': Xte})[0].argmax(1)
    R['MLPInt8Acc'] = pct((p8 == y[te]).mean())
    R['MLPInt8Agree'] = pct((p8 == preds['MLP']).mean())
    ts = []
    for i in range(2000):
        l1 = L[te[i % len(te)]][None]
        t0 = time.perf_counter()
        feat_norm(l1)
        ts.append(time.perf_counter() - t0)
    R['FeatLat'] = f'{1000 * np.median(ts[100:]):.3f}'
    for n in ['SVM', 'RF', 'KNN', 'LR']:
        X1 = F[mode[n]][te]
        ts = []
        for i in range(300):
            t0 = time.perf_counter()
            M[n].predict(X1[i % len(X1)][None])
            ts.append(time.perf_counter() - t0)
        R[f'{n}Lat'] = f'{1000 * np.median(ts[20:]):.3f}'
        R[f'{n}Size'] = f'{len(pickle.dumps(M[n])) / 1024:.1f}'
    R['MLPRawLat'], R['MLPRawSize'] = R['MLPLat'], R['MLPSize']

    # MediaPipe latency (single process, warm), dataset resolution and 640x480
    import mediapipe.python.solutions.hands as mp_hands
    sample = D['path'].sample(min(300, len(D)), random_state=SEED).tolist()
    for tag, size in [('MPLat', None), ('MPLatVGA', (640, 480))]:
        ts = []
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
            for p in sample:
                img = cv2.imread(p)
                if size:
                    img = cv2.resize(img, size)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                t0 = time.perf_counter()
                hands.process(rgb)
                ts.append(time.perf_counter() - t0)
        R[tag] = f'{1000 * np.median(ts[30:]):.1f}'
    ts = []
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:   # video mode: tracking between frames
        frames = [cv2.cvtColor(cv2.resize(cv2.imread(p), (640, 480)), cv2.COLOR_BGR2RGB)
                  for p in sorted(D[D['label'] == 'B']['path'])[:300]]
        for f in frames:
            t0 = time.perf_counter()
            hands.process(f)
            ts.append(time.perf_counter() - t0)
    R['MPLatTrack'] = f'{1000 * np.median(ts[30:]):.1f}'
    tot = float(R['MPLat']) + float(R['FeatLat']) + float(R['MLPLat'])
    R['PipeLat'], R['PipeFPS'] = f'{tot:.1f}', f'{1000 / tot:.0f}'
    totv = float(R['MPLatTrack']) + float(R['FeatLat']) + float(R['MLPLat'])
    R['PipeLatVideo'], R['PipeFPSVideo'] = f'{totv:.1f}', f'{1000 / totv:.0f}'
    R['Hardware'] = cpu_name().replace('(R)', '').replace('(TM)', '').replace('12th Gen ', '')
    R['NumCores'] = str(os.cpu_count())

    # example images for Fig. 5 (landmarks saved next to them for make_figures.py)
    qdir = os.path.join(OUT, 'qualitative')
    os.makedirs(qdir, exist_ok=True)
    ok = preds['MLP'] == y[te]
    rng = np.random.default_rng(7)
    right = rng.choice(np.flatnonzero(ok), 6, replace=False)
    wrong = rng.choice(np.flatnonzero(~ok), min(2, int((~ok).sum())), replace=False)
    for q, i in enumerate(list(right) + list(wrong), 1):
        row = D.iloc[te[i]]
        shutil.copy(row['path'], os.path.join(qdir, f'q{q}.jpg'))
        S = max(row['w'], row['h'])
        lm = L[te[i]]
        with open(os.path.join(qdir, f'q{q}_landmarks.csv'), 'w') as fh:   # unit box, y pointing up
            fh.write('joint,x,y\n')
            for j in range(21):
                fh.write(f'{j},{lm[j, 0] * S / row["w"]:.4f},{1 - lm[j, 1] * S / row["h"]:.4f}\n')
        R[f'QualTrue{"ABCDEFGH"[q - 1]}'] = LETTERS[y[te[i]]]
        R[f'QualPred{"ABCDEFGH"[q - 1]}'] = LETTERS[preds['MLP'][i]]

    R['RuntimeMain'] = f'{(time.time() - t_start) / 60:.0f}'
    R.save()
    print('saved', len(R.R), 'values', flush=True)


if __name__ == '__main__':
    main()
