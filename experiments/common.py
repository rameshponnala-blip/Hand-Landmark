"""Shared paths, landmark features, the MLP classifier and result writers.

Every experiment script imports this module, so feature definitions and
splits are identical across scripts.
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KAGGLE = os.path.join(ROOT, 'data', 'kaggle', 'datasets')
CACHE = os.path.join(ROOT, 'data', 'cache')
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'outputs')
PDATA = os.path.join(OUT, 'figure_data')
LMK = os.path.join(ROOT, 'data', 'landmarks')
for _d in (CACHE, RES, PDATA):
    os.makedirs(_d, exist_ok=True)

SEED = 42
LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
K = len(LETTERS)
COLS = [f'{a}{k}' for k in range(21) for a in 'xyz']
TIPS = [4, 8, 12, 16, 20]
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
FINGER_OF = {0: 'Wrist', **{j: f for f, ch in zip(['Thumb', 'Index', 'Middle', 'Ring', 'Little'], CHAINS)
                             for j in ch[1:]}}


# data
def dataset_root(handle):
    """Folder that kagglehub extracted for an 'owner/name' handle (latest version)."""
    versions = sorted(glob.glob(os.path.join(KAGGLE, *handle.split('/'), 'versions', '*')),
                      key=lambda p: int(os.path.basename(p)))
    if not versions:
        raise FileNotFoundError(f'{handle} not downloaded; run experiments/download_data.py')
    return versions[-1]


def load_landmarks(name):
    """Landmark table written by s1_extract.py (one row per image, sorted by path).

    With the environment variable PILOT=1 every dataset is replaced by the small
    test set, which lets the scripts be smoke-tested in a minute.
    """
    if os.environ.get('PILOT'):
        name = 'test'
    pkl = os.path.join(CACHE, f'landmarks_{name}.pkl')
    if os.path.exists(pkl):
        return pd.read_pickle(pkl)
    # no cache yet: use the csv files from the repo (paths in them are relative to KAGGLE)
    csv = os.path.join(LMK, f'landmarks_{name}.csv.gz')
    if not os.path.exists(csv):
        raise FileNotFoundError(f'{pkl} not found, run experiments/s1_extract.py first')
    df = pd.read_csv(csv, dtype={'dhash': 'uint64', 'handedness': str}, float_precision='round_trip',
                     keep_default_na=False, na_values={c: [''] for c in ('mp_ms', 'score', *COLS)})
    df['path'] = [os.path.join(KAGGLE, *p.split('/')) for p in df['path']]
    return df


def letters_only(df, methods=('default',)):
    """Rows whose label is a letter A-Z and whose hand was found by one of `methods`."""
    out = df[df['label'].isin(LETTERS) & df['method'].isin(methods)]
    return out.reset_index(drop=True)


def labels(df):
    return df['label'].map({c: i for i, c in enumerate(LETTERS)}).to_numpy()


def as_L(df):
    return df[COLS].to_numpy(np.float32).reshape(-1, 21, 3)


def protocol_b(y):
    """Fixed stratified 80/20 hold-out split shared by all scripts."""
    return train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)


# features
def _palm(L):
    return np.maximum(np.linalg.norm(L[:, 9, :2] - L[:, 0, :2], axis=1), 1e-6)


def inplane_angle(L):
    """Angle of the wrist->middle-MCP vector relative to 'straight up' in the image."""
    v = L[:, 9, :2] - L[:, 0, :2]
    return np.arctan2(v[:, 0], -v[:, 1])


def rotate_xy(L, phi):
    """Rotate landmarks in the image plane about the wrist by phi (radians, per sample)."""
    phi = np.broadcast_to(phi, (len(L),))
    c, s = np.cos(phi)[:, None], np.sin(phi)[:, None]
    out = L.copy()
    x, y = L[..., 0] - L[:, :1, 0], L[..., 1] - L[:, :1, 1]
    out[..., 0] = L[:, :1, 0] + c * x - s * y
    out[..., 1] = L[:, :1, 1] + s * x + c * y
    return out


def feat_raw(L):
    return L.reshape(len(L), -1)


def feat_center(L):
    return (L - L[:, :1]).reshape(len(L), -1)


def feat_norm(L):
    """Proposed descriptor: wrist-centred and divided by palm size (Eqs. 1-3)."""
    return ((L - L[:, :1]) / _palm(L)[:, None, None]).reshape(len(L), -1).astype(np.float32)


def feat_rot(L):
    """Like feat_norm, but first rotated so that the palm axis points straight up."""
    return feat_norm(rotate_xy(L, -inplane_angle(L)))


def orientation(L):
    """In-plane angle (cos, sin) and the unit palm normal."""
    th = inplane_angle(L)
    n = np.cross(L[:, 5] - L[:, 0], L[:, 17] - L[:, 0])
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
    return np.column_stack([np.cos(th), np.sin(th), n]).astype(np.float32)


def feat_hybrid(L):
    """Rotation-normalised shape plus an explicit orientation code."""
    return np.hstack([feat_rot(L), orientation(L)])


def _angle(u, v):
    cos = (u * v).sum(-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-9)
    return np.arccos(np.clip(cos, -1, 1))


def feat_angles(L):
    """34 geometric invariants: 15 joint angles, 4 finger spreads, 15 normalised distances."""
    f = []
    for ch in CHAINS:
        for a, b, c in zip(ch[:-2], ch[1:-1], ch[2:]):
            f.append(_angle(L[:, a] - L[:, b], L[:, c] - L[:, b]))
    dirs = [L[:, ch[-1]] - L[:, ch[1]] for ch in CHAINS]
    f += [_angle(dirs[i], dirs[i + 1]) for i in range(4)]
    s = np.maximum(np.linalg.norm(L[:, 9] - L[:, 0], axis=1), 1e-6)
    for i in range(5):
        for j in range(i + 1, 5):
            f.append(np.linalg.norm(L[:, TIPS[i]] - L[:, TIPS[j]], axis=1) / s)
    f += [np.linalg.norm(L[:, t] - L[:, 0], axis=1) / s for t in TIPS]
    return np.column_stack(f).astype(np.float32)


FEATURES = {'raw': feat_raw, 'center': feat_center, 'norm': feat_norm,
            'rot': feat_rot, 'angles': feat_angles, 'hybrid': feat_hybrid}


# MLP
class MLPClassifier:
    """d-128-64-K perceptron (PyTorch, CPU) with early stopping on a 10% validation split."""

    def __init__(self, hidden=(128, 64), dropout=0.2, lr=1e-3, batch=64, epochs=100,
                 patience=10, seed=SEED, threads=2):
        self.hidden, self.dropout, self.lr, self.batch = hidden, dropout, lr, batch
        self.epochs, self.patience, self.seed, self.threads = epochs, patience, seed, threads

    def _build(self, d):
        import torch.nn as nn
        layers, prev = [], d
        for i, h in enumerate(self.hidden):
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if i == 0 and self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, K))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        ia, iv = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=self.seed)
        self.val_idx = iv
        Xa, ya = torch.tensor(X[ia]), torch.tensor(y[ia])
        Xv, yv = torch.tensor(X[iv]), torch.tensor(y[iv])
        self.model = self._build(X.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lossf = torch.nn.CrossEntropyLoss()
        best, best_state, wait = np.inf, None, 0
        self.history = {k: [] for k in ('loss', 'acc', 'val_loss', 'val_acc')}
        for ep in range(self.epochs):
            self.model.train()
            perm = torch.tensor(rng.permutation(len(ya)))
            tl, tc = 0.0, 0
            for s in range(0, len(perm), self.batch):
                b = perm[s:s + self.batch]
                out = self.model(Xa[b])
                loss = lossf(out, ya[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl += loss.item() * len(b)
                tc += (out.argmax(1) == ya[b]).sum().item()
            self.model.eval()
            with torch.no_grad():
                ov = self.model(Xv)
                vl = lossf(ov, yv).item()
                va = (ov.argmax(1) == yv).float().mean().item()
            for k, v in zip(self.history, (tl / len(ya), tc / len(ya), vl, va)):
                self.history[k].append(v)
            if vl < best - 1e-5:
                best, wait = vl, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.best_epoch = ep + 1
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def logits(self, X):
        import torch
        with torch.no_grad():
            return self.model(torch.tensor(np.asarray(X, np.float32))).numpy()

    def predict_proba(self, X, T=1.0):
        z = self.logits(X) / T
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.logits(X).argmax(1)

    def n_params(self, d=63):
        return sum(p.numel() for p in self._build(d).parameters())


# metrics
def scores(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average='macro', zero_division=0)
    return {'acc': accuracy_score(y, p), 'prec': pr, 'rec': rc, 'f1': f1}


def mcnemar(ok_a, ok_b):
    """Exact McNemar test on paired correctness vectors; returns (b, c, p)."""
    from scipy.stats import binomtest
    b, c = int(np.sum(ok_a & ~ok_b)), int(np.sum(~ok_a & ok_b))
    return b, c, (binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)


def pct(x, nd=2):
    return f'{100 * x:.{nd}f}'


def pm(a, nd=2):
    return f'{100 * np.mean(a):.{nd}f} $\\pm$ {100 * np.std(a):.{nd}f}'


def pfmt(p):
    return '$<$0.001' if p < 1e-3 else f'{p:.3f}'


def ints(n):
    return f'{int(n):,}'


# outputs
class Results:
    """Values reported by one script (as strings), kept in results/<name>.json."""

    def __init__(self, name):
        self.path = os.path.join(RES, f'{name}.json')
        self.R = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def __setitem__(self, k, v):
        assert k.isalnum(), k
        self.R[k] = str(v)

    def __getitem__(self, k):
        return self.R[k]

    def save(self):
        json.dump(self.R, open(self.path, 'w'), indent=1)


def write_dat(name, df):
    """Data for a figure, written to outputs/figure_data/<name>.dat."""
    df.to_csv(os.path.join(PDATA, f'{name}.dat'), sep=' ', index=False, float_format='%.5g')


def cpu_name():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
        return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
    except Exception:
        import platform
        return platform.processor()


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t
