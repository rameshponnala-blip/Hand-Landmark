"""Plots the figures of the paper from outputs/figure_data into outputs/figures (PNG).

Run after the experiment scripts. Figures whose data file is missing are skipped.
"""
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import LETTERS, ROOT  # noqa: E402

DATA = os.path.join(ROOT, 'outputs', 'figure_data')
QUAL = os.path.join(ROOT, 'outputs', 'qualitative')
FIGS = os.path.join(ROOT, 'outputs', 'figures')
os.makedirs(FIGS, exist_ok=True)

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
INK, INK2, MUTED, GRID, AXIS, BG = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
SEQ = LinearSegmentedColormap.from_list('seq', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
# same colour for a feature type in all plots
METHOD = {'raw': C[0], 'norm': C[1], 'rot': C[2], 'angles': C[3], 'hybrid': C[4], 'aug': C[5]}
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'axes.linewidth': 0.8,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'legend.fontsize': 8,
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.dpi': 200, 'lines.linewidth': 2,
})


def dat(name):
    p = os.path.join(DATA, f'{name}.dat')
    return pd.read_csv(p, sep=r'\s+') if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(FIGS, f'{name}.png'), bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('wrote', name, flush=True)


def bars(ax, x, groups, labels, colors=None, width=0.8):
    n = len(groups)
    w = width / n
    for i, (g, lab) in enumerate(zip(groups, labels)):
        ax.bar(x + (i - (n - 1) / 2) * w, g, w, color=(colors or C)[i], label=lab, edgecolor=BG, linewidth=1)


# Fig. 3 training curves
d = dat('training_curves')
if d is not None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    a1.plot(d['epoch'], d['acc'], color=C[0], label='train')
    a1.plot(d['epoch'], d['val_acc'], color=C[1], ls='--', label='validation')
    a1.set(xlabel='Epoch', ylabel='Accuracy (%)')
    a2.plot(d['epoch'], d['loss'], color=C[0], label='train')
    a2.plot(d['epoch'], d['val_loss'], color=C[1], ls='--', label='validation')
    a2.set(xlabel='Epoch', ylabel='Cross-entropy', yscale='log')
    a1.legend(loc='lower right')
    fig.suptitle('Fig. 3  MLP training curves (Protocol B)', x=0.01, ha='left', color=INK, fontsize=10)
    save(fig, 'fig03_training_curves')

# Fig. 4 confusion matrix
d = dat('confusion')
if d is not None:
    M = np.zeros((26, 26))
    M[d['y'].astype(int), d['x'].astype(int)] = d['c']
    np.fill_diagonal(M, np.nan)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(np.minimum(M, 2), cmap=SEQ, vmin=0, vmax=2)
    ax.set_xticks(range(26), LETTERS, fontsize=7)
    ax.set_yticks(range(26), LETTERS, fontsize=7)
    ax.set(xlabel='Predicted letter', ylabel='True letter')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[0, 0.5, 1, 1.5, 2])
    cb.ax.set_yticklabels(['0', '0.5', '1', '1.5', '>=2'])
    cb.set_label('% of true class', color=INK2)
    cb.outline.set_visible(False)
    ax.set_title('Fig. 4  Off-diagonal confusions, Protocol B (diagonal omitted)', loc='left')
    save(fig, 'fig04_confusion_matrix')

# Fig. 5 qualitative examples
S2 = os.path.join(ROOT, 'results', 's2_main.json')
if os.path.exists(os.path.join(QUAL, 'q1_landmarks.csv')) and os.path.exists(S2):
    import json
    R = json.load(open(S2))                   # true / predicted letters of the eight examples (s2_main.py)
    lab = pd.DataFrame([{'image': f'q{q}.jpg', 'true': R[f'QualTrue{c}'], 'predicted': R[f'QualPred{c}']}
                        for q, c in enumerate('ABCDEFGH', 1) if f'QualTrue{c}' in R])
    lab.to_csv(os.path.join(QUAL, 'labels.csv'), index=False)
    fig, axes = plt.subplots(2, 4, figsize=(8, 4.4))
    for ax, (_, r) in zip(axes.flat, lab.iterrows()):
        img = plt.imread(os.path.join(QUAL, r['image']))
        h, w = img.shape[:2]
        lm = pd.read_csv(os.path.join(QUAL, r['image'].replace('.jpg', '_landmarks.csv')))
        px, py = lm['x'].to_numpy() * w, (1 - lm['y'].to_numpy()) * h
        ax.imshow(img)
        for a, b in BONES:
            ax.plot([px[a], px[b]], [py[a], py[b]], color='#ffffff', lw=2.2)
            ax.plot([px[a], px[b]], [py[a], py[b]], color=C[0], lw=1.2)
        ax.scatter(px, py, s=9, color=C[1], edgecolor='#ffffff', linewidth=0.5, zorder=3)
        ok = r['true'] == r['predicted']
        ax.set_title(f"true {r['true']}, predicted {r['predicted']}" + ('' if ok else '  (error)'),
                     fontsize=8, color=INK if ok else '#c62828')
        ax.set_axis_off()
    fig.suptitle('Fig. 5  Qualitative results on Protocol-B test images', color=INK, fontsize=10)
    save(fig, 'fig05_qualitative')

# Fig. 6 near-duplicates
d = dat('neardup_hist')
if d is not None:
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar(d['d'], d['pct'], 0.8, color=C[0])
    ax.set_xticks(range(0, 17, 2), [str(v) for v in range(0, 16, 2)] + ['>=16'])
    ax.set(xlabel='Hamming distance to the closest training image (bits)', ylabel='Test images (%)')
    ax.grid(axis='x', visible=False)
    ax.set_title('Fig. 6  Near-duplicate test images (dHash), random split of ASL Alphabet', loc='left')
    save(fig, 'fig06_near_duplicates')

# Fig. 7 Surrey LOSO
d = dat('surrey_loso')
if d is not None:
    cols = [('MLPraw', 'raw'), ('MLPnorm', 'normalized (ours)'), ('MLProt', 'rotation-normalized'),
            ('MLPangles', 'angles'), ('MLPhybrid', 'hybrid')]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(len(d))
    bars(ax, x, [d[c] for c, _ in cols], [lab for _, lab in cols],
         colors=[METHOD[k] for k in ('raw', 'norm', 'rot', 'angles', 'hybrid')])
    ax.set_xticks(x, d['signer'])
    ax.set(xlabel='Held-out signer', ylabel='Accuracy on held-out signer (%)', ylim=(80, 100))
    ax.grid(axis='x', visible=False)
    ax.legend(ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
    ax.set_title('Fig. 7  Surrey, leave-one-signer-out (MLP, detected hands)', loc='left', pad=22)
    save(fig, 'fig07_surrey_loso')

# Fig. 8 rotation
d = dat('rotation_curve')
if d is not None:
    series = [('Raw', 'raw', '-', 'o', 'raw'), ('Norm', 'normalized (ours)', '-', 'o', 'norm'),
              ('Aug', 'normalized + augmentation', '--', 's', 'aug'), ('Rot', 'rotation-normalized', '-', '^', 'rot'),
              ('Hybrid', 'hybrid', '-', 'D', 'hybrid'), ('Angles', 'angles', '-', 'x', 'angles')]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for c, lab, ls, mk, key in series:
        ax.plot(d['deg'], d[c], color=METHOD[key], ls=ls, marker=mk, ms=4, label=lab)
    ax.set(xlabel='In-plane rotation of the test landmarks |theta| (degrees)', ylabel='Accuracy (%)', xlim=(0, 45))
    ax.set_xticks(range(0, 46, 5))
    ax.legend(loc='lower left', ncol=2)
    ax.set_title('Fig. 8  Robustness to rotation (landmarks rotated in the image plane)', loc='left')
    save(fig, 'fig08_rotation')

# Fig. 9 calibration
r1, r2, rc = dat('reliability_raw'), dat('reliability_cal'), dat('risk_coverage')
if r1 is not None and rc is not None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a1.plot([0, 100], [0, 100], color=MUTED, ls='--', lw=1, label='ideal')
    a1.plot(r1['conf'], r1['acc'], color=C[1], marker='o', ms=4, label='T = 1')
    if r2 is not None:
        a1.plot(r2['conf'], r2['acc'], color=C[0], marker='o', ms=4, label='T = fitted')
    a1.set(xlabel='Confidence (%)', ylabel='Accuracy (%)', xlim=(0, 100), ylim=(0, 100), title='(a) Reliability')
    a1.legend(loc='upper left')
    a2.plot(rc['coverage'], rc['risk'], color=C[0])
    a2.set(xlabel='Coverage (%)', ylabel='Selective risk (%)', xlim=(50, 100), title='(b) Risk-coverage')
    a2.set_ylim(bottom=0)
    fig.suptitle('Fig. 9  Calibration and selective prediction (Protocol B)', x=0.01, ha='left', color=INK, fontsize=10)
    fig.tight_layout()
    save(fig, 'fig09_calibration')

# Fig. 10 landmark importance
d = dat('importance')
if d is not None:
    M = np.zeros((26, 21))
    M[d['y'].astype(int), d['x'].astype(int)] = np.maximum(d['c'], 0)
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    im = ax.imshow(M, cmap=SEQ, vmin=0, aspect='auto')
    for xb in (0.5, 4.5, 8.5, 12.5, 16.5):
        ax.axvline(xb, color='#ffffff', lw=1)
    ax.set_xticks(range(21), [str(j) for j in range(21)], fontsize=7)
    ax.set_yticks(range(26), LETTERS, fontsize=7)
    ax.set(xlabel='Landmark (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 little)', ylabel='Letter')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Recall drop (percentage points)', color=INK2)
    cb.outline.set_visible(False)
    ax.set_title('Fig. 10  Landmark importance (occlusion of one landmark)', loc='left')
    save(fig, 'fig10_landmark_importance')

# extra: detection rate per letter
d = dat('detrate')
if d is not None:
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.bar(d['label'], d['rate'], 0.75, color=C[0])
    ax.set(ylabel='Hands detected (%)', ylim=(0, 100))
    ax.grid(axis='x', visible=False)
    ax.set_title('MediaPipe detection rate per class, ASL Alphabet (default detector)', loc='left')
    save(fig, 'extra_detection_rate')

# extra: Table VII per signer
d = dat('surrey_signers')
if d is not None:
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(d))
    bars(ax, x, [100 * d['det_all'], 100 * d['acc'], 100 * d['e2e']],
         ['hands detected (with recovery)', 'accuracy on detected hands', 'end-to-end accuracy'], width=0.75)
    ax.set_xticks(x, d['signer'])
    ax.set(xlabel='Held-out signer', ylabel='%', ylim=(0, 100))
    ax.grid(axis='x', visible=False)
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
    ax.set_title('Table VII  Surrey per unseen signer (missed hands count as errors in end-to-end)', loc='left', pad=22)
    save(fig, 'table07_surrey_signers')

# extra: hardest letters for unseen signers
d = dat('surrey_letters_loso')
if d is not None:
    d = d.sort_values('acc')
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.bar(d['label'], d['acc'], 0.75, color=C[0])
    ax.set(ylabel='LOSO accuracy (%)', ylim=(50, 100), xlabel='Letter (sorted)')
    ax.grid(axis='x', visible=False)
    ax.set_title('Surrey, accuracy per letter for unseen signers (24 static letters)', loc='left')
    save(fig, 'extra_surrey_letters')
