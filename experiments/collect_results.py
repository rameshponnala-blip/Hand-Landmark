"""Puts all values from results/s*.json into one table, results/all_results.csv.

Also computes the few numbers that need more than one script (size/speed ratios, Holm-corrected
McNemar p-values, end-to-end accuracy on ASL Alphabet Test).
"""
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import LETTERS, RES, load_landmarks  # noqa: E402

R, SRC = {}, {}
for f in sorted(glob.glob(os.path.join(RES, 's*.json'))):
    name = os.path.basename(f)[:-5]
    for k, v in json.load(open(f)).items():
        R[k], SRC[k] = v, name          # s8 overwrites the single-seed values of s3/s6


def num(k):
    return float(R[k].replace(',', ''))


def put(k, v, src='derived'):
    R[k], SRC[k] = v, src


try:
    put('CNNMLPParamRatio', f"{num('CNNParams') / num('MLPParams'):.0f}")
    put('SpeedUp', f"{num('CNNLat') / num('MLPLat'):.0f}")
    put('SizeRatio', f"{num('CNNSize') / num('MLPSize'):.0f}")
except KeyError:
    pass
for k in list(R):                       # also keep the mean of "mean $\pm$ std" values
    m = re.fullmatch(r'([\d.,]+) \$\\pm\$ [\d.]+', R[k])
    if m and f'{k}Mean' not in R:
        put(f'{k}Mean', m.group(1))

# Holm correction over all McNemar tests
pkeys = [k for k in R if re.fullmatch(r'McNemar[A-Za-z]+', k) and f'{k}b' in R]
praw = {k: (0.001 if '<' in R[k] else float(R[k])) for k in pkeys}
running = 0.0
for i, k in enumerate(sorted(pkeys, key=lambda k: praw[k])):
    running = max(running, min(1.0, (len(pkeys) - i) * praw[k]))
    put(f'{k}Holm', f'<{running:.3f}' if '<' in R[k] else f'{running:.3f}')
put('McNemarCount', str(len(pkeys)))
if 'DetAlphabetDefault' in R:
    put('DetAlphabetDefaultMiss', f"{100 - num('DetAlphabetDefault'):.2f}")

# end-to-end on ASL Alphabet Test = accuracy on detected hands * detection rate
try:
    T = load_landmarks('test')
    T = T[T['label'].isin(LETTERS)]
    det = (~T['method'].isin(['none', 'unreadable'])).mean()
    put('CrossTestDetected', f"{int(round(det * len(T)))} of {len(T)}")
    for k in ('CrossTestMLPnormMean', 'HandXnormMirrorTestMean'):
        if k in R:
            put(k.replace('Mean', 'Etoe'), f'{num(k) * det:.2f}')
except FileNotFoundError:
    pass


def plain(v):
    v = v.replace('$\\pm$', '+-').replace('$<$', '<').replace('$\\times$', 'x').replace('\\%', '%').replace('~', ' ')
    return v.replace(' \\\\ ', '; ').replace(' & ', ' ')


out = os.path.join(RES, 'all_results.csv')
with open(out, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['key', 'value', 'source'])
    for k in sorted(R):
        w.writerow([k, plain(R[k]), SRC[k]])
print(f'{len(R)} values -> {out}')
