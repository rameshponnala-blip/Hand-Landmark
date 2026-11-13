"""Step 1: MediaPipe hand landmarks for every image of the three datasets.

For each image we record the 21 landmarks, the detector that found the hand
('default', or one of the recovery steps 'pad', 'flip', 'upscale', 'lowconf'
used for gap G5), and a 64-bit difference hash used for the near-duplicate
analysis (gap G1). Coordinates are stored in a common pixel aspect:
x*w/S, y*h/S, z*w/S with S = max(w, h).

Usage:  .venv\\Scripts\\python.exe experiments\\s1_extract.py [--limit N]
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE, COLS, dataset_root  # noqa: E402

IMG_EXT = ('.jpg', '.jpeg', '.png')


# dataset listings
def list_alphabet():
    base = glob.glob(os.path.join(dataset_root('grassknoted/asl-alphabet'), '**', 'asl_alphabet_train', 'A'),
                     recursive=True)[0]
    base = os.path.dirname(base)
    items = []
    for label in sorted(os.listdir(base)):
        for f in sorted(os.listdir(os.path.join(base, label))):
            if f.lower().endswith(IMG_EXT):
                m = re.search(r'(\d+)', f)
                items.append({'path': os.path.join(base, label, f), 'label': label.upper() if len(label) == 1 else label,
                              'signer': 'K', 'frame': int(m.group(1)) if m else -1})
    return items


def list_test():
    base = dataset_root('danrasband/asl-alphabet-test')
    items = []
    for label in sorted(os.listdir(base)):
        d = os.path.join(base, label)
        if not os.path.isdir(d) or label == 'asl-alphabet-test':      # skip the duplicated copy
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXT):
                items.append({'path': os.path.join(d, f), 'label': label.upper() if len(label) == 1 else label,
                              'signer': 'T', 'frame': -1})
    return items


def list_surrey():
    base = dataset_root('mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out')
    items = []
    for p in sorted(glob.glob(os.path.join(base, '**', 'color_*.png'), recursive=True)):
        parts = p.split(os.sep)
        letter, signer = parts[-2], parts[-3]
        m = re.search(r'color_\d+_(\d+)', os.path.basename(p))
        items.append({'path': p, 'label': letter.upper(), 'signer': signer, 'frame': int(m.group(1)) if m else -1})
    return items


LISTERS = {'alphabet': list_alphabet, 'test': list_test, 'surrey': list_surrey}


# worker
def _init():
    global H, H_LOW
    import mediapipe.python.solutions.hands as mp_hands
    H = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    H_LOW = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)


def _detect(hands, img):
    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None
    lm = np.array([[p.x, p.y, p.z] for p in res.multi_hand_landmarks[0].landmark], np.float64)
    cls = res.multi_handedness[0].classification[0]
    return lm, cls.label, cls.score


def _dhash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view('>u8')[0])


def _process(item):
    import time
    img = cv2.imread(item['path'], cv2.IMREAD_COLOR)
    out = dict(item, method='unreadable', h=0, w=0, mp_ms=np.nan, handedness='', score=np.nan, dhash=0)
    if img is None:
        return out
    h, w = img.shape[:2]
    out.update(h=h, w=w, dhash=_dhash(img))
    t0 = time.perf_counter()
    r = _detect(H, img)
    out['mp_ms'] = 1000 * (time.perf_counter() - t0)
    method = 'default'
    if r is None:                                          # G5: recovery cascade
        p = int(0.25 * max(h, w))
        padded = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT,
                                    value=[float(v) for v in img.reshape(-1, 3).mean(0)])
        r, method = _detect(H, padded), 'pad'
        if r is not None:
            lm = r[0]
            lm[:, 0] = (lm[:, 0] * (w + 2 * p) - p) / w
            lm[:, 1] = (lm[:, 1] * (h + 2 * p) - p) / h
            lm[:, 2] = lm[:, 2] * (w + 2 * p) / w
    if r is None:
        r, method = _detect(H, cv2.flip(img, 1)), 'flip'
        if r is not None:
            r[0][:, 0] = 1.0 - r[0][:, 0]
    if r is None and min(h, w) < 256:
        r, method = _detect(H, cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)), 'upscale'
    if r is None:
        r, method = _detect(H_LOW, img), 'lowconf'
    if r is None:
        out['method'] = 'none'
        return out
    lm, hand, score = r
    S = max(w, h)
    lm[:, 0] *= w / S
    lm[:, 1] *= h / S
    lm[:, 2] *= w / S
    out.update(method=method, handedness=hand, score=score)
    out.update({c: v for c, v in zip(COLS, lm.astype(np.float32).ravel())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='pilot run: images per dataset')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--only', nargs='*', default=list(LISTERS))
    args = ap.parse_args()
    for name in args.only:
        items = LISTERS[name]()
        if args.limit:
            rng = np.random.default_rng(0)
            items = [items[i] for i in sorted(rng.choice(len(items), min(args.limit, len(items)), replace=False))]
        print(f'{name}: {len(items)} images', flush=True)
        with Pool(args.workers, initializer=_init) as pool:
            rows = []
            for i, r in enumerate(pool.imap(_process, items, chunksize=32)):
                rows.append(r)
                if (i + 1) % 5000 == 0:
                    print(f'  {i + 1}/{len(items)}', flush=True)
        df = pd.DataFrame(rows).sort_values('path').reset_index(drop=True)
        df.to_pickle(os.path.join(CACHE, f'landmarks_{name}.pkl'))
        print(df['method'].value_counts().to_string(), flush=True)


if __name__ == '__main__':
    main()
