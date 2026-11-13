"""Step 7 (review M2): a stronger image baseline.

MobileNetV2 trained on hand crops (square box around the MediaPipe landmarks,
40% margin) instead of full frames, with strong per-sample augmentation on the
GPU: random scale/shift/rotation, horizontal flip (the image counterpart of
mirror augmentation), brightness/contrast/saturation jitter and random erasing.
Evaluated on the same detected images as the landmark models.

Usage:  .venv\\Scripts\\python.exe experiments\\s7_cnn_crop.py [--epochs 8]
"""
import argparse
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torchvision

sys.path.insert(0, os.path.dirname(__file__))
from common import (COLS, LETTERS, RES, SEED, Results, labels, letters_only, load_landmarks, mcnemar,  # noqa: E402
                    pct, pfmt, protocol_b, scores)

SIZE = 224
MARGIN = 1.4
ALL = ('default', 'pad', 'flip', 'upscale', 'lowconf')
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def crop_one(path, lm):
    """Square crop around the 21 landmarks (stored as x*S, y*S with S = max(w, h))."""
    img = cv2.imread(path)
    h, w = img.shape[:2]
    S = max(w, h)
    xy = lm.reshape(21, 3)[:, :2] * S
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * MARGIN
    side = max(side, 32)
    M = np.float32([[SIZE / side, 0, SIZE / 2 - cx * SIZE / side], [0, SIZE / side, SIZE / 2 - cy * SIZE / side]])
    out = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def load_crops(df):
    lms = df[COLS].to_numpy(np.float32)
    with ThreadPoolExecutor(16) as ex:
        return np.stack(list(ex.map(crop_one, df['path'].tolist(), lms)))


def augment(x):
    """Per-sample geometric and photometric augmentation of a float batch in [0, 1]."""
    n, dev = len(x), x.device
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(15)
    sc = 0.85 + 0.3 * torch.rand(n, device=dev)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    tx, ty = (torch.rand(2, n, device=dev) * 2 - 1) * 0.1
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = Fnn.affine_grid(theta, x.shape, align_corners=False)
    x = Fnn.grid_sample(x, grid, padding_mode='border', align_corners=False)
    b = 0.7 + 0.6 * torch.rand(n, 1, 1, 1, device=dev)
    c = 0.7 + 0.6 * torch.rand(n, 1, 1, 1, device=dev)
    s = 0.6 + 0.8 * torch.rand(n, 1, 1, 1, device=dev)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * c + m
    g = x.mean(1, keepdim=True)
    x = ((x - g) * s + g) * b
    # random erasing of one rectangle in half of the images
    eh = (torch.rand(n, device=dev) * 0.3 + 0.1) * SIZE
    ew = (torch.rand(n, device=dev) * 0.3 + 0.1) * SIZE
    ey, ex = torch.rand(n, device=dev) * (SIZE - eh), torch.rand(n, device=dev) * (SIZE - ew)
    r = torch.arange(SIZE, device=dev).view(1, SIZE)
    my = (r >= ey[:, None]) & (r < (ey + eh)[:, None])
    mx = (r >= ex[:, None]) & (r < (ex + ew)[:, None])
    mask = (my[:, :, None] & mx[:, None, :]) & (torch.rand(n, 1, 1, device=dev) < 0.5)
    x = torch.where(mask[:, None], torch.rand(n, 3, 1, 1, device=dev), x)
    return x.clamp(0, 1)


def to_batch(arr, dev, train=False):
    x = torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float() / 255
    if train:
        x = augment(x)
    return (x - MEAN.to(dev)) / STD.to(dev)


def predict(model, imgs, dev, mask=None, bs=256):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(imgs), bs):
            with torch.autocast('cuda', dtype=torch.float16):
                outs.append(model(to_batch(imgs[s:s + bs], dev)).float().cpu().numpy())
    z = np.concatenate(outs)
    if mask is not None:
        z[:, ~mask] = -np.inf
    return z.argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = 'cuda'
    R = Results('s7_cnn_crop')
    t0 = time.time()

    D = letters_only(load_landmarks('alphabet'))
    y = labels(D)
    tr, te = protocol_b(y)
    rng = np.random.default_rng(SEED)
    va = rng.choice(tr, int(0.1 * len(tr)), replace=False)
    ta = np.setdiff1d(tr, va)
    print('cropping images', flush=True)
    imgs = load_crops(D)

    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, len(LETTERS)))
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(ta) / 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler()
    best, state, wait = np.inf, None, 0
    for ep in range(args.epochs):
        model.train()
        idx = np.random.permutation(ta)
        tl = tc = 0
        for s in range(0, len(idx), 128):
            b = idx[s:s + 128]
            x, t = to_batch(imgs[b], dev, True), torch.from_numpy(y[b]).to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(x)
                loss = Fnn.cross_entropy(out, t, label_smoothing=0.1)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(b)
            tc += (out.argmax(1) == t).sum().item()
        pv = predict(model, imgs[va], dev)
        model.eval()
        with torch.no_grad():
            vl = np.mean([Fnn.cross_entropy(model(to_batch(imgs[va[s:s + 256]], dev)).float(),
                                            torch.from_numpy(y[va[s:s + 256]]).to(dev)).item()
                          for s in range(0, len(va), 256)])
        print(f'  epoch {ep + 1}: loss {tl / len(ta):.4f} acc {tc / len(ta):.4f} | val loss {vl:.4f} '
              f'acc {(pv == y[va]).mean():.4f} | {(time.time() - t0) / 60:.0f} min', flush=True)
        if vl < best:
            best, wait, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 3:
                break
    model.load_state_dict(state)
    R['CNNCropTrainMin'] = f'{(time.time() - t0) / 60:.0f}'

    p = predict(model, imgs[te], dev)
    s = scores(y[te], p)
    R['CNNCropTestAcc'], R['CNNCropTestF1'] = pct(s['acc']), pct(s['f1'])
    pb = np.load(os.path.join(RES, 'protoB_preds.npz'))
    assert np.array_equal(pb['te'], te)
    b, c, pval = mcnemar(pb['MLP'] == y[te], p == y[te])
    R['McNemarCNNCrop'], R['McNemarCNNCropb'], R['McNemarCNNCropc'] = pfmt(pval), str(b), str(c)
    del imgs

    for name in ('test', 'surrey'):
        df = letters_only(load_landmarks(name), ALL)
        ye = labels(df)
        mask = np.isin(np.arange(len(LETTERS)), np.unique(ye))
        pe = predict(model, load_crops(df), dev, mask)
        tag = name.capitalize()
        R[f'Cross{tag}CNNCrop'] = pct((pe == ye).mean())
        R[f'Cross{tag}CNNCropF1'] = pct(scores(ye, pe)['f1'])
        print(tag, R[f'Cross{tag}CNNCrop'], flush=True)
    R.save()
    print('saved', len(R.R), 'values', flush=True)


if __name__ == '__main__':
    main()
