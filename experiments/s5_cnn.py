"""Step 5: MobileNetV2 image baseline (PyTorch, GPU) on the same Protocol-B split.

The CNN sees the full RGB image. It is evaluated on the Protocol-B test images,
on the varied-background test set and on Surrey, and compared with the
landmark MLP using McNemar's test. CPU latency is measured with ONNX Runtime
on one thread, like the MLP.

Usage:  .venv\\Scripts\\python.exe experiments\\s5_cnn.py [--epochs-frozen 10 --epochs-ft 5]
"""
import argparse
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
from common import (LETTERS, RES, SEED, Results, ints, labels, letters_only, load_landmarks, mcnemar,  # noqa: E402
                    pct, pfmt, protocol_b, scores)

SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_images(paths, side=SIZE):
    def rd(p):
        img = cv2.imread(p)
        return cv2.cvtColor(cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    with ThreadPoolExecutor(16) as ex:
        return np.stack(list(ex.map(rd, paths)))


def to_batch(arr, dev, train=False):
    x = torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float() / 255
    if train:                                    # light photometric jitter only
        x = x * (0.8 + 0.4 * torch.rand(len(x), 1, 1, 1, device=dev))
        x = x.clamp(0, 1)
    return (x - MEAN.to(dev)) / STD.to(dev)


def run_epoch(model, imgs, ys, dev, opt=None, bs=128):
    train = opt is not None
    model.train(train)
    idx = np.random.permutation(len(ys)) if train else np.arange(len(ys))
    tot, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            x, t = to_batch(imgs[b], dev, train), torch.from_numpy(ys[b]).to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(x)
                loss = Fnn.cross_entropy(out, t)
            if train:
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            tot += loss.item() * len(b)
            correct += (out.argmax(1) == t).sum().item()
            n += len(b)
    return tot / n, correct / n


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


def fit(model, params, lr, epochs, data, dev, patience=3):
    (xa, ya), (xv, yv) = data
    opt = torch.optim.Adam(params, lr=lr)
    best, state, wait = np.inf, None, 0
    for ep in range(epochs):
        tl, ta = run_epoch(model, xa, ya, dev, opt)
        vl, va = run_epoch(model, xv, yv, dev)
        print(f'  epoch {ep + 1}: loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}', flush=True)
        if vl < best:
            best, wait, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)


def main():
    global scaler
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs-frozen', type=int, default=10)
    ap.add_argument('--epochs-ft', type=int, default=5)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = 'cuda'
    scaler = torch.amp.GradScaler()
    R = Results('s5_cnn')
    t0 = time.time()

    D = letters_only(load_landmarks('alphabet'))
    y = labels(D)
    tr, te = protocol_b(y)
    rng = np.random.default_rng(SEED)
    va = rng.choice(tr, int(0.1 * len(tr)), replace=False)
    ta = np.setdiff1d(tr, va)
    print('loading images', flush=True)
    imgs = load_images(D['path'].tolist())

    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, len(LETTERS)))
    model = model.to(dev)
    data = ((imgs[ta], y[ta]), (imgs[va], y[va]))
    for p in model.features.parameters():
        p.requires_grad = False
    print('stage 1: frozen backbone', flush=True)
    fit(model, model.classifier.parameters(), 1e-3, args.epochs_frozen, data, dev)
    for p in model.features.parameters():
        p.requires_grad = True
    print('stage 2: fine-tuning', flush=True)
    fit(model, model.parameters(), 1e-5, args.epochs_ft, data, dev)
    R['CNNTrainMin'] = f'{(time.time() - t0) / 60:.0f}'

    p = predict(model, imgs[te], dev)
    s = scores(y[te], p)
    R['CNNTestAcc'], R['CNNTestF1'] = pct(s['acc']), pct(s['f1'])
    R['CNNParams'] = ints(sum(q.numel() for q in model.parameters()))
    pb = np.load(os.path.join(RES, 'protoB_preds.npz'))
    assert np.array_equal(pb['te'], te)
    b, c, pv = mcnemar(pb['MLP'] == y[te], p == y[te])
    R['McNemarCNN'], R['McNemarCNNb'], R['McNemarCNNc'] = pfmt(pv), str(b), str(c)

    # cross-dataset: all images (the CNN needs no hand detector) and the detected subset
    for name in ('test', 'surrey'):
        try:
            df = load_landmarks(name)
        except FileNotFoundError:
            continue
        df = df[df['label'].isin(LETTERS) & (df['method'] != 'unreadable')].reset_index(drop=True)
        ye = labels(df)
        mask = np.isin(np.arange(len(LETTERS)), np.unique(ye))
        pe = predict(model, load_images(df['path'].tolist()), dev, mask)
        tag = name.capitalize()
        R[f'Cross{tag}CNNAll'] = pct((pe == ye).mean())
        det = (df['method'] != 'none').to_numpy()
        R[f'Cross{tag}CNN'] = pct((pe[det] == ye[det]).mean())
        R[f'Cross{tag}CNNF1'] = pct(scores(ye[det], pe[det])['f1'])

    # CPU latency of the CNN with ONNX Runtime, one thread
    import onnxruntime as ort
    onnx_path = os.path.join(RES, 'cnn_fp32.onnx')
    m_cpu = model.float().cpu().eval()
    torch.onnx.export(m_cpu, torch.zeros(1, 3, SIZE, SIZE), onnx_path, input_names=['x'],
                      output_names=['logits'], dynamo=False)
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=['CPUExecutionProvider'])
    xs = to_batch(imgs[te[:50]], 'cpu').numpy()
    ts = []
    for i in range(300):
        t1 = time.perf_counter()
        sess.run(None, {'x': xs[i % 50][None]})
        ts.append(time.perf_counter() - t1)
    lat = 1000 * np.median(ts[30:])
    R['CNNLat'], R['CNNFPS'] = f'{lat:.2f}', f'{1000 / lat:.0f}'
    R['CNNSize'] = f'{os.path.getsize(onnx_path) / 1024:.1f}'
    R.save()
    print('saved', len(R.R), 'values', flush=True)


if __name__ == '__main__':
    main()
