"""Predict letters for images (or a webcam) with the trained MLP in results/mlp_fp32.onnx.

    python experiments/demo_predict.py img1.jpg some_folder/
    python experiments/demo_predict.py --webcam

Predictions with confidence below --tau (0.8 by default, same as in the paper) are printed as
"abstain". The training images are all right hands, so use --flip for a left hand.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import LETTERS, ROOT, feat_norm  # noqa: E402

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


class Recognizer:
    def __init__(self, model, tau=0.8, static=True):
        import mediapipe.python.solutions.hands as mp_hands
        import onnxruntime as ort
        self.hands = mp_hands.Hands(static_image_mode=static, max_num_hands=1, min_detection_confidence=0.5)
        self.hands_low = mp_hands.Hands(static_image_mode=static, max_num_hands=1, min_detection_confidence=0.3)
        self.sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
        self.tau, self.recover = tau, static

    @staticmethod
    def _detect(hands, img):
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None
        return np.array([[p.x, p.y, p.z] for p in res.multi_hand_landmarks[0].landmark], np.float32)

    def _detect_with_recovery(self, img):
        # same fallback steps as s1_extract.py
        lm = self._detect(self.hands, img)
        if lm is not None or not self.recover:
            return lm
        h, w = img.shape[:2]
        p = int(0.25 * max(h, w))
        padded = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT,
                                    value=[float(v) for v in img.reshape(-1, 3).mean(0)])
        lm = self._detect(self.hands, padded)
        if lm is not None:
            lm[:, 0] = (lm[:, 0] * (w + 2 * p) - p) / w
            lm[:, 1] = (lm[:, 1] * (h + 2 * p) - p) / h
            lm[:, 2] = lm[:, 2] * (w + 2 * p) / w
            return lm
        lm = self._detect(self.hands, cv2.flip(img, 1))
        if lm is not None:
            lm[:, 0] = 1.0 - lm[:, 0]
            return lm
        if min(h, w) < 256:
            lm = self._detect(self.hands, cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            if lm is not None:
                return lm
        return self._detect(self.hands_low, img)

    def landmarks(self, img):
        # same scaling as s1_extract.py: x*w/S, y*h/S, z*w/S with S = max(w, h)
        lm = self._detect_with_recovery(img)
        if lm is None:
            return None
        h, w = img.shape[:2]
        S = max(h, w)
        return lm * np.array([w / S, h / S, w / S], np.float32)

    def predict(self, img):
        L = self.landmarks(img)
        if L is None:
            return 'no hand', 0.0, None
        z = self.sess.run(None, {'x': feat_norm(L[None])})[0][0]
        p = np.exp(z - z.max())
        p /= p.sum()
        k = int(p.argmax())
        return (LETTERS[k] if p[k] >= self.tau else 'abstain'), float(p[k]), L


def files(args):
    for a in args:
        if os.path.isdir(a):
            yield from sorted(f for f in glob.glob(os.path.join(a, '**', '*'), recursive=True)
                              if f.lower().endswith(IMG_EXT))
        else:
            yield a


def main():
    ap = argparse.ArgumentParser(description='ASL letter recognition demo')
    ap.add_argument('inputs', nargs='*')
    ap.add_argument('--webcam', action='store_true')
    ap.add_argument('--tau', type=float, default=0.8)
    ap.add_argument('--model', default=os.path.join(ROOT, 'results', 'mlp_fp32.onnx'))
    ap.add_argument('--flip', action='store_true')
    a = ap.parse_args()
    if not a.inputs and not a.webcam:
        ap.error('give image files/folders or --webcam')

    if a.webcam:
        rec = Recognizer(a.model, a.tau, static=False)
        cap = cv2.VideoCapture(0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1) if a.flip else frame
            label, conf, L = rec.predict(frame)
            if L is not None:
                S = max(frame.shape[:2])
                for x, y, _ in L:
                    cv2.circle(frame, (int(x * S), int(y * S)), 3, (52, 104, 235), -1)
            cv2.putText(frame, f'{label}  {conf:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (214, 120, 42), 3)
            cv2.imshow('ASL fingerspelling (q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    rec = Recognizer(a.model, a.tau)
    print('image,prediction,confidence')
    for f in files(a.inputs):
        img = cv2.imread(f)
        if img is None:
            print(f'{f},unreadable,')
            continue
        label, conf, _ = rec.predict(cv2.flip(img, 1) if a.flip else img)
        print(f'{f},{label},{conf:.3f}')


if __name__ == '__main__':
    main()
