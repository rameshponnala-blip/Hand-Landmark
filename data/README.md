# Data

The images are not part of this repository. Download them with

```bash
python experiments/download_data.py
```

which stores them under `data/kaggle/datasets/<owner>/<dataset>/versions/<n>/`, the layout the scripts expect.
A Kaggle account and API token may be needed (see https://github.com/Kaggle/kagglehub).
Please follow the terms of each dataset.

| Dataset | Kaggle handle | Content |
|---|---|---|
| ASL Alphabet | [`grassknoted/asl-alphabet`](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | 87,000 images (200×200 px), 29 classes (A-Z, del, space, nothing), one signer |
| ASL Alphabet Test | [`danrasband/asl-alphabet-test`](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test) | 870 images, 29 classes, varied backgrounds |
| Surrey ASL Fingerspelling Dataset A | [`mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out`](https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out) | 5 signers, 24 static letters (no J, Z); the 65,774 colour images are used |

## Landmark files

`landmarks_alphabet.csv.gz`, `landmarks_test.csv.gz` and `landmarks_surrey.csv.gz` are the output of
`experiments/s1_extract.py` (MediaPipe 0.10.20, one hand, `static_image_mode=True`). One row per image:

| Column | Meaning |
|---|---|
| `path` | image path relative to `data/kaggle/datasets/` |
| `label`, `signer`, `frame` | class, signer (`K` = ASL Alphabet, `T` = ASL Alphabet Test, Surrey signers A-E), frame number |
| `method` | detector step that found the hand: `default` (confidence 0.5), recovery steps `pad`, `flip`, `upscale`, `lowconf` (confidence 0.3), or `none` / `unreadable` |
| `h`, `w` | image size in pixels |
| `mp_ms` | MediaPipe time for the default detector (ms) |
| `handedness`, `score` | MediaPipe handedness label and score |
| `dhash` | 64-bit difference hash of the image (near-duplicate analysis) |
| `x0, y0, z0 … x20, y20, z20` | the 21 landmarks in a common pixel aspect: x·w/S, y·h/S, z·w/S with S = max(w, h) |

If `data/cache/` is empty, `common.load_landmarks` reads these files instead, so most experiments run
without the images.

## References

- A. Nagaraj, "ASL Alphabet," Kaggle, 2018.
- D. Rasband, "ASL Alphabet Test," Kaggle, 2018.
- N. Pugeault and R. Bowden, "Spelling it out: Real-time ASL fingerspelling recognition," in *Proc. IEEE Int.
  Conf. Comput. Vis. Workshops (ICCV Workshops)*, 2011, pp. 1114-1119.
