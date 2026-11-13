"""Download the public Kaggle datasets used in the paper (retries on flaky connections).

Usage:  .venv\\Scripts\\python.exe experiments\\download_data.py
"""
import os
import time

os.environ.setdefault('KAGGLEHUB_CACHE', os.path.join(os.path.dirname(__file__), '..', 'data', 'kaggle'))
import kagglehub  # noqa: E402

DATASETS = [
    'danrasband/asl-alphabet-test',                               # varied backgrounds (cross-dataset test)
    'grassknoted/asl-alphabet',                                   # main training set
    'mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out',  # Surrey ASL Dataset A (5 signers)
]

for handle in DATASETS:
    for attempt in range(1, 9):
        try:
            print(handle, '->', kagglehub.dataset_download(handle), flush=True)
            break
        except Exception as e:  # network errors are common here; back off and retry
            print(f'{handle}: attempt {attempt} failed ({type(e).__name__}); retrying', flush=True)
            time.sleep(min(60, 5 * attempt))
    else:
        raise SystemExit(f'Could not download {handle}')
