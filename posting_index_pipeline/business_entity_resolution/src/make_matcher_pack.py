"""Package the pruned development set (stage-1 top-20 candidates of the 100k train references)
for matcher work: features, labels, split, stage-1 scores and membership of the top-10 and
adaptive candidate sets, plus gold links (incl. ones outside the candidates) and a README.
Contains IDs, numeric features and labels only (no raw text); share privately, not on GitHub.
"""
import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from prune_analysis import partition_of, rank_within

README = """# Matcher development pack (v5 pruned candidates)

Pairs = the **stage-1 top-20 candidates** per reference for 100,000 train references (50k US,
50k India), from the v4 retrieval (text TF-IDF + reverse + fine-tuned multilingual-e5, recall
99.88%) followed by a stage-1 XGBoost ranker on retrieval signals only (trained out-of-fold on
the train split). `in_top10` / `in_adaptive7` mark the other two shortlisted candidate sets
(adaptive = stage-1 rank <= 3, or stage-1 p >= 0.005 with rank <= 15; about 7.2 per reference).

## Files
- `pairs_top20.npz`: `X` (float32, one row per pair, columns = `features.json`), `y` (1 = match),
  `ref_id`, `target_id`, `split` (train/dev/holdout), `stage1_score`, `stage1_rank`, `in_top10`,
  `in_adaptive7`.
- `gold.json`: every true match per reference, **including those outside the candidates**. Use
  it for end-to-end F0.5.
- `countries.json`: reference -> country (US/India).

```python
import numpy as np, json
d = np.load("pairs_top20.npz")
X, y, split = d["X"], d["y"], d["split"]
features = json.load(open("features.json"))
train = split == "train"; dev = split == "dev"; holdout = split == "holdout"
top10 = d["in_top10"]          # restrict to the top-10 set with X[top10], ...
```

## Protocol (keep it, so numbers compare)
- Split per reference: `h = int(sha256(ref_id)[:8], 16) % 100` -> train <70, dev <85, else holdout
  (already in `split`).
- Tune on **dev**, report **holdout once** per final config.
- Score = macro F0.5 over **all** references of a partition (gold.json keys), with singletons:
  an empty prediction scores 1 when gold is empty, and anything else scores 0.
- Decision rule used so far: per reference, the top-k that maximises expected F0.5
  (1.25*sum(top-k p) / (0.25*sum(p) + k)), empty if prod(1-p) is higher or best p < floor
  (floor 0.7-0.8, chosen on dev). See `decide()` in `submission_pipeline.py` (vishal-progress).

## Baselines (holdout, 15,113 references; XGBoost on GPU, 255 leaves, lr 0.05, early stopping)
| candidate set | per ref | holdout recall | holdout F0.5 |
|---|---:|---:|---:|
| top 20 | 20 | 99.67% | 0.9732 |
| top 10 | 10 | 99.32% | 0.9728 |
| adaptive | 7.2 | 99.28% | 0.9728 |

The leaderboard (test) scored 0.964 for the unpruned v4 matcher (holdout 0.9731); the top teams
are at about 0.993. A perfect matcher on these candidates would score about 0.995-0.998, so the
matcher is the lever.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True, help='matcher_v4/pairs')
    p.add_argument('--prune', type=Path, required=True, help='prune_v1 (stage1_scores.npy, masks)')
    p.add_argument('--output', type=Path, required=True, help='zip file to write')
    a = p.parse_args()
    started = time.monotonic()
    names = json.loads((a.pairs / 'report.json').read_text())['features']
    pairs = pd.read_csv(a.pairs / 'pairs.tsv.gz', sep='\t', header=None, names=['ref', 'target'], dtype=str)
    codes, ref_names = pd.factorize(pairs['ref'])
    s1 = np.load(a.prune / 'stage1_scores.npy')
    rank = rank_within(codes.astype(np.int64), s1)
    top20 = np.load(a.prune / 'mask_top20.npy')
    rows = np.flatnonzero(top20)
    top10 = np.load(a.prune / 'mask_top10.npy')[rows]
    adaptive = np.load(a.prune / 'mask_adaptive_t0.005_min3_max15.npy')[rows]
    x = np.load(a.pairs / 'x.npy', mmap_mode='r')[rows]
    y = np.load(a.pairs / 'y.npy')[rows]
    ref_id = pairs['ref'].to_numpy()[rows]
    split_of = {s: partition_of(s) for s in ref_names}
    work = a.output.with_suffix('.dir')
    work.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(work / 'pairs_top20.npz', X=np.asarray(x, dtype=np.float32), y=y.astype(np.int8),
                        ref_id=np.asarray(ref_id, dtype=str),
                        target_id=np.asarray(pairs['target'].to_numpy()[rows], dtype=str),
                        split=np.array([split_of[s] for s in ref_id], dtype=str), stage1_score=s1[rows],
                        stage1_rank=rank[rows].astype(np.int16), in_top10=top10, in_adaptive7=adaptive)
    (work / 'features.json').write_text(json.dumps(names))
    shutil.copy(a.pairs / 'gold.json', work / 'gold.json')
    shutil.copy(a.pairs / 'countries.json', work / 'countries.json')
    (work / 'README.md').write_text(README)
    with zipfile.ZipFile(a.output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(work.iterdir()):
            z.write(f, f'matcher_dev_pack_v5/{f.name}',
                    compress_type=zipfile.ZIP_STORED if f.suffix == '.npz' else zipfile.ZIP_DEFLATED)
    shutil.rmtree(work)
    print(json.dumps({'rows': int(len(rows)), 'references': int(len(ref_names)),
                      'positives': int(y.sum()), 'top10_rows': int(top10.sum()),
                      'adaptive_rows': int(adaptive.sum()), 'zip_bytes': a.output.stat().st_size,
                      'seconds': round(time.monotonic() - started)}, indent=2))


if __name__ == '__main__':
    main()
