# Amazon ML Challenge 2026

## Status
- Repo structure set up (see README.md)
- HPC (IIT Delhi PADUM) access done and verified: A100 80GB GPU confirmed working
  on queue `scai_q`, project `scai`. Team setup guide pushed to
  https://github.com/Ayush121511/padum-setup — HPC setup itself is DONE, don't
  redo it unless something breaks.
- Miniconda env `amlc` created on HPC (`~/scratch/miniconda3`, python 3.11)
- Dataset not yet dropped (challenge Round 1 runs 25 Sept - 28 Sept 2026, 72hr window)

## Timeline
- Round 1: 25 Sept 2026 - 28 Sept 2026 (dataset + problem statement drop Day 1)
- Submission: 1-2 page approach doc + code/notebook, zipped
- Grand Finale: 7 Oct 2026 (top 10 teams, live virtual)

## Repo layout
- `data/raw/` - original dataset, untouched (gitignored)
- `data/processed/` - cleaned/feature-engineered data (gitignored)
- `notebooks/` - EDA and experiments
- `src/` - reusable pipeline code (preprocessing, features, models, eval)
- `models/` - saved model artifacts (gitignored)
- `submissions/` - generated submission files (gitignored)
- `references/hpc/` - HPC setup docs/scripts (personal copy, has real Kerberos ID)

## Workflow once dataset drops
1. Drop dataset in `data/raw/`
2. EDA in `notebooks/`
3. Move stable logic into `src/`
4. Track leaderboard metric explicitly, optimize for it directly
5. Get a working baseline submitted early, then iterate
6. For GPU training: push code to HPC, run via `references/hpc/batchjob.sh`
   (already configured for `scai`/`scai_q`)

## Conventions
- No comments unless explaining non-obvious WHY
- Don't over-engineer — 72hr time box, working baseline beats clever architecture
