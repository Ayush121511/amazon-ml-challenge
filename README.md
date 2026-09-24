# Amazon ML Challenge 2026

## Timeline
- Round 1: 25 Sept 2026 - 28 Sept 2026 (72 hours, dataset drops Day 1)
- Grand Finale: 7 Oct 2026 (top 10 teams)

## Structure
- `data/raw/` - original dataset, untouched
- `data/processed/` - cleaned/feature-engineered data
- `data/external/` - any external data used
- `notebooks/` - EDA and experiments
- `src/` - reusable pipeline code (preprocessing, features, models, eval)
- `models/` - saved model artifacts
- `submissions/` - generated submission files
- `references/` - problem statement, rules, scoring docs

## Workflow
1. Drop dataset in `data/raw/` on Day 1.
2. EDA in `notebooks/`.
3. Move stable logic into `src/`.
4. Track leaderboard metric explicitly, optimize for it directly.
5. Get a working baseline submitted early.
