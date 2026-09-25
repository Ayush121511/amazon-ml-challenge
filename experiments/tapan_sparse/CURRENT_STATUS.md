# First full-test submission status

Snapshot from the latest user-shared Padum output on 25 September 2026. PBS state may have changed; check it live before acting.

## Completed

- Train preprocessing: 2,206,821 references and 10,320,219 targets. Test preprocessing: 1,732,544 references and 9,969,589 targets; all three test manifests passed.
- The 100,000-reference full-pool training benchmark and 10,000-reference candidate/matcher experiment produced the offline measurements in [README.md](README.md).
- Matcher job `1065613.pbshpc` finished with dev macro F0.5 `0.89449544`, holdout `0.89640584`, and dev-selected threshold `0.70`. The model trained on 900,811 pairs and 18,533 positives after purging validation target IDs from training.
- The first full-test cache job `1065639.pbshpc` was killed by its 4-hour walltime after logging at least 9.55 million cached targets. It did not write its final manifest, so the original dependent job could not run.

## Active attempt at the time of this snapshot

- Recovery job `1066027.pbshpc` was queued (`Q`) with no estimated start time. PBS reported `Not Running: Insufficient amount of resource: ncpus`.
- Continuation job `1066028.pbshpc` was held (`H`) on `afterok:1066027.pbshpc`. It should run automatically if recovery succeeds.
- No completed cache manifest, `VALIDATED.json`, test prediction, or leaderboard score has been confirmed. Do not treat the queued jobs as a completed submission.

The recovery job runs `test_cache_resume.py` as a small synthetic check, then reuses complete cache chunks and processes only missing targets. The continuation runs US, France, and India sequentially, then scores, assembles, and validates both TSVs. Its 12-hour allocation is a limit, not a measured duration; retrieval and scoring write checkpoints that can be reused if it needs another job.

## Check and submit

On Padum, from `~/scratch/amazon_ml_2026`:

```bash
qstat -u aib262140
test -f artifacts/test_prototype_v1/cache/manifest.json && echo 'cache complete'
test -f artifacts/test_prototype_v1/submission/VALIDATED.json && cat artifacts/test_prototype_v1/submission/VALIDATED.json
```

Once validation reports `PASS`, retrieve `artifacts/test_prototype_v1/submission/matching_results.tsv` and `candidate_pairs.tsv`. Upload them through the competition portal. The PBS scripts do not upload to the portal.

If recovery fails, inspect `er_cache_resume.o1066027`; if the continuation fails, inspect `er_test_all.o1066028`. Check the actual job state and log before scheduling a replacement. Do not restart the original cache from zero or duplicate a running job.

After a valid first submission, use `audit_matcher_errors.py` on the development set to separate missing-candidate loss, rejected true candidates, and false positives. That will determine whether retrieval or matcher changes are more valuable than speculative model additions.
