# Preprocessing v2

Run `padum/preprocess_cpu.pbs` using PBS, after copying the updated code to Padum.
It requests 4 CPUs, 1 GPU and 8 GB memory in scai_q. The user's queue output
confirmed scai_q requires at least one GPU and standard blocks project scai.
The program itself uses only CPUs; the allocation is released when it finishes.

The Python implementation uses only the standard library and processes one row
at a time, with three independent file workers. No network or proxy is needed.
The existing Conda Python executable is invoked directly, without activation.

The first batch processes all three training sources. Output is compressed TSV
under `artifacts/preprocessed_v2/train`, with JSON manifests of row/country counts,
missing addresses, ordered-ID SHA256, elapsed time and input file metadata.
Original TSV fields and row order are preserved. Training labels are not read.

Derived fields include Unicode-normalized casefolded names/addresses, separate
Latin-accent-folded versions, numeric address tokens, heuristic postal candidates and
an explicit missing-address flag. No business suffix removal, external address
lookup, destructive deduplication or learned vocabulary is performed. Postal
candidates can be house numbers; never use them as authoritative hard filters.

Version 2 preserves Unicode marks and ZWJ/ZWNJ in normalized fields, and removes
diacritics only from Latin bases in folded fields. Version 1 could remove Indic
vowel signs/viramas. Its full run completed, but its derived fields should not
be used for retrieval. Originals are intact. Version 2 writes a separate folder.
This is not transliteration, translation, or full Unicode script identification.

Completed files can be reused with --resume when source path/size/mtime and
preprocessing version match and output size agrees. This is an operational
resume check, not a full content-integrity audit. Interrupted .partial files are
recomputed. Do not launch two jobs writing the same output directory. Existing
incompatible completed output is rejected instead of overwritten.

Local validation: twelve unit tests passed, including Tamil, Gujarati, Hindi,
Bengali, Telugu, Arabic marks, mixed scripts and joiners; 1,000 records from each
of six source files were processed successfully. The v2 remote run is pending.
Entity/component-aware validation splitting and retrieval are separate next
steps; no model score or leakage-safe learned split is claimed by this stage.

Submit from the remote login node:

```bash
cd ~/scratch/amazon_ml_2026
qsub padum/preprocess_cpu.pbs
qstat -u aib262140
```

PBS combines stdout/stderr into er_preprocess_v2.o<job-number> in the submission
directory. After completion inspect that log and the three output manifests.
To process test data later, use the same CLI with --split test and the same
output root in a compute allocation.
