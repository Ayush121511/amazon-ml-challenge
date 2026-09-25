"""Stage 1 HPC job: preprocess all 6 source files in parallel, then measure
blocking recall on the training set. Run from the repo root on HPC.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(__file__))
from build_normalized import SOURCES, build_one


def _build(args):
    name, path = args
    build_one(name, path)


def main():
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        list(ex.map(_build, SOURCES.items()))
    print(f"\n=== all preprocessing done in {time.time()-t0:.1f}s ===\n", flush=True)

    import eval_blocking
    eval_blocking.main()


if __name__ == "__main__":
    main()
