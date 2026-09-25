"""Writes the earlier diagnostic reference samples, regenerated from their seeds, as ID lists."""
import argparse
from pathlib import Path

from retrieval_experiment import sample_references
from blocking_v2 import select


def previous_samples(source1):
    """retrieval_v1 (2,000), then blocking_v2 / sparse_fullpool (400), then selective_index_v1 (400)."""
    retrieval_v1 = [r['entity_id'] for r in sample_references(source1, per_country=1000, seed=2026)]
    sparse = [r['entity_id'] for r in select(source1, set(retrieval_v1), 200)]
    selective = [r['entity_id'] for r in select(source1, set(retrieval_v1) | set(sparse), 200)]
    return {'retrieval_v1': retrieval_v1, 'sparse_fullpool': sparse, 'selective_index_v1': selective}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source1', type=Path, required=True, help='Preprocessed train_source1.tsv.gz')
    p.add_argument('--output', type=Path, required=True, help='Directory for <sample>_ids.txt files')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for name, ids in previous_samples(a.source1).items():
        (a.output / f'{name}_ids.txt').write_text('\n'.join(ids) + '\n', encoding='utf-8')
        print(f'{name}: {len(ids):,} IDs', flush=True)


if __name__ == '__main__':
    main()
