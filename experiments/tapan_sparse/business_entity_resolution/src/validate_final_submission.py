"""Official matching checks plus streaming checks of the full candidate file."""
import argparse
import csv
import importlib.util
import itertools
import json
from pathlib import Path


def validate(args):
    stamp = args.output / 'VALIDATED.json'
    if stamp.exists():
        stamp.unlink()
    spec = importlib.util.spec_from_file_location('organizer_validator', args.validator)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    errors, warnings = official.validate(str(args.output / 'matching_results.tsv'), None,
                                         str(args.test_dir), check_ids=True)
    for warning in warnings:
        print(f'Organizer warning: {warning}', flush=True)
    if errors:
        raise ValueError('Organizer validation failed: ' + '; '.join(errors))
    valid = set()
    for source in (2, 3):
        valid.update(official.read_ids(str(args.test_dir / f'test_source{source}.tsv')))
    count = 0
    with (args.output / 'matching_results.tsv').open(encoding='utf-8', newline='') as mf, \
         (args.output / 'candidate_pairs.tsv').open(encoding='utf-8', newline='') as cf:
        matches, candidates = csv.DictReader(mf, delimiter='\t'), csv.DictReader(cf, delimiter='\t')
        if matches.fieldnames != official.MATCHING_HEADER or candidates.fieldnames != official.CANDIDATE_HEADER:
            raise ValueError('Unexpected submission headers')
        for mr, cr in itertools.zip_longest(matches, candidates):
            if mr is None or cr is None or mr['source1_entity_id'] != cr['source1_entity_id']:
                raise ValueError('Candidate/reference alignment mismatch')
            if None in mr or None in cr or cr['candidate_entity_ids'] is None:
                raise ValueError('Incorrect TSV column count')
            predicted = set(filter(None, mr['matched_entity_ids'].split(',')))
            ids = list(filter(None, cr['candidate_entity_ids'].split(',')))
            unique = set(ids)
            if len(unique) != len(ids) or not unique <= valid or not predicted <= unique:
                raise ValueError('Duplicate/nonexistent candidates or predictions outside candidates')
            count += 1
    report = {'status': 'PASS', 'references': count,
              'checks': ['official matching validation with target-ID checks',
                         'streaming candidate headers, exact aligned reference coverage, unique/existing target IDs, prediction subset']}
    stamp.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validator', type=Path, required=True)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    validate(parser.parse_args())


if __name__ == '__main__':
    main()
