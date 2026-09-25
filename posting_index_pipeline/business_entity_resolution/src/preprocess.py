"""Streaming, label-independent preprocessing. No third-party dependencies."""
import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

VERSION = 2
RAW = ['entity_id', 'business_name', 'business_address', 'country']
EXTRA = ['name_norm', 'name_folded', 'address_norm', 'address_folded',
         'address_numbers', 'postal_candidates', 'address_missing']


def normalize(value, fold=False):
    value = unicodedata.normalize('NFKC', value).casefold()
    if fold:
        # Strip marks only when attached to a Latin base, never globally.
        result = []
        latin_base = False
        for char in unicodedata.normalize('NFD', value):
            if unicodedata.category(char).startswith('M'):
                if not latin_base:
                    result.append(char)
            else:
                latin_base = 'LATIN' in unicodedata.name(char, '')
                result.append(char)
        value = unicodedata.normalize('NFC', ''.join(result))
    # Marks include Indic vowel signs/viramas. Joiners affect shaping too.
    value = value.replace('&', ' and ')
    return ' '.join(''.join(c if c.isalnum() or
                           unicodedata.category(c).startswith('M') or
                           c in '\u200c\u200d' else ' ' for c in value).split())


def transform(row):
    if not row['entity_id'] or not row['country']:
        raise ValueError('Empty entity_id or country')
    address = row['business_address']
    # Candidates, not validated postal codes: house numbers can have this length.
    patterns = {'us': r'(?<!\d)\d{5}(?:-\d{4})?(?!\d)',
                'india': r'(?<!\d)[1-9]\d{5}(?!\d)',
                'france': r'(?<!\d)\d{5}(?!\d)'}
    pattern = patterns.get(row['country'].strip().casefold())
    postal = sorted(set(re.findall(pattern, address))) if pattern else []
    return {**row,
            'name_norm': normalize(row['business_name']),
            'name_folded': normalize(row['business_name'], True),
            'address_norm': normalize(address),
            'address_folded': normalize(address, True),
            'address_numbers': ' '.join(sorted(set(re.findall(r'\d+', address)))),
            'postal_candidates': ' '.join(postal),
            'address_missing': str(int(not address.strip()))}


def process_file(task):
    source, target, limit, resume = task
    source, target = Path(source), Path(target)
    stat = source.stat()
    signature = {'version': VERSION, 'source': str(source.resolve()),
                 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'limit': limit}
    manifest = target.with_suffix(target.suffix + '.json')
    if resume and target.exists() and manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous['input'] == signature and previous['output_bytes'] == target.stat().st_size:
            print(f'SKIP completed {source.name}', flush=True)
            return previous
    if target.exists() or manifest.exists():
        raise FileExistsError(f'{target}: existing output does not qualify for resume; use a new output directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.partial')
    start = time.monotonic()
    count = 0
    countries, missing = Counter(), Counter()
    digest = hashlib.sha256()
    with source.open(encoding='utf-8-sig', newline='') as inp, gzip.open(
            partial, 'wt', encoding='utf-8', newline='', compresslevel=1) as out:
        reader = csv.DictReader(inp, delimiter='\t')
        if reader.fieldnames != RAW:
            raise ValueError(f'Unexpected columns in {source}: {reader.fieldnames}')
        writer = csv.DictWriter(out, fieldnames=RAW + EXTRA, delimiter='\t')
        writer.writeheader()
        for row in reader:
            if limit and count >= limit:
                break
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed row {count + 2} in {source}')
            writer.writerow(transform(row))
            digest.update((row['entity_id'] + '\n').encode('utf-8'))
            count += 1
            countries[row['country']] += 1
            missing[row['country']] += int(not row['business_address'].strip())
            if count % 250000 == 0:
                print(f'{source.name}: {count:,} rows', flush=True)
    os.replace(partial, target)
    report = {'input': signature, 'rows': count, 'countries': dict(countries),
              'missing_address': dict(missing), 'ordered_id_sha256': digest.hexdigest(),
              'seconds': round(time.monotonic() - start, 2),
              'output_bytes': target.stat().st_size}
    temporary = manifest.with_suffix('.partial')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    os.replace(temporary, manifest)
    print(f'DONE {source.name}: {count:,} rows in {report["seconds"]}s', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'test', 'all'], default='train')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--limit', type=int, default=0, help='Rows per file; 0 means full data')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error('workers must be positive; limit must be nonnegative')
    tasks = []
    for split in (['train', 'test'] if args.split == 'all' else [args.split]):
        for number in range(1, 4):
            filename = f'{split}_source{number}.tsv'
            source = args.data / split / filename
            if not source.is_file():
                parser.error(f'Missing input: {source}')
            target = args.output / split / (filename + '.gz')
            tasks.append((str(source), str(target), args.limit, args.resume))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(process_file, tasks))
    print(f'Complete: {sum(r["rows"] for r in reports):,} rows. Originals and labels unchanged.', flush=True)


if __name__ == '__main__':
    main()
