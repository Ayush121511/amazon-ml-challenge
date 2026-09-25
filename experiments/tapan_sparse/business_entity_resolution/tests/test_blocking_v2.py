import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from preprocess import transform, RAW, EXTRA
from blocking_v2 import keys


class BlockingTests(unittest.TestCase):
    def test_unicode_keys(self):
        row = transform(dict(zip(RAW,['q','தமிழ் ગુજરાતી','12 road','India'])))
        self.assertIn(('India','name_words','தமிழ்'),keys(row))

    def test_full_pool_positive_exclusion_and_oversize(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = {1:[['q','Unique Café','12 Garden','India'],['excluded','Other','Elsewhere','India']],
                    2:[['a','Unique Café','12 Garden','India'],['b','Unique Café','12 Garden','US']],
                    3:[['c','Different Store','88 Street','India']]}
            for source, values in rows.items():
                path = root/f'train_source{source}.tsv.gz'
                with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=RAW+EXTRA,delimiter='\t')
                    writer.writeheader()
                    writer.writerows(transform(dict(zip(RAW,v))) for v in values)
                Path(str(path)+'.json').write_text(json.dumps({'input':{'version':2,'limit':0},'output_bytes':path.stat().st_size}))
            (root/'exclude.json').write_text(json.dumps([{'entity_id':'excluded'}]))
            (root/'gold.tsv').write_text('source1_entity_id\tmatched_entity_ids\nq\ta\n')
            script=Path(__file__).resolve().parents[1]/'src/blocking_v2.py'
            subprocess.run([sys.executable,str(script),'--data',str(root),'--exclude',str(root/'exclude.json'),
                            '--gold',str(root/'gold.tsv'),'--output',str(root/'out'),'--per-country','1'],
                           check=True,capture_output=True)
            report=json.loads((root/'out/report.json').read_text())
            self.assertEqual(report['all']['recall_at_50'],1)
            result=json.loads((root/'out/candidates.jsonl').read_text())
            self.assertEqual(result['entity_id'],'q')
            self.assertNotIn('b',result['ranked'])
