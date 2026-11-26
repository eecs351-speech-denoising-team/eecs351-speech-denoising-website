"""Listening material/CSV checks; fixture scores are not study results."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from build_listening_pack import build,FIELDS
from summarize_listening import decode_ratings,summarize


class ListeningTests(unittest.TestCase):
    def setUp(self):
        self.key=[dict(trial_id=f'T{i:02d}',sample_code=code,file=f'p232_{i:03d}',method=method)
                  for i in range(1,7) for code,method in zip('ABCD',['identity','fir','spectral_subtraction','wiener'])]
        self.ratings=[dict(listener_id='TEST_FIXTURE',trial_id=r['trial_id'],sample_code=r['sample_code'],
                           **{field:'3' for field in FIELDS},notes='unit-test fixture') for r in self.key]

    def test_blank_template_is_not_data(self):
        blank=[dict(listener_id='',trial_id=r['trial_id'],sample_code=r['sample_code'],
                    **{field:'' for field in FIELDS}) for r in self.key]
        self.assertEqual(decode_ratings(self.key,blank),[])
        self.assertEqual(summarize([]),[])

    def test_complete_listener_decodes_and_balances_methods(self):
        decoded=decode_ratings(self.key,self.ratings)
        self.assertEqual(len(decoded),24)
        summary=summarize(decoded)
        self.assertEqual(len(summary),4)
        for row in summary:
            self.assertEqual(row['listeners'],1)
            self.assertEqual(row['sample_ratings'],6)
            self.assertEqual(row['mean_overall_quality'],3)
            self.assertEqual(row['between_listener_sd_overall_quality'],'')

    def test_partial_invalid_and_duplicate_ratings_rejected(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):decode_ratings(self.key,self.ratings[:-1])
        altered=[dict(r) for r in self.ratings];altered[0]['overall_quality']=''
        with self.assertRaisesRegex(ValueError,'four integer'):decode_ratings(self.key,altered)
        altered[0]['overall_quality']='6'
        with self.assertRaises(ValueError):decode_ratings(self.key,altered)
        with self.assertRaisesRegex(ValueError,'Duplicate'):decode_ratings(self.key,self.ratings+[self.ratings[0]])

    def test_pack_excludes_key_and_leaves_ratings_blank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cases=[]
            for i in range(1,7):
                file=f'p232_{i:03d}';folder=root/'examples'/file;folder.mkdir(parents=True)
                for name in ['clean','noisy','fir','spectral_subtraction','wiener']:
                    (folder/(name+'.wav')).write_bytes(('unit-test '+name).encode())
                cases.append(dict(file=file,purpose='predeclared_low_snr_listening',noise='bus'))
            (root/'case_selection.json').write_text(json.dumps(dict(cases=cases)))
            (root/'plan.json').write_text(json.dumps(dict(freeze_sha256='test-hash',primary={m:m for m in ['identity','fir','spectral_subtraction','wiener']})))
            build(root)
            with zipfile.ZipFile(root/'listening_pack.zip') as archive:
                names=archive.namelist()
                self.assertEqual(sum(name.endswith('.wav') for name in names),30)
                self.assertFalse(any('key' in name or 'organizer' in name for name in names))
                self.assertIn('index.html',names)
            with (root/'listening_pack/ratings_template.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),24)
            self.assertTrue(all(not row[field] for row in rows for field in FIELDS))
            with (root/'organizer/listening_key.csv').open() as f:key=list(csv.DictReader(f))
            for trial in {r['trial_id'] for r in key}:
                self.assertEqual(len({r['method'] for r in key if r['trial_id']==trial}),4)


if __name__=='__main__':unittest.main()
