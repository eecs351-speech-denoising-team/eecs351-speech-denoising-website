"""Failure-path checks for frozen data selection, ZIP integrity and STOI."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import prepare_data as data
from score_stoi import score
import numpy as np


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'data').mkdir()
        for path in (data.ROOT/'data').glob('*_manifest.csv'):
            shutil.copyfile(path, self.root/'data'/path.name)

    def rewrite_manifest(self, split, edit):
        path = self.root/'data'/f'{split}_manifest.csv'
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        edit(rows)
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['File','Noise','NominalSNR'])
            writer.writeheader()
            writer.writerows(rows)

    def test_duplicate_and_changed_balance_rejected(self):
        self.rewrite_manifest('tuning', lambda rows: rows.__setitem__(1, rows[0]))
        with self.assertRaisesRegex(ValueError, 'distinct'):
            data.read_manifests(self.root)
        shutil.copyfile(data.ROOT/'data/tuning_manifest.csv', self.root/'data/tuning_manifest.csv')
        self.rewrite_manifest('evaluation', lambda rows: rows[0].update(Noise='living'))
        with self.assertRaisesRegex(ValueError, 'balance'):
            data.read_manifests(self.root)

    def test_cross_split_overlap_rejected(self):
        train = data.read_manifests(self.root)['tuning'][0]
        self.rewrite_manifest('evaluation', lambda rows: rows.__setitem__(0, train))
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            data.read_manifests(self.root)

    def test_remote_zip_member_crc_and_header_validation(self):
        for method in [zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED]:
            blob = io.BytesIO()
            content = b'original PCM samples'*100
            with zipfile.ZipFile(blob, 'w', compression=method) as archive:
                archive.writestr('folder/p257_004.wav', content)
            raw = blob.getvalue()
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                info = archive.infolist()[0]
            def fetch_range(url, byte_range):
                start, end = map(int, byte_range.split('-'))
                return raw[start:end+1], len(raw)
            with patch.object(data, 'fetch', side_effect=fetch_range):
                extracted = data.remote_member('https://fixture.invalid', info)
            self.assertEqual(extracted, content)
            self.assertTrue(data.check_crc(extracted, info))
            self.assertFalse(data.check_crc(extracted[:-1], info))
            with patch.object(data, 'fetch', return_value=(b'bad!'+raw[4:30],len(raw))):
                with self.assertRaisesRegex(ValueError, 'invalid'):
                    data.remote_member('https://fixture.invalid', info)

    def test_offline_setup_resume_and_corruption(self):
        manifests = data.read_manifests(self.root)
        archives = self.root/'archives'
        archives.mkdir()
        logs = {}
        for name, splits in [('log_trainset_28spk.txt',['tuning']),
                             ('log_testset.txt',['evaluation','exploratory'])]:
            logs[name] = ''.join(f'{r["File"]} {r["Noise"]} {r["NominalSNR"]}\n'
                                 for split in splits for r in manifests[split])
        with zipfile.ZipFile(archives/'logfiles.zip','w') as archive:
            archive.writestr('log_readme.txt','Fixture condition logs')
            for name, content in logs.items():
                archive.writestr(name,content)
        for name in data.BITSTREAMS:
            if name == 'logfiles':
                continue
            rows = manifests['tuning'] if 'trainset' in name else manifests['evaluation']+manifests['exploratory']
            with zipfile.ZipFile(archives/(name+'.zip'),'w',compression=zipfile.ZIP_DEFLATED) as archive:
                for row in rows:
                    archive.writestr(name+'/'+row['File']+'.wav', b'fixture-bytes-for-CRC')
        with redirect_stdout(io.StringIO()):
            data.prepare(self.root,self.root/'data',archives,2)
            data.prepare(self.root,self.root/'data',archives,2)
        report=json.loads((self.root/'outputs/setup/download_inventory.json').read_text())
        self.assertEqual(report['status'],'PASS')
        self.assertEqual(report['wav_count'],1286)
        target=self.root/'data'/report['files'][0]['path']
        target.write_bytes(b'corrupted')
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError,'existing WAV fails'):
            data.prepare(self.root,self.root/'data',archives,2)
        self.assertEqual(json.loads((self.root/'outputs/setup/download_inventory.json').read_text())['status'],'FAIL')
        # Log agreement is independently enforced before downloading any audio.
        (self.root/'data/log_testset.txt').write_text('p257_004 wrong_noise 2.5\n')
        with self.assertRaisesRegex(ValueError,'official log mismatch'):
            data.validate_logs(manifests,self.root/'data')

    def test_stoi_identity_and_invalid_inputs(self):
        t=np.arange(48000)/16000
        signal=(0.5+0.3*np.sin(2*np.pi*3*t))*np.sin(2*np.pi*180*t)
        self.assertAlmostEqual(score(signal,signal,16000)['score'],1.0,places=9)
        self.assertAlmostEqual(score(np.zeros_like(signal),signal,16000)['score'],0.0,places=9)
        with self.assertRaisesRegex(ValueError,'identical lengths'):
            score(signal,signal[:-1],16000)
        with self.assertRaisesRegex(ValueError,'NaN or Inf'):
            score(np.full(signal.shape,np.nan),signal,16000)
        with self.assertRaisesRegex(ValueError,'too short'):
            score(np.zeros(100),np.zeros(100),16000)
        with self.assertRaisesRegex(ValueError,'silent'):
            score(np.zeros(16000),np.zeros(16000),16000)


if __name__ == '__main__':
    unittest.main()
