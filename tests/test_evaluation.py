"""Held-out completeness, fixed-plan and stratified paired-summary checks."""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from evaluation_analysis import evaluation_plan,validate_results,bootstrap_indices,aggregate_results,sha256,verify_freeze
from prepare_data import read_manifests
from run_evaluation import choose_examples


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frozen=json.loads((ROOT/'config/frozen_denoisers.json').read_text())
        self.plan=evaluation_plan(self.frozen,sha256(ROOT/'config/frozen_denoisers.json'))
        self.plan['bootstrap']['replicates']=10
        self.manifest=read_manifests(ROOT)['evaluation']
        self.rows=[]
        for record in self.manifest:
            for variant in self.plan['variants']:
                ds=0 if variant['method']=='identity' else -0.001
                dn=0 if variant['method']=='identity' else 1.0
                self.rows.append(dict(file=record['File'],noise=record['Noise'],nominal_snr_db=record['NominalSNR'],
                    speaker=record['File'].split('_')[0],variant=variant['id'],method=variant['method'],role=variant['role'],
                    stoi_in=0.8,stoi_out=0.8+ds,delta_stoi=ds,snr_in_db=5.,snr_out_db=5.+dn,delta_snr_db=dn))

    def test_plan_preserves_frozen_options(self):
        before=copy.deepcopy(self.frozen)
        plan=evaluation_plan(self.frozen,'test-hash')
        self.assertEqual(len(plan['variants']),8)
        self.assertEqual(self.frozen,before)
        self.assertEqual(len([v for v in plan['variants'] if v['role']=='primary']),4)
        for selected in self.frozen['selected']:
            actual=next(v for v in plan['variants'] if v['id']==selected['id'])
            self.assertEqual(actual['options'],selected['options'])
        verify_freeze(ROOT,self.frozen)
        self.frozen['protocol']['source_sha256']['code/denoise_speech.m']='0'*64
        with self.assertRaisesRegex(ValueError,'Source changed'):verify_freeze(ROOT,self.frozen)

    def test_missing_duplicate_and_label_changes_rejected(self):
        validate_results(self.rows,self.plan['variants'],self.manifest)
        with self.assertRaises(ValueError):validate_results(self.rows[:-1],self.plan['variants'],self.manifest)
        changed=copy.deepcopy(self.rows);changed[-1]=changed[0]
        with self.assertRaisesRegex(ValueError,'Duplicate'):validate_results(changed,self.plan['variants'],self.manifest)
        changed=copy.deepcopy(self.rows);changed[0]['noise']='wrong'
        with self.assertRaisesRegex(ValueError,'labels'):validate_results(changed,self.plan['variants'],self.manifest)

    def test_stratified_resampling_is_deterministic(self):
        records=self.rows[::8]
        a=bootstrap_indices(records,replicates=5);b=bootstrap_indices(records,replicates=5)
        np.testing.assert_array_equal(a,b)
        for draws in a:
            for noise in ['bus','office','cafe']:
                self.assertEqual(sum(records[i]['noise']==noise for i in draws),80)
            self.assertEqual(sum(records[i]['speaker']=='p232' for i in draws),120)

    def test_summary_counts_and_zero_baseline_interval(self):
        tables=aggregate_results(self.rows,self.plan,self.manifest)
        self.assertEqual(len(tables['overall']),8)
        self.assertEqual(len(tables['by_condition']),96)
        self.assertEqual(len(tables['by_speaker']),16)
        self.assertTrue(all(row['pairs']==20 for row in tables['by_condition']))
        self.assertTrue(all(row['pairs']==120 for row in tables['by_speaker']))
        baseline=next(r for r in tables['overall'] if r['method']=='identity')
        self.assertEqual(baseline['delta_stoi_ci_low'],0)
        self.assertEqual(baseline['delta_snr_db_ci_high'],0)

    def test_listening_selection_is_predeclared_and_failures_labeled(self):
        selected=choose_examples(self.manifest,self.rows,self.plan)
        cases=selected['cases']
        pilot=[c for c in cases if c['purpose']=='predeclared_low_snr_listening']
        self.assertEqual(len(pilot),6)
        self.assertEqual({c['nominal_snr_db'] for c in pilot},{2.5})
        self.assertEqual(len({(c['noise'],c['speaker']) for c in pilot}),6)
        self.assertTrue(all(c['purpose'].startswith('posthoc') for c in cases[6:]))


if __name__=='__main__':unittest.main()
