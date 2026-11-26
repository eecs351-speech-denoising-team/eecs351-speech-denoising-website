"""Independent calibration of alignment, amplitude-sensitive SNR and selection."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from speech_metrics import aligned_signals, common_valid_interval, snr_db, baseline_metrics, enhancement_metrics
from tuning_candidates import build_candidates
from run_tuning import summarize, freeze_selection


class MetricTests(unittest.TestCase):
    def test_common_interval_excludes_boundaries_for_every_method(self):
        n=16000
        s=np.sin(np.arange(n)*0.1)
        y=s+0.1*np.cos(np.arange(n)*0.3)
        changed=y.copy(); changed[:64]=100; changed[-64:]=-100
        ref,noisy,processed,interval=aligned_signals(s,y,np.column_stack([y,changed]),[[1,n],[65,n-64]])
        self.assertEqual(interval,(65,n-64))
        self.assertEqual(len(ref),n-128)
        np.testing.assert_array_equal(processed[:,0],processed[:,1])
        np.testing.assert_array_equal(noisy,y[64:-64])

    def test_known_snr_and_no_gain_fitting(self):
        t=np.arange(16000)/16000
        s=0.1*np.sin(2*np.pi*1000*t)
        noise=0.05*np.sin(2*np.pi*2000*t)
        self.assertAlmostEqual(snr_db(s,s+noise),20*np.log10(2),places=10)
        self.assertAlmostEqual(snr_db(s,2*s),0,places=12)
        self.assertEqual(snr_db(s,s),float('inf'))

    def test_no_automatic_delay_search(self):
        s=np.random.default_rng(351).normal(size=16000)
        delayed=np.r_[0,s[:-1]]
        self.assertLess(snr_db(s,delayed),-2.5)

    def test_identity_deltas_are_zero(self):
        t=np.arange(32000)/16000
        s=(0.2+0.1*np.sin(2*np.pi*3*t))*np.sin(2*np.pi*180*t)
        noisy=s+0.03*np.random.default_rng(351).normal(size=len(s))
        baseline=baseline_metrics(s,noisy,16000)
        result=enhancement_metrics(s,noisy,16000,baseline)
        self.assertEqual(result['delta_snr_db'],0)
        self.assertAlmostEqual(result['delta_stoi'],0,places=14)

    def test_invalid_shapes_and_ranges_rejected(self):
        for ranges in [[[0,100]],[[1,101]],[[50,40]],[[1.5,100]],[[1,30],[40,100]]]:
            with self.assertRaises(ValueError): common_valid_interval(100,ranges)
        with self.assertRaises(ValueError): snr_db(np.ones(10),np.ones(11))
        with self.assertRaises(ValueError): snr_db(np.zeros(10),np.zeros(10))

    def test_predeclared_grid_and_incomplete_freeze(self):
        candidates=build_candidates()
        self.assertEqual(len(candidates),41)
        self.assertEqual(len({c['id'] for c in candidates}),41)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            summarize([],candidates,['p236_136'])
        # Fixture rankings exercise matching after independently selecting modes.
        summaries=[dict(candidate=c['id'],family=c['family'],method=c['method'],
                        mean_stoi=i/100,mean_delta_snr_db=0) for i,c in enumerate(candidates)]
        summaries.reverse()
        frozen=freeze_selection({},candidates,summaries)
        self.assertEqual(len(frozen['selected']),6)
        lookup={c['id']:c for c in candidates}
        for pair in frozen['matched_noise_mode_pairs']:
            anchor=dict(lookup[pair['anchor_candidate']]['options'])
            other=dict(pair['counterpart_options'])
            self.assertNotEqual(anchor.pop('noiseMode'),other.pop('noiseMode'))
            self.assertEqual(anchor,other)


if __name__=='__main__':
    unittest.main()
