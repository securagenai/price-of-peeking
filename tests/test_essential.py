import unittest, tempfile, json, subprocess, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts.core import real as r,synthetic as sy,synthetic_base as sb,amendment as am,followup as fu
from scripts.core.anytime.extension_v02 import Bettor
from scripts.run_experiments_derived import binding
ROOT=Path(__file__).resolve().parents[1]
class EssentialTests(unittest.TestCase):
    def test_hw16_and_byte_decode(self):
        x=np.zeros((5,256),np.int16);x[:,0]=[0,1,-1,-32768,32767]
        self.assertEqual(r.hw16(x,'pqm4').tolist(),[0,1,16,1,15])
        b=x.astype('<i2').view(np.uint8).reshape(5,512)
        np.testing.assert_array_equal(r.hw16(b,'ref'),r.hw16(x,'pqm4'))
    def test_threshold_ties_and_constant(self):
        self.assertEqual(r.threshold([1,1,1,2])['status'],'DEGENERATE_TARGET')
        self.assertEqual(r.threshold([0,0,1,2])['median'],.5)
    def test_rng_namespace_and_reset(self):
        a=r.labels(4,24,0,0,20,.4);np.testing.assert_array_equal(a,r.labels(4,24,0,0,20,.4))
        self.assertFalse(np.array_equal(a,r.labels(4,24,0,1,20,.4)))
        for method in ['plugin','ons_gain','ons_literal']:
            self.assertEqual(Bettor(method).step(.7),0.)
    def test_predictability_and_swap(self):
        p=np.array([.8,.1]);y=np.array([1,0]);self.assertAlmostEqual(r.swap(p,y)[0],-r.swap(p,y[::-1])[0])
        for d in [-1.,1.]:self.assertEqual(fu.step(np.zeros(1),np.ones(1),np.zeros(1),np.array([d]))[2][0],0.)
    def test_fixed_randomization_reproducible(self):
        d=np.linspace(-.3,.7,16);a=sy.fixed_p_paths({'raw':d},[16,32],42,31);b=sy.fixed_p_paths({'raw':d},[16,32],42,31)
        self.assertEqual(a,b);self.assertTrue(all(0<p<=1 for p in a['raw']))
    def test_odd_scale_joint_sign_invariance(self):
        c={'past_window_pairs':8,'initial_pairs':2,'initial_scale':.25,'quantiles':[.1,.9],'quantile_method':'linear','epsilon':1e-6};x=np.linspace(-.8,.8,20);sign=np.where(np.arange(20)%2,1,-1)
        a,s=sb.transform_path(x,c);b,t=sb.transform_path(x*sign,c);np.testing.assert_allclose(a*sign,b);np.testing.assert_allclose(s,t)
    def test_censoring_and_multiplicity(self):
        self.assertIsNone(sy.grid_n80([.8,.7],[10,20])['grid_N80']);self.assertEqual(sy.grid_n80([.8,.9],[10,20])['status'],'AT_NMIN_BOUNDARY')
        self.assertGreater(am.gate_parameters(43,.05)['quantile_999'],0)
    def test_streaming_welch_matches_batch(self):
        g=np.random.default_rng(9);x=g.normal(size=(100,3));Y=g.integers(0,2,(1,100)).astype(float);a={};b={}
        r.welch_accumulate(a,x,Y)
        for lo in range(0,100,20):r.welch_accumulate(b,x[lo:lo+20],Y[:,lo:lo+20])
        np.testing.assert_allclose(r.welch_value(a)[0],r.welch_value(b)[0],atol=1e-12)
    def test_checkpoint_no_duplicate_and_identity(self):
        with tempfile.TemporaryDirectory() as p:
            out=Path(p);jobs=[{'i':0},{'i':1}];calls=[]
            def calc(j):calls.append(j);return {'count':j['i']}
            r.checkpoint_stage(out,'tiny',jobs,'id',calc,r.Budget(),1);r.checkpoint_stage(out,'tiny',jobs,'id',calc,r.Budget(),1)
            self.assertEqual(len(calls),2);self.assertEqual(r.require_complete(out,'tiny')['status'],'COMPLETE')
            with self.assertRaises(ValueError):r.checkpoint_stage(out,'tiny',jobs,'changed',calc,r.Budget(),1)
    def test_no_real_access_without_flag_and_output_isolation(self):
        result=subprocess.run([sys.executable,'-B','-m','scripts.run_experiments_derived','--workflow','real','--stage','init','--output-root',str(ROOT/'forbidden')],cwd=ROOT,capture_output=True)
        self.assertNotEqual(result.returncode,0);self.assertFalse((ROOT/'forbidden').exists())
    def test_synthetic_fit_payoff_and_both_procedures(self):
        c={'training_rows_per_scenario':32,'training_reference_amplitude':.8};w=sb.fit_witness('linear',c,8);x,y=sb.generate('linear',64,.2,9);d=sb.payoffs(w,x,y)
        self.assertTrue(np.isfinite(d).all());self.assertLessEqual(abs(d).max(),1)
        p=sy.fixed_p_paths({'raw':d},[32,64],11,7);e=sy.wealth_path(d,'ons_gain',[32,64],[.05]);self.assertEqual(len(p['raw']),2);self.assertEqual(len(e['terminal_logwealth']),2)
    def test_provider_schema_and_explicit_root_synthetic_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tiny.npy';np.save(path,np.zeros((3,256),np.int16));inv={'captures':{'pqm4_variable':{'implementation':'pqm4','files':{'mult_a':{'0':{'relative_path':'tiny.npy','header':r.header(path)}}}}}}
            provider=r.Provider(tmp,inv,{'approved':True,'scope':'REGISTERED_STAGED_EXECUTION','candidate_sha256':'x'},'x')
            self.assertEqual(provider.metadata('pqm4_variable','mult_a',[(0,0),(0,2)]).tolist(),[0,0])
            np.save(path,np.ones((4,256),np.int16))
            with self.assertRaises(ValueError):provider.metadata('pqm4_variable','mult_a',[(0,0)])
    def test_export_scope_has_no_excluded_capture_or_reserve(self):
        schema=r.read(ROOT/'configs/input_schema.json')
        self.assertEqual(set(schema),{'ref_fixed','ref_variable','pqm4_fixed','pqm4_variable'})
        for key,v in schema.items():
            if key.endswith('_variable'):
                self.assertTrue(all(int(c)<10 for fam in v['files'].values() for c in fam))
    def test_fresh_run_version_guard(self):
        from scripts import run_experiments_derived as driver
        with patch.object(driver.importlib.metadata,'version',return_value='0.0.0'):
            with self.assertRaises(RuntimeError):driver.require_recorded_environment()
    def test_tiny_synthetic_stages_end_to_end(self):
        from scripts import run_experiments_derived as driver
        cfg=r.read(ROOT/'configs/synthetic.json');cfg.update(scenarios=['linear'],null_scenarios=[],candidate_amplitudes=[.2],training_rows_per_scenario=32,main_grid=[16,32],pilot_grid=[16,32],randomizations=7)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);r.new(out/'RERUN_FREEZE.json',{'scope':'SYNTHETIC_TEST_ONLY'})
            driver.synthetic(out,'training',cfg,r.Budget(),1,'test')
            def small_jobs(stage,cfg,selection):return [{'case':'linear','amplitude':.2,'replicate':i,'seed':42+i} for i in range(2 if stage=='prediction' else 1)]
            with patch.object(driver.sy,'jobs',small_jobs):
                for stage in ['pilot','prediction','main']:driver.synthetic(out,stage,cfg,r.Budget(),4,'test')
            self.assertTrue((out/'PREDICTION_COMPARISON.json').exists())
            self.assertEqual(r.read(out/'MAIN_CHECKPOINT.json')['status'],'COMPLETE')
if __name__=='__main__':unittest.main()
