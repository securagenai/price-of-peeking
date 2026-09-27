"""Synthetic-only v0.2. No data-root, real labels, or waveform serialization."""
import argparse
import itertools
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
from .common import atomic_json, sha
from .processes import PredictableFraction, mixture_logwealth, swap_score as historical_swap_score
from .witnesses import RunningRidge, GeometricMLP, PredictableProduct

WITNESSES = ('ridge', 'mlp', 'univariate_mean', 'second_order')
STRATEGIES = ('plugin', 'ons_gain', 'ons_literal')
METHODS = tuple(s+'_'+w for s in STRATEGIES for w in WITNESSES+('mixture4',)) + ('lr_ridge_rowwise', 'lr_mlp_rowwise')

def swap_score(p,y):
    value=historical_swap_score(p,y)
    # Exact identity orientation for tied labels; avoid cancellation residue.
    return 0. if y[0]==y[1] else value

class ONS:
    """SKIT PMLR202 Algorithm1, loss f=-D for gain convention; literal also kept.

    z=f/(1-lambda*f); A+=z*z; lambda'=clip(lambda-c*z/A,0,.5).
    This explicitly separates the printed sign convention from log-gain ONS.
    """
    def __init__(self, literal=False):
        self.fraction=0.; self.a=1.; self.literal=literal
    def value(self): return self.fraction
    def update(self,d):
        if not math.isfinite(d) or abs(d)>1+1e-12: raise ValueError('payoff outside [-1,1]')
        f=d if self.literal else -d
        z=f/(1-self.fraction*f)
        self.a+=z*z
        self.fraction=float(np.clip(self.fraction-2/(2-math.log(3))*z/self.a,0,.5))

class Bettor:
    def __init__(self,strategy):
        self.rule=PredictableFraction() if strategy=='plugin' else ONS(strategy=='ons_literal')
        self.logwealth=0.
    def step(self,d):
        if not np.isfinite(d) or abs(d)>1+1e-12: raise ValueError('invalid payoff')
        lam=self.rule.value()
        self.logwealth+=math.log1p(lam*d)
        self.rule.update(d)  # only NEXT bet can use d
        return self.logwealth

def synthetic_pairs(case,n,seed,d=8):
    """Streaming rows, independent label/background RNGs; no whole-stream buffer."""
    ss=np.random.SeedSequence(seed); rx,ry=(np.random.default_rng(s) for s in ss.spawn(2))
    last=np.zeros(d);shift=np.zeros(d)
    for t in range(n//2):
        xs=[]; ys=[]
        for i in range(2):
            j=2*t+i; y=int(ry.integers(2)); noise=rx.normal(size=d);x=noise.copy()
            if case=='drift': x+=-6+12*j/max(1,n-1)
            elif case=='ar1': last=.95*last+noise; x=last.copy()
            elif case=='heavy_tails': x=rx.standard_t(2.5,size=d)*(1+j/n)
            elif case=='block_shifts':
                if j%32==0: shift=rx.normal(scale=5,size=d)
                x+=shift
            elif case=='linear': x[0]+=2*(2*y-1)
            elif case=='nonlinear': x[0]=rx.choice([-1.,1.])*(1+2*y)+.25*noise[0]
            elif case=='second_order':
                x[0]=rx.choice([-1.,1.])+.2*noise[0];x[1]=np.sign(x[0])*(2*y-1)+.2*noise[1]
            else: raise ValueError(case)
            xs.append(x);ys.append(y)
        yield np.asarray(xs),np.asarray(ys)

def pointwise_lr(witness,x,y,p=.5):
    """Row map explicitly invoked once per row; no second-row feature conditioning.

    witness is fixed from previous complete pairs. This is predictable on the
    finer row filtration too. Caller releases wealth only at complete pairs.
    """
    out=[]
    for row,label in zip(x,y):
        q=witness.predict(row[None,:])[0]
        if np.any(q<=0) or not np.isfinite(q).all() or not np.isclose(q.sum(),1): raise ValueError('pmf')
        out.append(float(math.log(q[label]/p)))
    return out

def scan(case,n,seed,alphas,grid):
    ridge=RunningRidge(8); mlp=GeometricMLP(8,seed=seed+19)
    product=PredictableProduct(); second=RunningRidge(1)
    unis=[RunningRidge(1) for _ in range(8)]
    books={s:{w:[Bettor(s) for _ in range(8 if w=='univariate_mean' else 1)] for w in WITNESSES} for s in STRATEGIES}
    wealth={m:0. for m in METHODS};tau={m:{str(a):None for a in alphas} for m in METHODS}
    for k,(x,y) in enumerate(synthetic_pairs(case,n,seed)):
        pr=ridge.predict(x);pm=mlp.predict(x);xx=product.transform(x);ps=second.predict(xx)
        pay={'ridge':[swap_score(pr,y)],'mlp':[swap_score(pm,y)],'second_order':[swap_score(ps,y)],
             'univariate_mean':[swap_score(u.predict(x[:,j:j+1]),y) for j,u in enumerate(unis)]}
        # All witnesses and fractions fixed before orientation of current labels.
        for s in STRATEGIES:
            for w in WITNESSES:
                logs=[b.step(d) for b,d in zip(books[s][w],pay[w])]
                wealth[s+'_'+w]=mixture_logwealth(logs)
            wealth[s+'_mixture4']=mixture_logwealth([wealth[s+'_'+w] for w in WITNESSES])
        for w,obj in [('ridge',ridge),('mlp',mlp)]: wealth['lr_'+w+'_rowwise']+=sum(pointwise_lr(obj,x,y))
        for m in METHODS:
            for a in alphas:
                if tau[m][str(a)] is None and wealth[m]>=math.log(1/a): tau[m][str(a)]=2*(k+1)
        # No fit or centering update before all pair processes settle.
        ridge.learn(x,y);mlp.learn(x,y);second.learn(xx,y);product.learn(x)
        for j,u in enumerate(unis):u.learn(x[:,j:j+1],y)
    return {'tau':tau,'rate_per_row':{m:wealth[m]/n for m in METHODS}}

def randomization_p(payoffs,seed,b=999):
    """Independent within-pair flips; never permute rows across time.

    Valid when conditional orientations are jointly uniform given all X,
    unordered pair labels, and independent frozen training. Inclusive ties.
    """
    d=np.asarray(payoffs,dtype=np.float64)
    rng=np.random.default_rng(seed);obs=float(d.sum());ge=0
    tol=1e-12*max(1.,float(np.abs(d).sum()))
    for lo in range(0,b,64):
        signs=2*rng.integers(0,2,size=(min(64,b-lo),len(d)))-1
        ge+=int(np.count_nonzero(signs@d>=obs-tol))
    return (1+ge)/(b+1)

def matched(case,n,seed,cfg):
    model=RunningRidge(8)
    for x,y in synthetic_pairs(case,cfg['training_rows'],seed+10000000): model.learn(x,y)
    count=model.n_seen;books={s:Bettor(s) for s in ('plugin','ons_gain')}
    tau={s:{str(a):None for a in cfg['alphas']} for s in books};ds=[];fixed={}
    for k,(x,y) in enumerate(synthetic_pairs(case,n,seed)):
        d=swap_score(model.predict(x),y);ds.append(d);rows=2*(k+1)
        for s,b in books.items():
            val=b.step(d)
            for a in cfg['alphas']:
                if tau[s][str(a)] is None and val>=math.log(1/a):tau[s][str(a)]=rows
        if rows in cfg['N_grid']: fixed[str(rows)]=randomization_p(ds,seed+20000000+rows,cfg['randomizations'])
    if model.n_seen!=count:raise RuntimeError('matched predictor not frozen')
    return {'tau':tau,'fixed_N_p':fixed,'training_rows':count,'evaluation_rows':n}

def wilson(h,n):
    z=1.959963984540054;p=h/n;den=1+z*z/n;c=(p+z*z/(2*n))/den
    r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,c-r),min(1.,c+r)]

def curve(hits,n,grid):
    ps=[h/n for h in hits];found=None
    for i,v in enumerate(ps):
        if v>=.8 and all(t>=.8 for t in ps[i:]):found=grid[i];break
    return {'counts':hits,'replicates':n,'probability':ps,'pointwise_MC_Wilson95':[wilson(h,n) for h in hits],
            'grid_N80':found,'N80_status':'RIGHT_CENSORED' if found is None else ('AT_NMIN_BOUNDARY' if found==grid[0] else 'FINITE_GRID'),
            'N80_interval':'NOT_ESTIMATED'}

def c_summary(records,method,alpha,n):
    cats={'nonpositive_rate':0,'nonpositive_observed_crossing':0,'nonpositive_observed_censored':0,'finite_prediction_observed_crossing':0,'finite_prediction_observed_censored':0}
    errors=[];bounds=[];preds=[];joint=[]
    for r in records:
        rate=r['pilot']['rate_per_row'][method];t=r['scan']['tau'][method][str(alpha)]
        if rate<=0:
            cats['nonpositive_rate']+=1
            cats['nonpositive_observed_censored' if t is None else 'nonpositive_observed_crossing']+=1
            continue
        pred=math.log(1/alpha)/rate;preds.append(pred)
        if t is not None:
            cats['finite_prediction_observed_crossing']+=1;errors.append(abs(math.log(pred/t)))
        else:
            cats['finite_prediction_observed_censored']+=1
            # True tau>N. Infimum absolute log error is zero if pred>N.
            bounds.append(max(0.,math.log(n/pred)))
        joint.append({'prediction_beyond_Nmax':pred>n,'observed_censored':t is None})
    return {'categories':cats,'finite_prediction_fraction':len(preds)/len(records),
            'finite_prediction_beyond_Nmax':sum(p>n for p in preds),
            'prediction_horizon_by_observed_censoring':{f'pred_beyond_{b}_obs_censored_{c}':sum(z['prediction_beyond_Nmax']==b and z['observed_censored']==c for z in joint) for b in (False,True) for c in (False,True)},
            'finite_comparison_count':len(errors),'finite_comparison_log_MAE':float(np.mean(errors)) if errors else None,
            'censored_absolute_log_error_lower_bound_mean':float(np.mean(bounds)) if bounds else None,
            'censored_log_error_upper_bound':None,'overall_log_MAE':'NOT_IDENTIFIED',
            'scope':'Conditional finite comparisons only; swap rate is log-growth, not PI; LR is prequential log score.'}

def summarize(state,cfg):
    out={'status':'COMPLETE_SYNTHETIC_ONLY','config':cfg,'real_rows_read':0,'cases':{}}
    for case,rs in state['cases'].items():
        n=len(rs);row={'replicates':n,'adaptive':{},'matched':{}}
        for m in METHODS:
            row['adaptive'][m]={}
            for a in cfg['alphas']:
                hit=[sum(r['scan']['tau'][m][str(a)] is not None and r['scan']['tau'][m][str(a)]<=N for r in rs) for N in cfg['N_grid']]
                z=curve(hit,n,cfg['N_grid'])
                if case in cfg['positive_cases']: z['method_C']=c_summary(rs,m,a,cfg['N_max'])
                row['adaptive'][m][str(a)]=z
        for m in ('fixed_N_pair_randomization','plugin','ons_gain'):
            row['matched'][m]={}
            for a in cfg['alphas']:
                hit=[]
                for N in cfg['N_grid']:
                    hit.append(sum(r['matched']['fixed_N_p'][str(N)]<=a if m=='fixed_N_pair_randomization' else
                                   r['matched']['tau'][m][str(a)] is not None and r['matched']['tau'][m][str(a)]<=N for r in rs))
                row['matched'][m][str(a)]=curve(hit,n,cfg['N_grid'])
        out['cases'][case]=row
    return out

def identity(config):
    paths=list(Path('scripts/anytime_valid').glob('*.py'))+[Path('docs/SYNTHETIC_EXTENSION_v0_2.md'),Path('docs/ANYTIME_VALID_FILTRATION_CORRECTION_v0_1.md')]
    return {'config_sha256':sha(config),'files':{str(p):sha(p) for p in sorted(paths)}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--benchmark',action='store_true');p.add_argument('--max-records',type=int,default=16);a=p.parse_args()
    cfg=json.loads(a.config.read_text());a.run.mkdir(parents=True,exist_ok=True)
    ident=identity(a.config);cp=a.run/('BENCHMARK_CHECKPOINT.json' if a.benchmark else 'EXTENSION_CHECKPOINT.json')
    final=a.run/('BENCHMARK.json' if a.benchmark else 'SYNTHETIC_EXTENSION_RESULTS.json')
    if final.exists():raise FileExistsError('already completed')
    if not a.benchmark:
        freeze=json.loads((a.run/'INTERNAL_SYNTHETIC_FREEZE.json').read_text())
        if freeze['identity']!=ident:raise ValueError('pre-execution freeze mismatch')
    state=json.loads(cp.read_text()) if cp.exists() else {'identity':ident,'cases':{},'cpu_seconds':0.,'status':'RUNNING'}
    if state['identity']!=ident:raise ValueError('checkpoint identity mismatch')
    start=time.process_time();completed=0
    for ci,case in enumerate(cfg['null_cases']+cfg['positive_cases']):
        positive=case in cfg['positive_cases'];limit=1 if a.benchmark else cfg['positive_repetitions' if positive else 'null_repetitions']
        rows=state['cases'].setdefault(case,[])
        for rep in range(len(rows),limit):
            if state['cpu_seconds']+time.process_time()-start>cfg['main_cpu_cap_seconds']:raise RuntimeError('INCOMPLETE compute cap; checkpoint preserved')
            seed=cfg['seed']+ci*100000+rep;N=cfg['N_max']
            rec={'replicate':rep,'seed':seed,'scan':scan(case,N,seed,cfg['alphas'],cfg['N_grid']),
                 'matched':matched(case,N,seed,cfg)}
            if positive:rec['pilot']=scan(case,cfg['pilot_rows'],seed+30000000,cfg['alphas'],cfg['N_grid'])
            rows.append(rec);state['cpu_seconds']+=time.process_time()-start;start=time.process_time()
            state['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;atomic_json(cp,state);completed+=1
            if completed>=a.max_records:
                print(json.dumps({'status':'CHECKPOINTED','counts':{k:len(v) for k,v in state['cases'].items()},'cpu_seconds':state['cpu_seconds']}));return 0
    state['status']='COMPLETE';atomic_json(cp,state)
    out={'status':'BENCHMARK_ONLY','case_cpu_total':state['cpu_seconds'],'peak_rss_kib':state['peak_rss_kib'],'cases':list(state['cases'])} if a.benchmark else summarize(state,cfg)
    out['cpu_seconds']=state['cpu_seconds'];atomic_json(final,out);print(json.dumps({'status':out['status'],'cpu_seconds':state['cpu_seconds']}))

if __name__=='__main__':main()
