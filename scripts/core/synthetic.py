"""Derived portable numerical core; historical producer identity is in RESULT_SOURCES. No import-time dataset access."""
import argparse


import datetime


import hashlib


import json


import math


from pathlib import Path


import resource


from statistics import NormalDist


import sys


import time


import numpy as np


from scripts.core.anytime.common import atomic_json,sha


from scripts.core.anytime.extension_v02 import Bettor,synthetic_pairs,wilson


from scripts.core.synthetic_base import (generate,fit_witness,payoffs,transform_path,payoff_diagnostics,select_amplitude,terminal_ratio)


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()


def new_json(path,obj):
    if path.exists():raise FileExistsError(path)
    atomic_json(path,obj)


def dense_grid(nmax):
    if nmax not in (8192,16384):raise ValueError('only frozen endpoints')
    count=round(4*math.log2(nmax/256))
    return sorted(set([256,nmax]+[2*math.floor(256*2**(j/4)/2+.5) for j in range(count+1)]))


def fixed_p_paths(paths,grid,seed,B=999):
    """Common independent pair-sign paths, nested prefixes; no row permutation."""
    lengths=[len(x) for x in paths.values()]
    if not lengths or len(set(lengths))!=1 or any(n%2 or n//2>lengths[0] for n in grid):raise ValueError('grid/path alignment')
    rng=np.random.default_rng(seed);indices=np.array(grid)//2-1;hits={k:np.zeros(len(grid),int) for k in paths}
    obs={k:np.cumsum(d)[indices] for k,d in paths.items()}
    tolerance={k:1e-12*np.maximum(1,np.cumsum(abs(d))[indices]) for k,d in paths.items()}
    for lo in range(0,B,64):
        signs=2*rng.integers(0,2,size=(min(64,B-lo),lengths[0]))-1
        for name,d in paths.items():
            scores=np.cumsum(signs*d,axis=1)[:,indices]
            hits[name]+=(scores>=obs[name]-tolerance[name]).sum(axis=0)
    return {name:((1+count)/(B+1)).tolist() for name,count in hits.items()}


def wealth_path(d,strategy,grid,alphas):
    b=Bettor(strategy);logs=[];tau={str(a):None for a in alphas};grid=set(grid)
    for j,v in enumerate(d,1):
        value=b.step(float(v))
        if not math.isfinite(value):raise ArithmeticError('nonfinite wealth')
        for a in alphas:
            if tau[str(a)] is None and value>=math.log(1/a):tau[str(a)]=2*j
        if 2*j in grid:logs.append(value)
    return {'terminal_logwealth':logs,'first_crossing_rows':tau}


def simulation(case,amplitude,seed,witness,cfg,grid,mode='main'):
    N=grid[-1]
    if case in cfg['scenarios']:x,y=generate(case,N,amplitude,seed)
    else:
        pairs=list(synthetic_pairs(case,N,seed));x=np.concatenate([p[0] for p in pairs]);y=np.concatenate([p[1] for p in pairs]);del pairs
    d=payoffs(witness,x,y);del x,y
    if mode=='pilot':return {'fixed_p':{'raw':fixed_p_paths({'raw':d},grid,seed+20000000,cfg['randomizations'])['raw']}}
    f,_=transform_path(d,cfg['scale']);paths={'raw':d,'odd_scale_adaptive':f}
    out={'betting':{name:{s:wealth_path(path,s,grid,cfg['alpha']) for s in cfg['strategies']} for name,path in paths.items()}}
    if mode!='prediction':out['fixed_p']=fixed_p_paths(paths,grid,seed+20000000,cfg['randomizations'])
    else:out['moments']={name:payoff_diagnostics(path) for name,path in paths.items()}
    return out


def grid_n80(probabilities,grid):
    n=next((i for i in range(len(grid)) if all(p>=.8 for p in probabilities[i:])),None)
    return {'grid_N80':None if n is None else grid[n],
            'status':'RIGHT_CENSORED' if n is None else ('AT_NMIN_BOUNDARY' if n==0 else 'FINITE_GRID'),
            'N80_CI':'NOT_ESTIMATED','interpolation':'NOT_PERFORMED'}


def curve(values,grid):
    a=np.asarray(values,bool);hits=a.sum(axis=0).tolist();R=len(a);p=(a.mean(axis=0)).tolist()
    return dict(counts=hits,replicates=R,probability=p,pointwise_MC_Wilson95=[wilson(h,R) for h in hits],**grid_n80(p,grid))


def summarize_main(records,cfg):
    grid=cfg['main_grid'];out={}
    for case in cfg['null_scenarios']+cfg['scenarios']:
        rows=[r['result'] for r in records if r['case']==case];group={}
        for payoff in cfg['payoffs']:
            group[payoff]={}
            for a in cfg['alpha']:
                key=str(a);z={'fixed_N':curve([[p<=a for p in r['fixed_p'][payoff]] for r in rows],grid)}
                for s in cfg['strategies']:
                    q=[r['betting'][payoff][s] for r in rows]
                    z[s]={'terminal_exceedance':curve([[v>=math.log(1/a) for v in r['terminal_logwealth']] for r in q],grid),
                          'first_crossing':curve([[r['first_crossing_rows'][key] is not None and r['first_crossing_rows'][key]<=N for N in grid] for r in q],grid)}
                group[payoff][key]=z
        out[case]=group
    return out


def summarize_predictions(records,cfg):
    out={};grid=cfg['main_grid']
    for case in cfg['scenarios']:
        rows=[r['result'] for r in records if r['case']==case];result={}
        for name in cfg['payoffs']:
            diagnostics=[r['moments'][name] for r in rows]
            eligible=name=='raw' and all(not m['cap_active'] and m['small_step_diagnostic'] and m['mu']>0 and
                       m['E_D2_over_variance'] is not None and m['E_D2_over_variance']<=1.05 for m in diagnostics)
            pred={'moment_diagnostics':diagnostics,'constant_bet_ratio_reference':{'status':'APPROXIMATION_DIAGNOSTICS_PASSED' if eligible else 'NOT_APPLICABLE',
                      'coefficient_alpha05_power08':terminal_ratio() if eligible else None,
                      'scope':'Hypothetical IID weak-signal constant-bet TERMINAL reference only; never a first-crossing or universal ONS/plugin ratio'},'terminal_wealth_approximation':{}}
            for s in cfg['strategies']:
                arr=np.array([r['betting'][name][s]['terminal_logwealth'] for r in rows]);means=arr.mean(0);var=arr.var(0,ddof=1)
                pred['terminal_wealth_approximation'][s]={}
                for a in cfg['alpha']:
                    probs=[float(mu>=math.log(1/a)) if v==0 else NormalDist().cdf((mu-math.log(1/a))/math.sqrt(v)) for mu,v in zip(means,var)]
                    pred['terminal_wealth_approximation'][s][str(a)]={'probability':probs,'mean_terminal_logwealth':means.tolist(),'variance_terminal_logwealth':var.tolist(),
                        'estimation_replicates':len(rows),'conditioning':'fixed scenario witness','uncertainty':'16-stream moment-estimation uncertainty NOT quantified by a transfer/N80 CI',**grid_n80(probs,grid)}
            result[name]=pred
        out[case]=result
    return out


def comparisons(pred,main,cfg):
    rows=[]
    for case in cfg['scenarios']:
        for payoff in cfg['payoffs']:
            for s in cfg['strategies']:
                for a in cfg['alpha']:
                    p=pred[case][payoff]['terminal_wealth_approximation'][s][str(a)]
                    observed=main[case][payoff][str(a)][s]['terminal_exceedance']
                    errors=np.asarray(p['probability'])-observed['probability'];pn=p['grid_N80'];on=observed['grid_N80']
                    rows.append({'case':case,'payoff':payoff,'strategy':s,'alpha':a,'compared_event':'terminal_exceedance_only',
                        'curve_errors':errors.tolist(),'curve_MAE':float(abs(errors).mean()),'curve_RMSE':float(np.sqrt(np.mean(errors**2))),
                        'predicted_grid_N80':pn,'predicted_status':p['status'],'observed_grid_N80':on,'observed_status':observed['status'],
                        'finite_grid_log_error':abs(math.log(pn/on)) if pn is not None and on is not None else None})
    return rows


def jobs(stage,cfg,selection):
    if stage=='pilot':
        return [{'case':case,'amplitude':amp,'replicate':r,'seed':cfg['seed']+10000+1000*i+100*j+r}
            for i,case in enumerate(cfg['scenarios']) for j,amp in enumerate(cfg['candidate_amplitudes']) for r in range(16)]
    if stage=='prediction':return [{'case':case,'amplitude':selection[case]['amplitude'],'replicate':r,'seed':cfg['seed']+2000000+100000*i+r}
            for i,case in enumerate(cfg['scenarios']) for r in range(16)]
    return ([{'case':case,'amplitude':None,'replicate':r,'seed':cfg['seed']+4000000+100000*i+r} for i,case in enumerate(cfg['null_scenarios']) for r in range(512)]+
            [{'case':case,'amplitude':selection[case]['amplitude'],'replicate':r,'seed':cfg['seed']+1000000+100000*i+r} for i,case in enumerate(cfg['scenarios']) for r in range(256)])

