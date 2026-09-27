"""Derived portable numerical core; historical producer identity is in RESULT_SOURCES. No import-time dataset access."""
import argparse


from collections import deque


import json


import math


from pathlib import Path


import resource


from statistics import NormalDist


import time


import numpy as np


from scripts.core.anytime.common import atomic_json,sha


from scripts.core.anytime.extension_v02 import Bettor,randomization_p


class OddScale:
    def __init__(self,config):self.cfg=config;self.history=deque(maxlen=config['past_window_pairs'])
    def step(self,d):
        if not math.isfinite(d):raise ValueError('finite payoff required')
        c=self.cfg
        if len(self.history)<c['initial_pairs']:scale=c['initial_scale']
        else:
            q=np.quantile(self.history,c['quantiles'],method=c['quantile_method']);scale=max(c['epsilon'],float(q[1]-q[0]))
        value=math.tanh(d/scale);self.history.append(abs(d));return value,scale


def transform_path(d,cfg):
    obj=OddScale(cfg);rows=[obj.step(float(v)) for v in d]
    return np.asarray(rows).T


def terminal_ratio(alpha=.05,power=.8):
    norm=NormalDist();zb=norm.inv_cdf(power);za=norm.inv_cdf(1-alpha)
    return (zb+math.sqrt(zb*zb+2*math.log(1/alpha)))**2/(za+zb)**2


def payoff_diagnostics(d):
    d=np.asarray(d,float)
    if d.ndim!=1 or not len(d) or not np.isfinite(d).all() or np.max(np.abs(d))>1+1e-12:raise ValueError('bounded finite vector')
    mu=float(d.mean());second=float(np.mean(d*d));variance=float(np.var(d))
    uncapped=mu/second if second>0 else 0.;lam=float(np.clip(uncapped,0,.5));step=np.abs(lam*d)
    log=np.log1p(lam*d);small=bool(step.max()<=.1);cap=bool(uncapped>.5)
    return {'mu':mu,'E_D_squared':second,'variance':variance,'lambda_uncapped':uncapped,
        'lambda_quadratic':lam,'cap_active':cap,'max_abs_lambda_D':float(step.max()),'p95_abs_lambda_D':float(np.quantile(step,.95)),
        'small_step_diagnostic':small,'g_quadratic':lam*mu-.5*lam*lam*second,
        'mean_log_increment':float(log.mean()),'variance_log_increment':float(log.var()),
        'E_D2_over_variance':second/variance if variance>0 else None,
        'unconstrained_g':mu*mu/(2*variance) if variance>0 and mu>0 and not cap and small else None,
        'scope':'moments of THIS payoff only; descriptive for adaptive scaling, conditional on frozen witness'}


def terminal_probability(n_pairs,g,v,alpha):
    if n_pairs<=0 or v<0 or not all(map(math.isfinite,[g,v])):raise ValueError('invalid moments')
    if v==0:return float(n_pairs*g>=math.log(1/alpha))
    return NormalDist().cdf((n_pairs*g-math.log(1/alpha))/math.sqrt(n_pairs*v))


def feature(case,x):
    if case=='linear':return x[:,0]
    if case=='nonlinear':return x[:,0]**2
    if case=='second_order':return x[:,0]*x[:,1]
    raise ValueError(case)


def generate(case,n,amplitude,seed):
    """Synthetic single-stream allocation only, never a multi-replicate array."""
    ss=np.random.SeedSequence(seed);rx,ry=(np.random.default_rng(s) for s in ss.spawn(2))
    y=ry.integers(2,size=n);x=rx.normal(size=(n,2));u=rx.choice([-1.,1.],size=n)
    if case=='linear':x[:,0]+=amplitude*(2*y-1)
    elif case=='nonlinear':x[:,0]=u*(1+amplitude*y)+.25*x[:,0]
    elif case=='second_order':x[:,0]=u+.2*x[:,0];x[:,1]=amplitude*u*(2*y-1)+x[:,1]
    else:raise ValueError(case)
    return x,y


def fit_witness(case,cfg,seed):
    x,y=generate(case,cfg['training_rows_per_scenario'],cfg['training_reference_amplitude'],seed)
    z=feature(case,x);mean=float(z.mean());scale=float(z.std())
    if scale==0:raise ValueError('constant training feature')
    design=np.column_stack([np.ones(len(z)),(z-mean)/scale])
    coef=np.linalg.solve(design.T@design+np.diag([0.,1.]),design.T@y)
    return {'scenario':case,'mean':mean,'scale':scale,'coefficients':coef.tolist(),'training_rows':len(y),'conditioning':'single frozen witness'}


def payoffs(witness,x,y):
    if len(y)%2 or len(x)!=len(y):raise ValueError('complete aligned pairs')
    z=(feature(witness['scenario'],x)-witness['mean'])/witness['scale']
    c=witness['coefficients'];p=np.clip(c[0]+c[1]*z,.005,.995)
    # Stable binary swap score, exactly zero for equal labels.
    return (p[::2]-p[1::2])*(y[::2]-y[1::2])


def event_path(d,strategy,grid,alphas):
    b=Bettor(strategy);maximum=0.;out={}
    for k,v in enumerate(d,1):
        log=b.step(float(v));maximum=max(maximum,log)
        if 2*k in grid:out[str(2*k)]={str(a):{'terminal_exceedance':log>=math.log(1/a),'first_crossing':maximum>=math.log(1/a)} for a in alphas}
    return out


def select_amplitude(candidates,cfg):
    if len(candidates)!=len(cfg['candidate_amplitudes']) or set(r['amplitude'] for r in candidates)!=set(cfg['candidate_amplitudes']):raise ValueError('exact frozen candidates required')
    eligible=[r for r in candidates if r['grid_N80'] is not None and 1024<=r['grid_N80']<=4096]
    if not eligible:return {'amplitude':cfg['fallback_amplitude'],'status':cfg['fallback_status']}
    best=min(eligible,key=lambda r:(abs(math.log(r['grid_N80']/2048)),r['amplitude']))
    return {'amplitude':best['amplitude'],'status':'PILOT_DIFFICULTY_TARGET_ATTAINED_ONLY','pilot_grid_N80':best['grid_N80']}

