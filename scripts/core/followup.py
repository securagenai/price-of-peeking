"""Derived portable numerical core; historical producer identity is in RESULT_SOURCES. No import-time dataset access."""
import argparse,datetime as dt,hashlib,json,math,os,resource,time,zipfile,stat


from pathlib import Path,PurePosixPath


import numpy as np


from threadpoolctl import threadpool_limits


from scripts.core.real import read,sha,new,Provider,rows_from_ranges,Budget


from scripts.core.legacy import welch_accumulate,welch_value


IDS=[24,25,86,87];ALPHAS=[.05,.01];LOOKS=[1000,2000,3000,4000,4096]


IDS=[24,25,86,87];ALPHAS=[.05,.01];LOOKS=[1000,2000,3000,4000,4096]


IDS=[24,25,86,87];ALPHAS=[.05,.01];LOOKS=[1000,2000,3000,4000,4096]


def natural_label(hw,dev):return (np.asarray(hw)>=dev['median']).astype(np.int8)


def predict(x,w,pi,lo,hi):
    return np.clip((x-np.asarray(w['mean'][lo:hi]))/np.asarray(w['scale'][lo:hi])*np.asarray(w['coef'][lo:hi])+pi,.005,.995)


def step(S,Q,W,d):
    W=W+np.log1p(np.clip(S/Q,0,.5)*d)
    if not np.isfinite(W).all():raise ArithmeticError('nonfinite wealth')
    return S+d,Q+d*d,W


def kernel(x,y,w,pi,lo,total,budget):
    n,d=x.shape;p=predict(x,w,pi,lo,lo+d);S=np.zeros(d);Q=np.ones(d);W=np.zeros(d)
    first={str(a):None for a in ALPHAS};sample=dict(first)
    for k in range(n//2):
        if k%32==0:budget.check()
        S,Q,W=step(S,Q,W,(p[2*k]-p[2*k+1])*int(y[2*k]-y[2*k+1]))
        for a in ALPHAS:
            ii=np.flatnonzero(W>=math.log(total/a))
            if first[str(a)] is None and len(ii):first[str(a)]=2*k+2;sample[str(a)]=lo+int(ii[0])
    eb={str(a):dict(rejected=first[str(a)] is not None,first_row=first[str(a)],first_sample=sample[str(a)],terminal_samples=int((W>=math.log(total/a)).sum()),max_log_e_terminal=float(W.max()),log_threshold=math.log(total/a)) for a in ALPHAS}
    stats={};start=0;wf=None;ws=None;undef=0
    for stop in LOOKS:
        budget.check();welch_accumulate(stats,x[start:stop],y[None,start:stop].astype(float));v,valid=welch_value(stats);v=v[0];valid=valid[0];undef+=int((~valid).sum())
        ii=np.flatnonzero(valid & (abs(v)>4.5))
        if wf is None and len(ii):wf=stop;ws=lo+int(ii[0])
        start=stop
    max_t=float(np.max(abs(v[valid]))) if valid.any() else None
    common=dict(terminal_samples=len(ii),max_abs_t_terminal=max_t,undefined_sample_looks=undef)
    return dict(e=eb,welch_any=dict(common,rejected=wf is not None,first_row=wf,first_sample=ws),welch_terminal=dict(common,rejected=bool(len(ii)),first_row=n if len(ii) else None,first_sample=lo+int(ii[0]) if len(ii) else None))


def merge(parts):
    out={}
    for key in ['welch_any','welch_terminal','e_0.05','e_0.01']:
        vals=[x['e'][key[2:]] if key.startswith('e_') else x[key] for x in parts]
        hits=[(x['first_row'],x['first_sample']) for x in vals if x['rejected']];first=min(hits) if hits else (None,None)
        r=dict(rejected=bool(hits),first_row=first[0],first_sample=first[1],terminal_samples=sum(x['terminal_samples'] for x in vals))
        if key.startswith('e_'):
            r.update(max_log_e_terminal=max(x['max_log_e_terminal'] for x in vals),log_threshold=vals[0]['log_threshold'])
            r['terminal_gap_nats']=r['log_threshold']-r['max_log_e_terminal']
        else:
            finite=[x['max_abs_t_terminal'] for x in vals if x['max_abs_t_terminal'] is not None]
            r.update(max_abs_t_terminal=max(finite) if finite else None,undefined_sample_looks=sum(x['undefined_sample_looks'] for x in vals))
        out[key]=r
    return out

