"""Derived portable numerical core; historical producer identity is in RESULT_SOURCES. No import-time dataset access."""
import math


from pathlib import Path


import numpy as np


from scipy.stats import binom


from scripts.core.legacy import read, new, canon, sha, labels, rows_from_ranges, require_complete, welch_accumulate, welch_value


from scripts.core.anytime.extension_v02 import wilson


STAGES = ('development', 'pilot', 'freeze', 'null', 'gate', 'natural', 'secondary', 'report')


def secondary_subset(manifest):
    wanted = {'ref_variable', 'ref_fixed', 'pqm4_variable', 'pqm4_fixed'}
    out = []
    seen = set()
    for bg in manifest['backgrounds']:
        if bg['key'] not in wanted or bg['key'] in seen:
            raise ValueError('unexpected/duplicate background campaign')
        seen.add(bg['key'])
        if len(bg['segments']) < 2:
            raise ValueError('two registered segments required')
        for s in bg['segments'][:2]:
            out.append(dict(capture=bg['key'], implementation=bg['implementation'],
                            background_id=s['id'], ranges=s['ranges'],
                            samples=13000 if bg['implementation']=='ref' else 20000))
    if seen != wanted or len({s['background_id'] for s in out}) != 8:
        raise ValueError('exactly eight distinct backgrounds required')
    return out


def gate_parameters(backgrounds, alpha, repetitions=256):
    if repetitions != 256 or backgrounds < 1 or alpha not in (.05, .01):
        raise ValueError('unregistered integrity gate')
    flagged = [k for k in range(repetitions+1) if wilson(k,repetitions)[0] > alpha]
    k = min(flagged)
    probability = float(binom.sf(k-1,repetitions,alpha))
    quantile = int(binom.ppf(.999,backgrounds,probability))
    return dict(alpha=alpha, backgrounds=backgrounds, repetitions=repetitions,
                minimum_flag_count=k, p_flag=probability, quantile_999=quantile,
                fail_if_flagged_backgrounds_greater_than=quantile,
                Wilson_z=1.959963984540054)


def gate_plan(manifest):
    counts = {im:sum(len(b['segments']) for b in manifest['backgrounds'] if b['implementation']==im) for im in ['ref','pqm4']}
    return dict(event='first_crossing_by_Nmax', Nmax=4096,
                rule='Wilson95 lower > alpha; FAIL iff family flag count > Binomial(B,p_flag) .999 quantile',
                families=[dict(implementation=im,target=t,model=m,bettor=s,**gate_parameters(counts[im],a))
                          for im in counts for t in ['mult_a','mult_b'] for m in ['ridge','mlp']
                          for s in ['plugin','ons_gain','lr'] for a in [.05,.01]])


def integrity_gate(output, plan):
    cp=require_complete(output,'null'); groups={}
    for i in range(len(cp['completed'])):
        r=read(output/'null'/f'{i:05d}.json');j=r['job']
        for m,methods in r['result'].items():
            for s,aa in methods.items():
                for a,events in aa.items():
                    e=events['crossing'];key=(j['capture'].split('_')[0],j['target'],m,s,float(a),j['background_id'])
                    x=groups.setdefault(key,[0,0]);x[0]+=e['counts'][-1];x[1]+=e['replicates']
    results=[]
    for f in plan['families']:
        key=(f['implementation'],f['target'],f['model'],f['bettor'],f['alpha'])
        tid=(0 if f['implementation']=='ref' else 2)+(f['target']=='mult_b')
        dev=read(output/'development'/f'{tid:05d}.json')['result']
        if dev['status']!='OK':
            results.append(dict(f,status='NOT_APPLICABLE',reason=dev['status']));continue
        matches={k[-1]:v for k,v in groups.items() if k[:-1]==key}
        if len(matches)!=f['backgrounds'] or any(v[1]!=256 for v in matches.values()):
            raise ValueError('integrity gate incomplete family')
        flags=sorted(bg for bg,(k,n) in matches.items() if wilson(k,n)[0]>f['alpha'])
        results.append(dict(f,flagged_backgrounds=flags,flag_count=len(flags),
                            status='FAIL' if len(flags)>f['quantile_999'] else 'PASS'))
    return dict(status='FAIL' if any(r['status']=='FAIL' for r in results) else 'PASS',
                null_checkpoint_sha256=sha(output/'NULL_CHECKPOINT.json'),families=results,
                scope='Integrity diagnostic, not a scientific endpoint or simultaneous scientific test')


def ensure_gate(output, plan):
    result=integrity_gate(output,plan);p=output/'INTEGRITY_GATE.json'
    if p.exists():
        if read(p)!=result:raise ValueError('integrity gate changed')
    else:new(p,result)
    if result['status']!='PASS':raise RuntimeError('STOP_B_INTEGRITY_GATE_FAIL')
    return result


def savings(times, horizon, budgets=(1024,2048,4096)):
    if not times or any(t is not None and (t<2 or t%2 or t>horizon) for t in times):
        raise ValueError('invalid first-crossing row')
    finite=np.array([t for t in times if t is not None],dtype=float)
    def quantiles(x):
        if not len(x):return dict(q25=None,median=None,q75=None,IQR=None,reason='NO_FINITE_STOPS')
        q=np.quantile(x,[.25,.5,.75],method='linear')
        return dict(q25=float(q[0]),median=float(q[1]),q75=float(q[2]),IQR=float(q[2]-q[0]))
    out=dict(denominator=len(times),horizon_rows=horizon,finite_stops=len(finite),
             censored_at_horizon=len(times)-len(finite),
             stopping_rows_conditional_on_finite=quantiles(finite),budgets={})
    for b in budgets:
        if b>horizon:raise ValueError('budget beyond observed horizon')
        stopped=finite[finite<b]
        out['budgets'][str(b)]=dict(stopped_strictly_before=int(len(stopped)),
             fraction_stopped_before=len(stopped)/len(times),equal_to_budget=int(np.sum(finite==b)),
             not_stopped_before=len(times)-len(stopped),
             stopping_rows_conditional_before_budget=quantiles(stopped),
             stop_over_budget_conditional_before_budget=quantiles(stopped/b),
             stop_over_budget_conditional_on_finite=quantiles(finite/b))
    out['interpretation']='Descriptive; noncrossings stay censored, never replaced by horizon as a stop'
    return out


def natural_savings(output):
    cp=require_complete(output,'natural');groups={}
    for i in range(len(cp['completed'])):
        r=read(output/'natural'/f'{i:05d}.json');j=r['job']
        if j['level']!=0:continue
        for model,methods in r['result'].items():
            for bettor,aa in methods.items():
                if bettor=='fixed_p':continue
                for alpha,e in aa.items():
                    key='|'.join([j['capture'],j['target'],model,bettor,alpha])
                    groups.setdefault(key,[]).extend(e['first_crossing_rows'])
    return {k:savings(v,4096) for k,v in groups.items()}


def secondary_kernel(x,Y,mean,scale,coef,intercept,budget,total_samples):
    """All-column Welch and sample-wise plugin, same labels and rows.

    x is one full-horizon column tile. Only aggregate per-replicate events leave
    this function; column tiles are OR-combined, never treated as repetitions.
    """
    n,d=x.shape;R=len(Y);p=np.clip((x-mean)/scale*coef+intercept,.005,.995)
    S=np.zeros((R,d));Q=np.ones((R,d));W=np.zeros((R,d));hits=np.zeros((2,R),bool)
    for k in range(n//2):
        if k%32==0:budget.check()
        D=(p[2*k]-p[2*k+1])[None,:]*(Y[:,2*k]-Y[:,2*k+1])[:,None]
        W+=np.log1p(np.clip(S/Q,0,.5)*D)
        if not np.isfinite(W).all():raise ArithmeticError('nonfinite secondary wealth')
        S+=D;Q+=D*D
        for j,a in enumerate([.05,.01]):hits[j]|=(W>=math.log(total_samples/a)).any(1)
    stats={};anylook=np.zeros(R,bool);terminal=None;undefined=0;start=0
    for stop in [1000,2000,3000,4000,4096]:
        budget.check();welch_accumulate(stats,x[start:stop],Y[:,start:stop].astype(float));v,valid=welch_value(stats)
        undefined+=int((~valid).sum());terminal=(np.abs(v)>4.5).any(1);anylook|=terminal;start=stop
    return dict(peeking=anylook,terminal=terminal,e=hits,undefined_sample_looks=undefined)


def secondary_job(provider,job,dev,cfg,budget):
    ids=rows_from_ranges(job['ranges']);w=dev['secondary_witness']
    Y=np.stack([labels(cfg['seed'],job['background_id'],job['target_id'],r,len(ids),dev['pi']) for r in range(job['rep_start'],job['rep_stop'])])
    result=dict(peeking=np.zeros(len(Y),bool),terminal=np.zeros(len(Y),bool),e=np.zeros((2,len(Y)),bool),undefined_sample_looks=0)
    for lo in range(0,job['samples'],cfg['secondary']['column_tile']):
        budget.check();hi=min(lo+cfg['secondary']['column_tile'],job['samples'])
        x=provider.traces(job['capture'],ids,lo,hi)
        e=secondary_kernel(x,Y,np.asarray(w['mean'][lo:hi]),np.asarray(w['scale'][lo:hi]),
                           np.asarray(w['coef'][lo:hi]),dev['pi'],budget,job['samples'])
        for k in ['peeking','terminal','e']:result[k]|=e[k]
        result['undefined_sample_looks']+=e['undefined_sample_looks'];del x
    return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}


def secondary_jobs(cfg,output):
    if cfg['secondary']['status']!='ACTIVE':return []
    out=[]
    for s in cfg['secondary']['backgrounds']:
        tid=0 if s['implementation']=='ref' else 2
        dev=read(output/'development'/f'{tid:05d}.json')['result']
        if dev['status']!='OK':continue
        for r in range(0,256,cfg['secondary']['replicate_tile']):
            out.append(dict(s,target_id=tid,rep_start=r,rep_stop=min(r+cfg['secondary']['replicate_tile'],256),
                            column_start=0,column_stop=s['samples']))
    return out


def secondary_report(output,cfg):
    cp=require_complete(output,'secondary');groups={}
    for i in range(len(cp['completed'])):
        r=read(output/'secondary'/f'{i:05d}.json');j=r['job'];v=r['result']
        g=groups.setdefault(str(j['background_id']),dict(peeking=np.zeros(256,bool),terminal=np.zeros(256,bool),e=np.zeros((2,256),bool),undefined_sample_looks=0))
        for key in ['peeking','terminal']:g[key][j['rep_start']:j['rep_stop']]|=v[key]
        g['e'][:,j['rep_start']:j['rep_stop']]|=v['e'];g['undefined_sample_looks']+=v['undefined_sample_looks']
    out={}
    for bg,g in groups.items():
        counts={k:int(g[k].sum()) for k in ['peeking','terminal']}
        counts.update({f'e_Bonferroni_{a}':int(g['e'][i].sum()) for i,a in enumerate([.05,.01])})
        out[bg]=dict(replicates=256,counts=counts,fractions={k:v/256 for k,v in counts.items()},
                    pointwise_Wilson95={k:wilson(v,256) for k,v in counts.items()},undefined_sample_looks=g['undefined_sample_looks'])
    return dict(status=cfg['secondary']['status'],backgrounds=out,
                scope='Conditional MC on eight position-selected fixed backgrounds; threshold 4.5 not calibrated alpha')

