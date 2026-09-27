"""Derived portable numerical core; historical producer identity is in RESULT_SOURCES. No import-time dataset access."""
import argparse


import copy


import datetime as dt


import hashlib


import json


import math


import os


from pathlib import Path


import resource


import time


import numpy as np


from sklearn.neural_network import MLPRegressor


from threadpoolctl import threadpool_limits


from scripts.core.anytime.common import atomic_json, sha


from scripts.core.synthetic import grid_n80


from scripts.core.amendment import ensure_gate, natural_savings, secondary_jobs, secondary_job, secondary_report


NAMESPACE={'training':11,'pilot':23,'background':37,'null':41,'noise':53,'randomization':67,'synthetic':79}


def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()


def read(p):
    def bad(x): raise ValueError(x)
    return json.loads(Path(p).read_text(),parse_constant=bad)


def canon(x): return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()


def new(p,x):
    p=Path(p)
    if p.exists(): raise FileExistsError(p)
    atomic_json(p,x)


def rng(seed,domain,*ids):
    if domain not in NAMESPACE or any(int(i)!=i or i<0 for i in ids): raise ValueError('seed namespace')
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed,spawn_key=(NAMESPACE[domain],*map(int,ids)))))


def labels(seed,background,target,replicate,n,pi):
    if not 0<pi<1: raise ValueError('DEGENERATE_TARGET')
    return (rng(seed,'null',NAMESPACE['background'],background,target,replicate).random(n)<pi).astype(np.int8)


def hw16(a,family):
    a=np.asarray(a)
    if family=='ref':
        if a.dtype!=np.uint8 or a.ndim!=2 or a.shape[1]!=512: raise ValueError('ref requires uint8[rows,512]')
        u=a[:,0].astype(np.uint16)|(a[:,1].astype(np.uint16)<<8)
    else:
        if a.dtype!=np.int16 or a.ndim!=2 or a.shape[1]!=256: raise ValueError('pqm4 requires int16[rows,256]')
        u=a[:,0].view(np.uint16)
    lut=np.array([int(i).bit_count() for i in range(256)],np.uint8)
    return lut[u&255]+lut[u>>8]


def threshold(h):
    m=float(np.median(h));y=(np.asarray(h)>=m).astype(np.int8);pi=float(y.mean())
    return {'median':m,'pi':pi,'status':'OK' if 0<pi<1 else 'DEGENERATE_TARGET'}


def window(p,d,w=17):
    if d<w or not 0<=p<d: raise ValueError('window')
    lo=min(max(int(p)-w//2,0),d-w);return [lo,lo+w]


def header(p):
    """Reads .npy header only, never maps or reads payload."""
    p=Path(p)
    with p.open('rb') as f:
        v=np.lib.format.read_magic(f)
        if v==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f)
        elif v==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f)
        else: raise ValueError('unsupported npy version')
        n=f.tell();f.seek(0);h=hashlib.sha256(f.read(n)).hexdigest()
    s=p.stat()
    if dtype.hasobject: raise ValueError('pickle forbidden')
    return {'shape':list(shape),'dtype':str(dtype),'fortran':order,'header_sha256':h,'header_bytes':n,
            'size':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev}


def rows_from_ranges(ranges):
    return [(x['chunk'],r) for x in ranges for r in range(x['start'],x['stop'])]


def segments(chunks,rows=10000,length=4096):
    ids=[(c,r) for c in chunks for r in range(rows)];out=[]
    for lo in range(0,len(ids)-length+1,length):
        group=ids[lo:lo+length];spans=[]
        for c,r in group:
            if spans and spans[-1]['chunk']==c and spans[-1]['stop']==r:spans[-1]['stop']=r+1
            else:spans.append(dict(chunk=c,start=r,stop=r+1))
        out.append(spans)
    return out,ids[len(out)*length:]


def check_ledger(ledger):
    seen=set()
    for s in ledger['backgrounds']:
        if s['implementation'] not in ['ref','pqm4'] or s['capture'] not in ['variable_key','fixed_key']:raise ValueError('campaign excluded')
        key=(s['implementation'],s['capture']);used=set()
        exposures={tuple(r) for r in s.get('prior_selection_ids',[])}
        exposures.update(rows_from_ranges(s.get('prior_selection_ranges',[])))
        exposures.update(rows_from_ranges(s.get('prior_waveform_ranges',[])))
        for segment in s['segments']:
            ids=rows_from_ranges(segment['ranges'])
            if len(ids)!=ledger['segment_rows'] or len(set(ids))!=len(ids) or used.intersection(ids):raise ValueError('segment overlap/size')
            if s['capture']=='variable_key' and (any(c not in range(2,10) for c,r in ids) or exposures.intersection(ids)):raise ValueError('evaluation exposure/reserve violation')
            used.update(ids)
            global_ids={(key,c,r) for c,r in ids}
            if seen.intersection(global_ids):raise ValueError('duplicate allocation')
            seen.update(global_ids)
    return True


class Incomplete(RuntimeError): pass


class Budget:
    def __init__(self,wall=25,cpu=14400,memory=1536):
        self.wall=wall;self.cpu=cpu;self.memory=memory;self.start=time.monotonic();self.cstart=time.process_time()
    def check(self):
        if time.monotonic()-self.start>=self.wall or time.process_time()-self.cstart>=self.cpu:raise Incomplete('CPU_OR_WALL_BUDGET')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>self.memory*1024:raise Incomplete('MEMORY_BUDGET')


class Provider:
    def __init__(self,root,inventory,approval,candidate_hash):
        if approval.get('candidate_sha256')!=candidate_hash or approval.get('approved') is not True or approval.get('scope')!='REGISTERED_STAGED_EXECUTION':raise PermissionError('explicit matching human approval required')
        self.root=Path(root);self.inventory=inventory
    def _load(self,key,kind,c):
        item=self.inventory['captures'][key]['files'][kind][str(c)];p=self.root/item['relative_path']
        if header(p)!=item['header']:raise ValueError('input header/stat identity changed')
        return np.load(p,mmap_mode='r',allow_pickle=False)
    def metadata(self,key,field,ids):
        family=self.inventory['captures'][key]['implementation'];out=np.empty(len(ids),np.uint8)
        for c in sorted({x[0] for x in ids}):
            ix=[i for i,(ch,r) in enumerate(ids) if ch==c];rr=[ids[i][1] for i in ix]
            out[ix]=hw16(self._load(key,field,c)[rr],family)
        return out
    def traces(self,key,ids,lo,hi):
        out=np.empty((len(ids),hi-lo),np.float64)
        for c in sorted({x[0] for x in ids}):
            ix=[i for i,(ch,r) in enumerate(ids) if ch==c];rr=[ids[i][1] for i in ix]
            out[ix]=self._load(key,'traces',c)[rr,lo:hi]
        if not np.isfinite(out).all():raise ArithmeticError('nonfinite traces')
        return out


def scan(provider,key,ids,y,d,budget,batch=512):
    """Pearson sufficient statistics: bounded row/column tiles, fit rows only."""
    if not np.var(y)>0:raise ValueError('constant target')
    n=len(y);sy=float(np.sum(y));vy=float(np.dot(y,y)-sy*sy/n);best=(-1.,0);means=[];sds=[];univariate=[]
    for lo in range(0,d,batch):
        hi=min(d,lo+batch);sx=np.zeros(hi-lo);ss=sx.copy();xy=sx.copy()
        for start in range(0,n,batch):
            budget.check();x=provider.traces(key,ids[start:start+batch],lo,hi);yy=y[start:start+batch]
            sx+=x.sum(0);ss+=(x*x).sum(0);xy+=yy@x
        var=np.maximum(0,ss-sx*sx/n);den=np.sqrt(vy*var)
        r=np.divide(xy-sy*sx/n,den,out=np.zeros_like(den),where=den>0)
        if not np.isfinite(r).all():raise ArithmeticError('nonfinite Pearson')
        j=int(np.argmax(abs(r)));candidate=(float(abs(r[j])),lo+j)
        if candidate[0]>best[0]:best=candidate
        means.extend((sx/n).tolist());sd=np.sqrt(var/n);sds.extend(sd.tolist())
        scale=np.where(sd>0,sd,1.)
        univariate.extend(((xy-sy*sx/n)/scale/(var/(scale*scale)+1)).tolist())
    if best[0]==0:raise ValueError('NO_NONCONSTANT_ASSOCIATION_FOR_POI')
    return {'poi':best[1],'abs_r':best[0],'mean':means,'sd':sds,'univariate_coef':univariate}


def probability(model,x):
    z=(np.asarray(x,float)-model['mean'])/model['scale']
    if model['kind']=='ridge':p=z@np.asarray(model['coef'])+model['intercept']
    else:
        for i,(w,b) in enumerate(zip(model['weights'],model['biases'])):
            z=z@np.asarray(w)+b
            if i<len(model['weights'])-1:z=np.maximum(z,0)
        p=z[:,0]
    if not np.isfinite(p).all():raise ArithmeticError('nonfinite prediction')
    return np.clip(p,.005,.995)


def fit_models(x,y,t,yt,seed,cfg,budget):
    mean=x.mean(0);sd=x.std(0);scale=np.where(sd>0,sd,1.);z=(x-mean)/scale;tz=(t-mean)/scale
    common={'mean':mean.tolist(),'scale':scale.tolist(),'waveform_sd':sd.tolist(),'probability_map':'clip(binary regression score,.005,.995); q0=1-q1'}
    coef=np.linalg.solve(z.T@z+np.eye(z.shape[1]),z.T@(y-y.mean()))
    ridge=dict(common,kind='ridge',coef=coef.tolist(),intercept=float(y.mean()))
    candidates=[]
    for width in cfg['mlp_widths']:
        m=MLPRegressor(hidden_layer_sizes=(width,),activation='relu',solver='adam',alpha=.001,learning_rate_init=.001,
                       batch_size=min(128,len(x)),max_iter=1,shuffle=True,random_state=int(seed+width),tol=0.,n_iter_no_change=cfg['epochs']+1)
        best=None;bestloss=math.inf;epochbest=0;stale=0
        for ep in range(cfg['epochs']):
            budget.check();m.partial_fit(z,y);p=np.clip(m.predict(tz),.005,.995);loss=float(np.mean((p-yt)**2))
            if not math.isfinite(loss):raise ArithmeticError('MLP loss')
            if loss<bestloss-1e-12:bestloss=loss;best=copy.deepcopy(m);epochbest=ep+1;stale=0
            else:stale+=1
            if stale>=cfg['patience']:break
        candidates.append((bestloss,width,epochbest,best))
    loss,width,ep,m=min(candidates,key=lambda z:z[:3]);mlp=dict(common,kind='mlp',weights=[a.tolist() for a in m.coefs_],biases=[a.tolist() for a in m.intercepts_],width=width,epoch=ep,tuning_mse=loss,
        candidates=[dict(mse=a,width=b,epoch=c) for a,b,c,d in candidates])
    return {'ridge':ridge,'mlp':mlp}


def swap(p,y):return (p[::2]-p[1::2])*(y[::2]-y[1::2])


def fixed_p(d,grid,random,B,budget):
    at=np.asarray(grid)//2-1;obs=np.cumsum(d)[at];hits=np.zeros(len(grid),int);tol=1e-12*np.maximum(1,np.cumsum(abs(d))[at])
    for i in range(0,B,64):
        budget.check();sign=2*random.integers(0,2,(min(64,B-i),len(d)))-1
        hits+=(np.cumsum(sign*d,axis=1)[:,at]>=obs-tol).sum(0)
    return ((hits+1)/(B+1)).tolist()


def epaths(p,Y,grid,alphas,pi,budget):
    """Batched replicates, frozen row-separable p; bets use settled pairs only."""
    p=np.asarray(p,float)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any((p<=0)|(p>=1)):raise ValueError('proper interior binary probabilities required')
    if grid!=sorted(set(grid)) or any(n%2 or n<2 or n>len(p) for n in grid):raise ValueError('pair grid')
    Y=np.asarray(Y);Y=Y[None,:] if Y.ndim==1 else Y
    if len(p)%2 or Y.shape[1]!=len(p) or not np.isin(Y,[0,1]).all():raise ValueError('paired binary rows')
    R=len(Y);log={s:np.zeros(R) for s in ['plugin','ons_gain']};fraction=np.zeros(R);A=np.ones(R);sd=np.zeros(R);s2=np.ones(R)
    if pi is not None:
        if not 0<pi<1:raise ValueError('DEGENERATE_TARGET')
        log['lr']=np.zeros(R)
    output={s:{str(a):{'terminal':[],'crossing':[]} for a in alphas} for s in log}
    hit={s:{str(a):np.zeros(R,bool) for a in alphas} for s in log};gridset=set(grid)
    first={s:{str(a):[None]*R for a in alphas} for s in log}
    for j in range(len(p)//2):
        if j%32==0:budget.check()
        d=(p[2*j]-p[2*j+1])*(Y[:,2*j]-Y[:,2*j+1])
        lam=np.clip(sd/s2,0,.5);log['plugin']+=np.log1p(lam*d);log['ons_gain']+=np.log1p(fraction*d)
        # Settlement precedes all updates. Literal historical ONS is not used.
        sd+=d;s2+=d*d;z=-d/(1+fraction*d);A+=z*z;fraction=np.clip(fraction-2/(2-math.log(3))*z/A,0,.5)
        if pi is not None:
            for row in [2*j,2*j+1]:
                log['lr']+=np.where(Y[:,row],math.log(p[row]/pi),math.log((1-p[row])/(1-pi)))
        for s in log:
            if not np.isfinite(log[s]).all():raise ArithmeticError('nonfinite wealth')
            for a in alphas:
                now=log[s]>=math.log(1/a)
                if pi is None:
                    for rep in np.flatnonzero(now & ~hit[s][str(a)]):first[s][str(a)][int(rep)]=2*j+2
                hit[s][str(a)]|=now
                if 2*j+2 in gridset:
                    output[s][str(a)]['terminal'].append(now.tolist());output[s][str(a)]['crossing'].append(hit[s][str(a)].tolist())
    result={s:{a:{e:np.asarray(v,bool).T.tolist() for e,v in events.items()} for a,events in alphas_.items()} for s,alphas_ in output.items()}
    if pi is None:
        for s in result:
            for a in result[s]:result[s][a]['first_crossing_rows']=first[s][a]
    return result


def select_noise(curves):
    eligible=[(k,c) for k,c in curves.items() if c['grid_N80'] is not None and 512<=c['grid_N80']<=2048]
    if not eligible:return {'multiplier':0,'status':'DIFFICULTY_NOT_ATTAINED_NO_ADDITIONAL_LEVEL'}
    k,c=min(eligible,key=lambda z:(abs(math.log(z[1]['grid_N80']/1024)),float(z[0])))
    return {'multiplier':float(k),'status':'PILOT_PROXY_ONLY','pilot_grid_N80':c['grid_N80']}


def curve(values,grid,conditional=False):
    v=np.asarray(values,bool);p=v.mean(0);out=dict(counts=v.sum(0).tolist(),denominator=len(v),fraction=p.tolist(),**grid_n80(p.tolist(),grid))
    if conditional:
        from scripts.core.anytime.extension_v02 import wilson
        out['pointwise_Wilson95']=[wilson(int(k),len(v)) for k in v.sum(0)]
    else:out['interval']='NOT_ASSIGNED_DEPENDENT_SEGMENTS'
    return out


def welch_accumulate(stats,x,Y):
    """Row-batch sufficient statistics; labels R x rows, x rows x samples."""
    n1=Y.sum(1);s1=Y@x;ss1=Y@(x*x)
    for key,v in [('n1',n1),('s1',s1),('ss1',ss1),('sx',x.sum(0)),('ssx',(x*x).sum(0))]:
        stats[key]=stats.get(key,0)+v
    stats['n']=stats.get('n',0)+len(x)
    return stats


def welch_value(s):
    n1=s['n1'][:,None];n0=s['n']-n1
    with np.errstate(divide='ignore',invalid='ignore'):
        a=s['s1']/n1;b=(s['sx']-s['s1'])/n0
        va=np.maximum(0,(s['ss1']-s['s1']**2/n1)/(n1-1));vb=np.maximum(0,(s['ssx']-s['ss1']-(s['sx']-s['s1'])**2/n0)/(n0-1))
        den=np.sqrt(va/n1+vb/n0);v=(a-b)/den
    valid=(n1>1)&(n0>1)&(den>0)&np.isfinite(v)
    return np.where(valid,v,0.),valid


def sample_e(Y,x,alpha,total_samples,budget):
    """Synthetic resource kernel: frozen q=.5+.1*x clipped, predictable plugin.

    Secondary production recipe uses the separately frozen univariate Ridge
    scores. This kernel measures the same state/update dimensions, not fitting.
    """
    R=len(Y);d=x.shape[1];S=np.zeros((R,d));Q=np.ones((R,d));W=np.zeros((R,d));hit=np.zeros(R,bool)
    for k in range(len(x)//2):
        if k%32==0:budget.check()
        difference=np.clip(.5+.1*x[2*k],.005,.995)-np.clip(.5+.1*x[2*k+1],.005,.995)
        D=difference[None,:]*(Y[:,2*k]-Y[:,2*k+1])[:,None]
        W+=np.log1p(np.clip(S/Q,0,.5)*D);S+=D;Q+=D*D
        hit|=(W>=math.log(total_samples/alpha)).any(1)
    return hit


def develop(provider,job,cfg,budget):
    key,target=job['capture'],job['target'];ids=[(0,i) for i in range(cfg['fit_rows'])]
    tune=[(0,i) for i in range(cfg['fit_rows'],cfg['rows_per_chunk'])]
    h=provider.metadata(key,target,ids);th=threshold(h)
    if th['status']!='OK':return dict(th,models=None)
    y=(h>=th['median']).astype(float);yt=(provider.metadata(key,target,tune)>=th['median']).astype(float)
    d=provider.inventory['captures'][key]['samples']
    try:sc=scan(provider,key,ids,y,d,budget,cfg['tile'])
    except ValueError as e:
        if str(e)!='NO_NONCONSTANT_ASSOCIATION_FOR_POI':raise
        return dict(th, status='NO_NONCONSTANT_ASSOCIATION', models=None)
    secondary=None
    if target=='mult_a' and cfg['secondary']['status']=='ACTIVE':
        secondary=dict(mean=sc['mean'],scale=[s if s>0 else 1. for s in sc['sd']],coef=sc['univariate_coef'],
                       intercept=th['pi'],recipe='per-sample standardized Ridge alpha1; clip(.005,.995)')
    lo,hi=window(sc['poi'],d);x=provider.traces(key,ids,lo,hi);t=provider.traces(key,tune,lo,hi)
    seed=int(rng(cfg['seed'],'training',job['id']).integers(0,2**31-100))
    return dict(th,secondary_witness=secondary,poi=sc['poi'],window=[lo,hi],development_abs_r=sc['abs_r'],models=fit_models(x,y,t,yt,seed,cfg,budget),fit_row_count=len(ids),tune_row_count=len(tune),training_seed=seed,
                selected_window_input_sha256=hashlib.sha256(x.tobytes()+t.tobytes()).hexdigest())


def pilot(provider,job,dev,cfg,budget):
    r=job['replicate'];n=cfg['segment_rows'];start=(r%cfg['pilot_background_segments'])*n;ids=[(1,i) for i in range(start,start+n)]
    x=provider.traces(job['capture'],ids,*dev['window']);y=(provider.metadata(job['capture'],job['target'],ids)>=dev['median']).astype(np.int8)
    noise=rng(cfg['seed'],'noise',NAMESPACE['pilot'],job['target_id'],job['level_id'],r).normal(size=x.shape)*dev['models']['ridge']['waveform_sd']*job['level']
    p=probability(dev['models']['ridge'],x+noise);d=swap(p,y)
    return {'fixed_p':fixed_p(d,cfg['pilot_grid'],rng(cfg['seed'],'randomization',NAMESPACE['pilot'],job['target_id'],job['level_id'],r),cfg['randomizations'],budget),'background_segment':r%cfg['pilot_background_segments']}


def evaluate(provider,job,dev,cfg,budget,is_null):
    ids=rows_from_ranges(job['ranges']);x=provider.traces(job['capture'],ids,*dev['window'])
    if job.get('level',0):x=x+rng(cfg['seed'],'noise',NAMESPACE['background'],job['target_id'],job['background_id']).normal(size=x.shape)*dev['models']['ridge']['waveform_sd']*job['level']
    y=None if is_null else (provider.metadata(job['capture'],job['target'],ids)>=dev['median']).astype(np.int8)
    Y=np.stack([labels(cfg['seed'],job['background_id'],job['target_id'],r,len(ids),dev['pi']) for r in range(job['rep_start'],job['rep_stop'])]) if is_null else y[None,:]
    out={}
    for name,m in dev['models'].items():
        p=probability(m,x);paths=epaths(p,Y,cfg['grid'],cfg['alpha'],dev['pi'] if is_null else None,budget)
        if not is_null:paths['fixed_p']=fixed_p(swap(p,y),cfg['grid'],rng(cfg['seed'],'randomization',NAMESPACE['background'],job['target_id'],job['background_id'],0 if job.get('level',0)==0 else 1),cfg['randomizations'],budget)
        else:
            paths={s:{a:{e:{'counts':np.asarray(v,bool).sum(0).tolist(),'replicates':len(Y)} for e,v in ev.items()} for a,ev in aa.items()} for s,aa in paths.items()}
        # Save only aggregate grid events; no row labels, predictions or waveforms.
        out[name]=paths
    return out


def get_jobs(stage,cfg,ledger,output):
    if stage=='secondary':return secondary_jobs(cfg,output)
    targets=[dict(id=i,capture=f'{impl}_variable',target=target) for i,(impl,target) in enumerate(( (im,t) for im in ['ref','pqm4'] for t in ['mult_a','mult_b']))]
    if stage=='development':return targets
    dev=[read(output/'development'/f'{i:05d}.json')['result'] for i in range(4)]
    if stage=='pilot':return [dict(capture=t['capture'],target=t['target'],target_id=t['id'],level=level,level_id=l,replicate=r) for t in targets if dev[t['id']]['status']=='OK' for l,level in enumerate(cfg['noise_levels']) for r in range(cfg['pilot_repetitions'])]
    frozen=read(output/'MODEL_CONDITION_FREEZE.json');jobs=[]
    for bg in ledger['backgrounds']:
        for t in targets:
            if not t['capture'].startswith(bg['implementation']+'_') or dev[t['id']]['status']!='OK':continue
            if stage=='natural' and bg['capture']!='variable_key':continue
            for seg in bg['segments']:
                common=dict(capture=bg['key'],target=t['target'],target_id=t['id'],background_id=seg['id'],ranges=seg['ranges'])
                if stage=='null':
                    jobs.extend(dict(common,rep_start=r,rep_stop=min(r+cfg['null_batch'],cfg['null_repetitions'])) for r in range(0,cfg['null_repetitions'],cfg['null_batch']))
                else:
                    levels=sorted(set([0,frozen['conditions'][str(t['id'])]['multiplier']]))
                    jobs.extend(dict(common,level=l) for l in levels)
    return jobs


def checkpoint_stage(output,stage,jobs,identity,calculate,budget,maximum):
    directory=output/stage;directory.mkdir(exist_ok=True);cp=output/(stage.upper()+'_CHECKPOINT.json')
    state=read(cp) if cp.exists() else {'identity':identity,'jobs_sha256':canon(jobs),'completed':[],'status':'RUNNING','expected':len(jobs)}
    if state['identity']!=identity or state['jobs_sha256']!=canon(jobs):raise ValueError('checkpoint binding')
    for i,h in enumerate(state['completed']):
        if sha(directory/f'{i:05d}.json')!=h:raise ValueError('shard changed')
    count=0
    for i in range(len(state['completed']),len(jobs)):
        budget.check();p=directory/f'{i:05d}.json'
        if p.exists():
            r=read(p)
            if r['identity']!=identity or r['job']!=jobs[i]:raise ValueError('orphan mismatch')
        else:new(p,dict(identity=identity,job=jobs[i],result=calculate(jobs[i])))
        state['completed'].append(sha(p));state['status']='COMPLETE' if len(state['completed'])==len(jobs) else 'RUNNING';atomic_json(cp,state)
        count+=1
        if count>=maximum:break
    if not jobs:state['status']='COMPLETE';atomic_json(cp,state)
    return state


def require_complete(output,stage):
    c=read(output/(stage.upper()+'_CHECKPOINT.json'))
    if c['status']!='COMPLETE':raise ValueError(stage+' incomplete')
    for i,h in enumerate(c['completed']):
        if sha(output/stage/f'{i:05d}.json')!=h:raise ValueError('stage artifact changed')
    return c


def freeze_models(output,cfg,identity):
    dc=require_complete(output,'development');pc=require_complete(output,'pilot');conditions={}
    for tid in range(4):
        d=read(output/'development'/f'{tid:05d}.json')['result']
        if d['status']!='OK':conditions[str(tid)]={'status':d['status'],'multiplier':0};continue
        grouped={str(l):[] for l in cfg['noise_levels']}
        for p in sorted((output/'pilot').glob('*.json')):
            r=read(p)
            if r['job']['target_id']==tid:grouped[str(r['job']['level'])].append([p<=.05 for p in r['result']['fixed_p']])
        curves={k:curve(v,cfg['pilot_grid']) for k,v in grouped.items()};conditions[str(tid)]=dict(select_noise(curves),curves=curves)
    path=output/'MODEL_CONDITION_FREEZE.json'
    if path.exists():
        old=read(path)
        if old['identity']!=identity or old['development']!=dc['completed'] or old['pilot']!=pc['completed'] or old['conditions']!=conditions:raise ValueError('existing model freeze differs')
    else:new(path,dict(UTC=utc(),identity=identity,development=dc['completed'],pilot=pc['completed'],conditions=conditions,natural_or_null_started=False))
    seal=output/'MODEL_CONDITION_SEAL.json'
    if not seal.exists():new(seal,{'sha256':sha(path),'identity':identity})
    verify_model_freeze(output,identity)


def verify_model_freeze(output,identity):
    seal=read(output/'MODEL_CONDITION_SEAL.json');path=output/'MODEL_CONDITION_FREEZE.json';f=read(path)
    if seal['identity']!=identity or seal['sha256']!=sha(path) or f['identity']!=identity:raise ValueError('model seal mismatch')
    if f['development']!=require_complete(output,'development')['completed'] or f['pilot']!=require_complete(output,'pilot')['completed']:raise ValueError('frozen model/condition inputs changed')
    return f


def report(output,cfg):
    require_complete(output,'null');require_complete(output,'natural');null={};natural={}
    for stage,groups in [('null',null),('natural',natural)]:
        for p in sorted((output/stage).glob('*.json')):
            r=read(p);j=r['job']
            for model,methods in r['result'].items():
                for s,a in methods.items():
                    if s=='fixed_p':
                        for alpha in cfg['alpha']:
                            k=json.dumps([j['capture'],j['target'],model,j['level'],'fixed_N',alpha]);groups.setdefault(k,[]).append([v<=alpha for v in a])
                        continue
                    for alpha,events in a.items():
                        for e,vs in events.items():
                            if e=='first_crossing_rows':continue
                            key=[j['capture'],j['target'],model,j['background_id'] if stage=='null' else j['level'],s,float(alpha),e]
                            if stage=='null':
                                slot=groups.setdefault(json.dumps(key),{'counts':np.zeros(len(cfg['grid']),int),'denominator':0});slot['counts']+=vs['counts'];slot['denominator']+=vs['replicates']
                            else:groups.setdefault(json.dumps(key),[]).extend(vs)
    from scripts.core.anytime.extension_v02 import wilson
    nc={}
    for k,v in null.items():
        if v['denominator']!=cfg['null_repetitions']:raise ValueError('incomplete conditional repetitions')
        p=(v['counts']/v['denominator']).tolist();nc[k]={'counts':v['counts'].tolist(),'denominator':v['denominator'],'fraction':p,'pointwise_Wilson95':[wilson(int(c),v['denominator']) for c in v['counts']],**grid_n80(p,cfg['grid'])}
    result={'scope':'REGISTERED_REAL_BACKGROUND_ONLY','null':nc,'natural':{k:curve(v,cfg['grid']) for k,v in natural.items()},'claims':'Null conditional on fixed background/frozen model; natural descriptive, no population power or independent-capture intervals.'}
    new(output/'RESULTS.json',result)
    import csv
    with (output/'CURVES.csv').open('x') as f:
        w=csv.writer(f);w.writerow(['scope','condition','N_rows','count','denominator','fraction','lower_pointwise_MC','upper_pointwise_MC','grid_N80','status'])
        for scope in ['null','natural']:
            for k,c in result[scope].items():
                for i,n in enumerate(cfg['grid']):
                    ci=c.get('pointwise_Wilson95',[[None,None]]*len(cfg['grid']))[i]
                    w.writerow([scope,k,n,c['counts'][i],c['denominator'],c['fraction'][i],*ci,c['grid_N80'],c['status']])
    lines=['# Registered real-background primary results','',result['claims'],'','| Natural condition | Grid N80 | Status |','|---|---:|---|']
    for k,c in result['natural'].items():lines.append(f"| {k} | {c['grid_N80']} | {c['status']} |")
    lines+=['','No Wilson intervals across physical segments. No natural LR or prefix prediction. Conditional null intervals are per fixed background and model. See CURVES.csv and checkpoints for complete records.']
    with (output/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    # Compact SVG small multiples; every natural condition, no best-case selection.
    curves=list(result['natural'].items());height=150*len(curves)+40
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="820" height="{height}">','<rect width="100%" height="100%" fill="white"/>']
    import html
    for j,(k,c) in enumerate(curves):
        top=30+j*150;pts=' '.join(f'{30+750*i/max(1,len(cfg["grid"])-1):.1f},{top+100*(1-p):.1f}' for i,p in enumerate(c['fraction']))
        svg.extend([f'<text x="15" y="{top-6}" font-size="10">{html.escape(k)}</text>',f'<polyline points="{pts}" fill="none" stroke="#1864ab"/>',f'<text x="20" y="{top+120}" font-size="10">rows {cfg["grid"][0]} to {cfg["grid"][-1]}, equally spaced grid indices; descriptive fractions 0–1</text>'])
    with (output/'NATURAL_CURVES.svg').open('x') as f:f.write('\n'.join(svg+['</svg>']))

