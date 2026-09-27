"""Portable orchestration of extracted historical numerical routines.

Fresh reruns, not the historical producer. Never executed on real data during export.
Run as python -m scripts.run_experiments_derived. Outputs must be outside this artifact.
"""
import argparse,json,time,resource,datetime,importlib.metadata
from pathlib import Path
from threadpoolctl import threadpool_limits
from scripts.core import real as r,amendment as am,synthetic as sy,synthetic_base as sb,followup as fu
from scripts.core.anytime.common import atomic_json,sha
ROOT=Path(__file__).resolve().parents[1]
def require_recorded_environment():
    for line in (ROOT/'requirements.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        package,version=line.split('==')
        if importlib.metadata.version(package)!=version:
            raise RuntimeError('Recorded package version required for fresh rerun: '+line)
def binding():
    return {str(p.relative_to(ROOT)):sha(p) for sub in ['scripts','configs'] for p in sorted((ROOT/sub).rglob('*')) if p.is_file() and p.suffix in ['.py','.json']}
def init(output,workflow,dataset):
    output.mkdir(parents=True,exist_ok=False)
    ident={'scope':'NEW_PORTABLE_RERUN_NOT_HISTORICAL_FREEZE','workflow':workflow,'code_config':binding(),'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if workflow=='real':
        if dataset is None:raise ValueError('--dataset-root required')
        schema=r.read(ROOT/'configs/input_schema.json');inv={'captures':{}}
        for key,v in schema.items():
            files={}
            for family,chunks in v['files'].items():
                files[family]={}
                for chunk,item in chunks.items():
                    if key.endswith('_variable') and int(chunk)>=10:continue
                    p=dataset/item['relative_path'];h=r.header(p)
                    if h['shape']!=item['shape'] or h['dtype']!=item['dtype']:raise ValueError('dataset schema differs')
                    files[family][chunk]={'relative_path':item['relative_path'],'header':h}
            inv['captures'][key]={'implementation':v['implementation'],'samples':v['samples'],'files':files}
        r.new(output/'INPUT_IDENTITIES.json',inv);ident['input_identities_sha256']=sha(output/'INPUT_IDENTITIES.json');ident['dataset_root']=str(dataset.resolve())
    r.new(output/'RERUN_FREEZE.json',ident)
    r.new(output/'COMPUTE_ACCOUNTING.json',{'cpu_seconds':0.,'GPU_seconds':0,'cap_CPU_seconds':14400,'scope':'Current process CPU plus conservative import allowance per launch; no hidden reruns'})
def synthetic(output,stage,cfg,budget,maximum,identity):
    wp=output/'WITNESSES.json'
    if stage=='training':
        r.new(wp,{case:sb.fit_witness(case,cfg,cfg['seed']+100+i) for i,case in enumerate(cfg['scenarios'])});return
    witnesses=r.read(wp);selection={} if stage=='pilot' else r.read(output/'INTENSITY_SELECTION.json')['selection']
    if stage=='main':
        f=r.read(output/'PREDICTION_FREEZE.json')
        for name,h in f.items():
            if sha(output/name)!=h:raise ValueError('prediction freeze changed')
    jobs=sy.jobs(stage,cfg,selection)
    def calc(j):return sy.simulation(j['case'],j['amplitude'],j['seed'],witnesses.get(j['case'],witnesses['linear']),cfg,cfg['pilot_grid'] if stage=='pilot' else cfg['main_grid'],stage)
    state=r.checkpoint_stage(output,stage,jobs,identity,calc,budget,maximum)
    if state['status']!='COMPLETE':return
    records=[dict(r.read(output/stage/f'{i:05d}.json')['job'],result=r.read(output/stage/f'{i:05d}.json')['result']) for i in range(len(jobs))]
    if stage=='pilot':
        selected={};curves={}
        for case in cfg['scenarios']:
            rows=[]
            for amp in cfg['candidate_amplitudes']:
                rr=[x for x in records if x['case']==case and x['amplitude']==amp]
                rows.append(dict(amplitude=amp,**sy.curve([[p<=.05 for p in x['result']['fixed_p']['raw']] for x in rr],cfg['pilot_grid'])))
            curves[case]=rows;selected[case]=sb.select_amplitude(rows,cfg)
        dest=output/'INTENSITY_SELECTION.json'
        if not dest.exists():r.new(dest,{'selection':selected,'pilot_curves':curves})
    elif stage=='prediction':
        if not (output/'TERMINAL_PREDICTIONS.json').exists():r.new(output/'TERMINAL_PREDICTIONS.json',sy.summarize_predictions(records,cfg))
        if not (output/'PREDICTION_FREEZE.json').exists():r.new(output/'PREDICTION_FREEZE.json',{name:sha(output/name) for name in ['WITNESSES.json','INTENSITY_SELECTION.json','TERMINAL_PREDICTIONS.json','RERUN_FREEZE.json']})
    elif stage=='main':
        if not (output/'MAIN_RESULTS.json').exists():
            result=sy.summarize_main(records,cfg);r.new(output/'MAIN_RESULTS.json',{'curves':result,'main_grid':cfg['main_grid']});r.new(output/'PREDICTION_COMPARISON.json',sy.comparisons(r.read(output/'TERMINAL_PREDICTIONS.json'),result,cfg))
def real(output,stage,cfg,budget,maximum,identity,freeze,dataset):
    if str(dataset.resolve())!=freeze['dataset_root']:raise ValueError('dataset-root changed')
    if sha(output/'INPUT_IDENTITIES.json')!=freeze['input_identities_sha256']:raise ValueError('input identities changed')
    ledger=r.read(ROOT/'configs/rows.json');r.check_ledger(ledger)
    provider=r.Provider(dataset,r.read(output/'INPUT_IDENTITIES.json'),{'approved':True,'scope':'REGISTERED_STAGED_EXECUTION','candidate_sha256':identity},identity)
    if stage=='pilot':r.require_complete(output,'development')
    if stage in ['secondary','report','followup']:r.require_complete(output,'natural')
    if stage=='followup':r.require_complete(output,'secondary')
    if stage=='freeze':r.freeze_models(output,cfg,identity);return
    if stage=='gate':am.ensure_gate(output,r.read(ROOT/'configs/integrity_gate.json'));return
    if stage in ['null','natural','secondary','report','followup']:r.verify_model_freeze(output,identity)
    if stage in ['natural','secondary','report','followup']:
        if r.read(output/'INTEGRITY_GATE.json')['status']!='PASS':raise ValueError('integrity gate not PASS')
    if stage=='report':
        if not (output/'RESULTS.json').exists():r.report(output,cfg)
        if not (output/'SECONDARY_RESULTS.json').exists():r.new(output/'SECONDARY_RESULTS.json',am.secondary_report(output,cfg))
        if not (output/'SAVINGS_DESCRIPTIVE.json').exists():r.new(output/'SAVINGS_DESCRIPTIVE.json',am.natural_savings(output))
        return
    if stage=='followup':
        jobs=[b for b in cfg['secondary']['backgrounds'] if b['background_id'] in [24,25,86,87]]
        def calc(j):
            dev=r.read(output/'development'/('00000.json' if j['implementation']=='ref' else '00002.json'))['result'];ids=r.rows_from_ranges(j['ranges']);y=fu.natural_label(provider.metadata(j['capture'],'mult_a',ids),dev);parts=[]
            for lo in range(0,j['samples'],1024):
                tile=output/'followup_tiles'/str(j['background_id'])/f'{lo:05d}.json'
                if tile.exists():
                    saved=r.read(tile)
                    if saved['identity']!=identity:raise ValueError('followup tile identity changed')
                    parts.append(saved['result']);continue
                budget.check();hi=min(lo+1024,j['samples']);result=fu.kernel(provider.traces(j['capture'],ids,lo,hi),y,dev['secondary_witness'],dev['pi'],lo,j['samples'],budget)
                tile.parent.mkdir(parents=True,exist_ok=True);r.new(tile,{'identity':identity,'result':result});parts.append(result)
            return {'background':j,'threshold':dev['median'],'comparators':fu.merge(parts)}
    else:
        jobs=r.get_jobs(stage,cfg,ledger,output)
        def calc(j):
            if stage=='development':return r.develop(provider,j,cfg,budget)
            dev=r.read(output/'development'/f"{j['target_id']:05d}.json")['result']
            if stage=='pilot':return r.pilot(provider,j,dev,cfg,budget)
            if stage=='secondary':return am.secondary_job(provider,j,dev,cfg,budget)
            return r.evaluate(provider,j,dev,cfg,budget,stage=='null')
    state=r.checkpoint_stage(output,stage,jobs,identity,calc,budget,maximum)
    if stage=='followup' and state['status']=='COMPLETE' and not (output/'FOLLOWUP_RESULTS.json').exists():
        r.new(output/'FOLLOWUP_RESULTS.json',{'status':'COMPLETE','scope':'NEW_DERIVED_RERUN_FOUR_DESCRIPTIVE_BACKGROUNDS','backgrounds':[r.read(output/stage/f'{i:05d}.json')['result'] for i in range(len(jobs))],'rerun_identity':identity})
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workflow',choices=['synthetic','real'],required=True);p.add_argument('--stage',choices=['init','training','development','pilot','prediction','main','freeze','null','gate','natural','secondary','followup','report'],required=True);p.add_argument('--dataset-root',type=Path);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--allow-real-data',action='store_true');p.add_argument('--max-records',type=int,default=1);a=p.parse_args()
    out=a.output_root.resolve()
    allowed={'synthetic':{'init','training','pilot','prediction','main'},'real':{'init','development','pilot','freeze','null','gate','natural','secondary','report','followup'}}
    if a.stage not in allowed[a.workflow]:raise ValueError('stage does not belong to workflow')
    if out==ROOT or ROOT in out.parents:raise ValueError('rerun state/learned weights must be outside distributable artifact')
    if a.workflow=='real' and not a.allow_real_data:raise PermissionError('explicit --allow-real-data required even for input initialization')
    if a.workflow=='real' and a.dataset_root is None:raise ValueError('--dataset-root required')
    if a.max_records<1:raise ValueError('positive record bound required')
    require_recorded_environment()
    if a.stage=='init':init(out,a.workflow,a.dataset_root);return
    freeze=r.read(out/'RERUN_FREEZE.json')
    if freeze['workflow']!=a.workflow or freeze['code_config']!=binding():raise ValueError('rerun code/config identity changed')
    cfg=r.read(ROOT/'configs'/f'{a.workflow}.json');identity=r.canon(freeze);account=r.read(out/'COMPUTE_ACCOUNTING.json');remaining=account['cap_CPU_seconds']-account['cpu_seconds']
    if a.stage=='followup':remaining=min(remaining,2000-account.get('stage_CPU_seconds',{}).get('followup',0))
    if remaining<30:r.new(out/'INCOMPLETE.json',{'reason':'CPU cap'});return
    budget=r.Budget(wall=25,cpu=remaining-2,memory=1536);started=time.process_time();wall=time.monotonic()
    try:
        with threadpool_limits(limits=1):
            if a.workflow=='synthetic':synthetic(out,a.stage,cfg,budget,a.max_records,identity)
            else:real(out,a.stage,cfg,budget,a.max_records,identity,freeze,a.dataset_root)
    except r.Incomplete as e:
        atomic_json(out/'INCOMPLETE.json',{'reason':str(e),'stage':a.stage});raise
    finally:
        charged=time.process_time()-started+2;account['cpu_seconds']+=charged;account.setdefault('stage_CPU_seconds',{}).setdefault(a.stage,0);account['stage_CPU_seconds'][a.stage]+=charged;account['peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;atomic_json(out/'COMPUTE_ACCOUNTING.json',account)
        log=out/'LAUNCH_ATTEMPTS.json';history=r.read(log) if log.exists() else [];history.append({'stage':a.stage,'wall_seconds':time.monotonic()-wall,'CPU_seconds_with_import_allowance':time.process_time()-started+2});atomic_json(log,history)
if __name__=='__main__':main()
