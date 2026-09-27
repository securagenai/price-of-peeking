"""Generate every result number/table/vector figure from hash-bound stored JSON.

No dataset loader, numpy.load, waveform plotting or experiment entrypoint exists.
Run from the project root. Existing input identity is never silently replaced.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR',os.path.join(__import__('tempfile').gettempdir(),'peeking-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from statistics import NormalDist

import argparse
ap=argparse.ArgumentParser();ap.add_argument('--artifact-root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output-root',type=Path,required=True);args=ap.parse_args()
ROOT=args.artifact_root.resolve();sys.path.insert(0,str(ROOT))
from scripts.core.legacy import read,sha
P=args.output_root.resolve();G=P/'generated';T=P/'tables';F=P/'figures'
FILES={'real': 'results/real.json', 'cfg': 'results/cfg.json', 'inventory': 'results/inventory.json', 'ledger': 'results/ledger.json', 'gate': 'results/gate.json', 'secondary': 'results/secondary.json', 'savings': 'results/savings.json', 'matched': 'results/matched.json', 'compute': 'results/compute.json', 'launches': 'results/launches.json', 'synthetic': 'results/synthetic.json', 'syn_config': 'results/syn_config.json', 'syn_prediction': 'results/syn_prediction.json', 'syn_moments': 'results/syn_moments.json', 'syn_flags': 'results/syn_flags.json', 'syn_savings': 'results/syn_savings.json', 'syn_compute': 'results/syn_compute.json', 'writing_gate': 'results/writing_gate.json', 'model_freeze': 'results/model_freeze.json'}

def write(p,s):p.write_text(s,encoding='utf-8')
def dump(p,x):write(p,json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def letters(i):
    s=''
    while True:
        s=chr(65+i%26)+s;i=i//26-1
        if i<0:return s

class Numbers:
    def __init__(self,hashes):self.entries={};self.hashes=hashes
    def put(self,value,source,field,fmt=None,name=None,derivation=None):
        name=name or 'Result'+letters(len(self.entries))
        text='--' if value is None else format(value,fmt) if fmt else str(value)
        self.entries[name]=dict(value=value,formatted=text,source_file=FILES[source],field=field,
            source_sha256=self.hashes[FILES[source]],derivation=derivation)
        return '\\'+name
    def save(self):
        write(G/'macros.tex','% Generated; edit inputs/code, never result values.\n'+''.join('\\newcommand{\\'+k+'}{'+v['formatted']+'}\n' for k,v in self.entries.items()))
        dump(G/'MACRO_PROVENANCE.json',self.entries)

def table(name,headers,rows,caption,long=False):
    cols='l'*len(headers)
    if long:
        s='\\begingroup\\scriptsize\n\\begin{longtable}{'+cols+'}\n\\caption{'+caption+'}\\label{tab:'+name+'}\\\\\n\\toprule\n'+' & '.join(headers)+'\\\\\n\\midrule\\endfirsthead\n\\toprule\n'+' & '.join(headers)+'\\\\\n\\midrule\\endhead\n'
        s+='\n'.join(' & '.join(r)+'\\\\' for r in rows)+'\n\\bottomrule\n\\end{longtable}\n\\endgroup\n'
    else:
        s='\\begin{table}[tbp]\n\\centering\\small\n\\caption{'+caption+'}\\label{tab:'+name+'}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{'+cols+'}\n\\toprule\n'+' & '.join(headers)+'\\\\\n\\midrule\n'
        s+='\n'.join(' & '.join(r)+'\\\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n'
    write(T/(name+'.tex'),s)

def main():
    expected=json.loads((ROOT/"docs/RESULT_SOURCES.json").read_text())["sources"]
    for k,v in expected.items():
        if sha(ROOT/v["artifact_file"])!=v["artifact_sha256"]:raise ValueError("artifact source mismatch: "+k)
    for d in [G,T,F]:d.mkdir(parents=True,exist_ok=True)
    hashes={path:sha(ROOT/path) for path in FILES.values()}
    freeze=G/'INPUT_IDENTITIES.json'
    if freeze.exists():
        if read(freeze)!=hashes:raise ValueError('paper input hash changed')
    else:dump(freeze,hashes)
    d={k:read(ROOT/v) for k,v in FILES.items()};assert d['writing_gate']['status']=='PASS'
    n=Numbers(hashes);cfg=d['cfg'];real=d['real'];syn=d['synthetic'];grid=cfg['grid'];sg=syn['main_grid']
    # Named numbers used in prose; every source and derivation is explicit.
    n.put(d['compute']['CPU_seconds'],'compute','CPU_seconds','.1f','RealCpu')
    n.put(d['compute']['wall_seconds'],'compute','wall_seconds','.1f','RealWall')
    n.put(max(x['peak_RSS_KiB'] for x in d['launches'])/1024,'launches','[*].peak_RSS_KiB','.1f','PeakMemory','max /1024')
    for name,key in [('FitRows','fit_rows'),('RowsPerChunk','rows_per_chunk'),('SegmentRows','segment_rows'),('NullReps','null_repetitions'),('Randomizations','randomizations'),('PilotReps','pilot_repetitions'),('CpuCap','cpu_cap')]:n.put(cfg[key],'cfg',key,name=name)
    n.put(cfg['rows_per_chunk']-cfg['fit_rows'],'cfg','rows_per_chunk,fit_rows',name='TuneRows',derivation='difference')
    n.put(grid[0],'cfg','grid[0]',name='GridMinimum');n.put(grid[-1],'cfg','grid[-1]',name='GridMaximum')
    n.put(len(d['gate']['families']),'gate','families',name='FamilyCount',derivation='length')
    n.put(sum(f.get('flag_count',0) for f in d['gate']['families']),'gate','families[*].flag_count',name='FlagCount',derivation='sum')
    n.put(len(d['secondary']['backgrounds']),'secondary','backgrounds',name='SecondaryBackgrounds',derivation='length')
    n.put(d['secondary']['backgrounds']['0']['pointwise_Wilson95']['e_Bonferroni_0.05'][1]*100,'secondary','backgrounds.0.pointwise_Wilson95.e_Bonferroni_0.05[1]','.2f','ZeroUpperPercent','times100')
    for name,key in [('SynTraining','training_rows_per_scenario'),('SynPredictionReps','prediction_repetitions'),('SynNullReps','main_null_repetitions_proposed'),('SynPositiveReps','main_positive_repetitions_proposed')]:n.put(d['syn_config'][key],'syn_config',key,name=name)
    n.put(sg[-1],'synthetic','main_grid[-1]',name='SynMaximum')
    n.put(len([x for x in d['syn_prediction'] if x['finite_grid_log_error'] is not None]),'syn_prediction','[*].finite_grid_log_error',name='FinitePredictions',derivation='count non-null')
    n.put(len(d['syn_prediction']),'syn_prediction','root',name='PredictionComparisons',derivation='length')
    za=NormalDist().inv_cdf(.95); zp=NormalDist().inv_cdf(.8)
    n.put(((zp+math.sqrt(zp*zp+2*math.log(20)))/(za+zp))**2,'syn_config','alphas, power target', '.3f','TerminalReference',derivation='theoretical terminal Gaussian formula at alpha=.05,power=.8; NOT_APPLICABLE')
    # Dataset/design and exact allocation summaries.
    rows=[];alloc=[]
    for bg in d['ledger']['backgrounds']:
        key=bg['key'];inv=d['inventory']['captures'][key];src='captures.'+key
        rows.append([esc(key),n.put(len(inv['chunks']),'inventory',src+'.chunks',derivation='length'),n.put(inv['rows_per_chunk'],'inventory',src+'.rows_per_chunk'),n.put(inv['samples'],'inventory',src+'.samples'),'RNG null + natural' if bg['capture']=='variable_key' else 'RNG null only'])
        segs=len(bg['segments']);used=segs*cfg['segment_rows'];left=sum(v['stop']-v['start'] for v in bg['leftover_ranges'])
        alloc.append([esc(key),esc(','.join(map(str,bg['eligible_chunks']))),n.put(segs,'ledger',f'backgrounds[{key}].segments',derivation='length'),n.put(used,'ledger',f'backgrounds[{key}].segments[*].ranges',derivation='sum lengths'),n.put(left,'ledger',f'backgrounds[{key}].leftover_ranges',derivation='sum lengths')])
    table('design',['Capture','Chunks','Rows/chunk','Samples','Role'],rows,'Audited unmasked capture dimensions. Allocation below restricts use; these are not independent devices.')
    table('allocation',['Capture','Eligible chunks','Segments','Used rows','Leftover rows'],alloc,'Registered background allocation. Variable development and pilot chunks are separate. Unused pqm4 variable reserve is excluded.')
    # Main natural and synthetic tables, all additional conditions in appendix.
    allrows=[];mainrows=[]
    for k,v in real['natural'].items():
        a=json.loads(k)
        if a[4]!='fixed_N':continue
        vals=[v];fields=[k]
        for b in ['plugin','ons_gain']:
            for ev in ['crossing','terminal']:
                kk=json.dumps(a[:4]+[b,a[5],ev]);vals.append(real['natural'][kk]);fields.append(kk)
        cells=[n.put(x['grid_N80'],'real','natural.'+f+'.grid_N80')+'{}'+('B' if x['status']=='AT_NMIN_BOUNDARY' else 'C' if x['grid_N80'] is None else '') for x,f in zip(vals,fields)]
        row=[esc(a[0].split('_')[0]+'/'+a[1][-1]),esc(a[2]),n.put(a[3],'real','natural key: '+k,derivation='multiplier from condition key'),n.put(a[5],'real','natural key: '+k,derivation='alpha from condition key')]+cells
        allrows.append(row)
        if a[2]=='ridge' and a[5]==.05:mainrows.append(row[:1]+row[2:3]+cells)
    table('natural',['Target','Waveform SD','Fixed','Plug C','Plug T','ONS C','ONS T'],mainrows,'Natural-label descriptive grid N80, primary Ridge at the primary level. B: lower-grid boundary; --C: right-censored. C/T denote first crossing/terminal.')
    table('natural_all',['Target','Model','SD','alpha','Fixed','Plug C','Plug T','ONS C','ONS T'],allrows,'Every registered natural condition and both alpha levels; same boundary/censoring notation.',True)
    sr=[]
    for case in d['syn_config']['scenarios']:
        for alpha in ['0.05','0.01']:
            c=syn['curves'][case]['raw'][alpha];items=[('fixed_N',c['fixed_N'])]
            for s in ['plugin','ons_gain']:
                for e in ['first_crossing','terminal_exceedance']:items.append((s+'.'+e,c[s][e]))
            cells=[n.put(v['grid_N80'],'synthetic',f'curves.{case}.raw.{alpha}.{key}.grid_N80')+'{}'+('C' if v['grid_N80'] is None else '') for key,v in items]
            sr.append([esc(case),n.put(float(alpha),'syn_config','alpha',derivation='registered alpha')]+cells)
    table('synthetic_n',['Scenario','alpha','Fixed','Plug C','Plug T','ONS C','ONS T'],sr,'Stored synthetic raw-payoff grid N80. Evaluation rows only; add the frozen independent training budget for a training-inclusive per-witness comparison.')
    nr=[]
    for case in d['syn_config']['null_scenarios']:
        for a in ['0.05','0.01']:
            c=syn['curves'][case]['raw'][a];items=[('fixed_N',c['fixed_N'])]+[(s+'.first_crossing',c[s]['first_crossing']) for s in ['plugin','ons_gain']]
            nr.append([esc(case),n.put(float(a),'syn_config','alpha')]+[n.put(v['counts'][-1],'synthetic',f'curves.{case}.raw.{a}.{s}.counts[-1]') for s,v in items])
    table('synthetic_null',['Null background','alpha','Fixed final count','Plugin ever count','ONS ever count'],nr,'Stored raw-payoff synthetic null rejection counts at the maximum horizon. Each cell has the frozen null repetition count; pointwise intervals are plotted/reported in the evidence.')
    gr=[]
    for i,f in enumerate(d['gate']['families']):
        gr.append([f['implementation']+'/'+f['target'][-1],f['model'],esc(f['bettor']),n.put(f['alpha'],'gate',f'families[{i}].alpha'),n.put(f['backgrounds'],'gate',f'families[{i}].backgrounds'),n.put(f.get('flag_count'),'gate',f'families[{i}].flag_count'),n.put(f['quantile_999'],'gate',f'families[{i}].quantile_999'),f['status']])
    table('gate_all',['Target','Model','Bettor','alpha','Backgrounds','Flags','Limit','Status'],gr,'All integrity-gate families. FAIL requires strictly more flags than the limit. This gate is not a simultaneous scientific claim.',True)
    table('gate',['Target','Model','Bettor','alpha','Backgrounds','Flags','Limit','Status'],[r for r,f in zip(gr,d['gate']['families']) if f['target']=='mult_a' and f['model']=='ridge'],'Primary-target/Ridge designed-null gate summary; all families are retained in the appendix.')
    se=[]
    for bg,v in d['secondary']['backgrounds'].items():
        se.append([n.put(int(bg),'secondary','backgrounds key '+bg)]+[n.put(v['counts'][k],'secondary',f'backgrounds.{bg}.counts.{k}') for k in ['peeking','terminal','e_Bonferroni_0.05','e_Bonferroni_0.01']])
    table('secondary',['Background ID','Welch any look','Welch terminal','e-Bonf primary','e-Bonf secondary'],se,'Full-column designed-null counts per registered background. Each denominator is the registered null repetition count; both e-Bonferroni levels are shown.')
    cpu=[]
    for stage in ['development','pilot','freeze','null','gate','natural','secondary','report']:
        rr=[r for r in d['launches'] if r['command'][4]==stage]
        cpu.append([stage,n.put(len(rr),'launches',f'command[4]=={stage}',derivation='count'),n.put(sum(r['CPU_seconds'] for r in rr),'launches',f'command[4]=={stage}; CPU_seconds','.1f',derivation='sum'),n.put(sum(r['wall_seconds'] for r in rr),'launches',f'command[4]=={stage}; wall_seconds','.1f',derivation='sum')])
    table('compute',['Stage','Invocations','CPU seconds','Wall seconds'],cpu,'Actual child CPU including imports plus driver accounting. No GPU. Formatting/build has a separate small allowance and is not included as experimental compute.')
    # Full numeric supplemental files; no per-row values are introduced.
    nullrows=[]
    for k,v in real['null'].items():
        a=json.loads(k)
        if a[-1]!='crossing':continue
        nullrows.append([esc(a[0].split('_')[0]+'/'+a[1][-1]),n.put(a[3],'real','condition '+k,derivation='background ID'),esc(a[2]),esc(a[4]),n.put(a[5],'real','condition '+k,derivation='alpha'),n.put(v['counts'][-1],'real','null.'+k+'.counts[-1]')])
    # Large per-background table is a standalone supplement, not anonymous main-body padding.
    table('null_backgrounds_all',['Target','BG','Model','Bettor','alpha','Count'],nullrows,'Every background/target/model/bettor first-crossing count at the registered horizon. All denominators are the registered R.',True)
    save_rows=[]
    for k,v in d['savings'].items():
        q=v['stopping_rows_conditional_on_finite']
        save_rows.append([esc(k.replace('_variable','').replace('mult_','').replace('ons_gain','ONS').replace('|','/')),
            n.put(v['finite_stops'],'savings',k+'.finite_stops'),n.put(q['median'],'savings',k+'.stopping_rows_conditional_on_finite.median'),n.put(q['IQR'],'savings',k+'.stopping_rows_conditional_on_finite.IQR')])
    table('savings_all',['Condition','Finite stops','Median row','IQR'],save_rows,'All undegraded finite-stop summaries. Full budget-specific fractions and ratios remain in the hash-bound JSON.',True)
    # Vector figures: fixed palette and redundant line/marker coding.
    plt.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'lines.linewidth':1.2,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#000000','#0072B2','#D55E00','#009E73','#CC79A7'];figure_provenance={}
    def savefig(fig,name,sources,description):
        fig.tight_layout();fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
        figure_provenance[name]=dict(sources={FILES[s]:hashes[FILES[s]] for s in sources},description=description,sha256=sha(F/(name+'.pdf')))
    fig,axs=plt.subplots(1,3,figsize=(5.6,2.4),sharey=True)
    for ax,case in zip(axs,d['syn_config']['scenarios']):
        c=syn['curves'][case]['raw']['0.05']
        items=[('Fixed N',c['fixed_N'],'-',colors[0]),('Plugin crossing',c['plugin']['first_crossing'],'-',colors[1]),('Plugin terminal',c['plugin']['terminal_exceedance'],'--',colors[1]),('ONS crossing',c['ons_gain']['first_crossing'],'-.',colors[2]),('ONS terminal',c['ons_gain']['terminal_exceedance'],':',colors[2])]
        for label,v,ls,col in items:ax.plot(sg,v['probability'],ls,color=col,label=label)
        ax.set_xscale('log',base=2);ax.set_ylim(-.02,1.02);ax.set_title(case.replace('_',' '));ax.set_xlabel('Evaluation rows');ax.grid(alpha=.2)
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.13));axs[0].set_ylabel('Detection probability');savefig(fig,'F1_synthetic',['synthetic'],'All frozen positive scenarios, raw payoff, alpha .05, both anytime events')
    fig,ax=plt.subplots(figsize=(5.6,3.0));categories=[];points=[]
    for case in d['syn_config']['scenarios']:
        c=syn['curves'][case]['raw']['0.05'];categories.append('synthetic\n'+case.replace('_',' '));points.append([c[s][e]['grid_N80']/c['fixed_N']['grid_N80'] if c[s][e]['grid_N80'] else np.nan for s in ['plugin','ons_gain'] for e in ['first_crossing','terminal_exceedance']])
    for capture in ['ref_variable','pqm4_variable']:
        for target in ['mult_a','mult_b']:
            rr=[r for r in d['matched']['rows'] if r['capture']==capture and r['target']==target and r['model']=='ridge' and r['multiplier']>0 and r['alpha']==.05]
            categories.append(capture.split('_')[0]+' degraded\n'+target[-1]);points.append([next(r['grid_ratio'] for r in rr if r['bettor']==s and r['event']==e) for s in ['plugin','ons_gain'] for e in ['crossing','terminal']])
    pp=np.array(points,dtype=float)
    for j,(label,marker) in enumerate(zip(['Plugin crossing','Plugin terminal','ONS crossing','ONS terminal'],['o','s','^','D'])):ax.plot(np.arange(len(categories))+(j-1.5)*.1,pp[:,j],marker,ls='none',color=colors[j+1],label=label)
    ax.axhline(1,color='gray',ls=':');ax.set_xticks(range(len(categories)),categories);ax.set_ylabel('Grid N80 ratio (anytime / fixed N)');ax.legend(ncol=2,loc='upper left');ax.grid(axis='y',alpha=.2)
    savefig(fig,'F2_ratios',['synthetic','matched'],'Primary Ridge alpha .05; synthetic raw and real degraded finite interior grid ratios, no universal-cost interpretation')
    fig,axs=plt.subplots(2,2,figsize=(5.6,4.4),sharey=True)
    for ax,bg in zip(axs.flat,d['ledger']['backgrounds']):
        for j,s in enumerate(['plugin','ons_gain','lr']):
            vv=[]
            for seg in bg['segments']:
                k=json.dumps([bg['key'],'mult_a','ridge',seg['id'],s,.05,'crossing']);v=real['null'][k];vv.append((v['fraction'][-1],*v['pointwise_Wilson95'][-1]))
            ar=np.array(vv);ax.errorbar(np.arange(len(vv))+(j-1)*.18,ar[:,0],yerr=[ar[:,0]-ar[:,1],ar[:,2]-ar[:,0]],fmt=['o','s','^'][j],ms=3,color=colors[j+1],label=s)
        ax.axhline(.05,color='black',ls=':');ax.set_title(bg['key'].replace('_',' '));ax.set_xlabel('Registered segment index');ax.grid(alpha=.15)
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.05));axs[0,0].set_ylabel('Rejection frequency');axs[1,0].set_ylabel('Rejection frequency')
    savefig(fig,'F3_nulls',['real','ledger'],'Primary a0/Ridge alpha .05, all105 backgrounds; pointwise conditional MC Wilson95, not simultaneous')
    fig,axs=plt.subplots(4,2,figsize=(5.6,7.2),sharex=True,sharey=True)
    for i,(capture,target) in enumerate(( (c,t) for c in ['ref_variable','pqm4_variable'] for t in ['mult_a','mult_b'])):
        level=max(json.loads(k)[3] for k in real['natural'] if json.loads(k)[:3]==[capture,target,'ridge'])
        for j,mult in enumerate([0,level]):
            ax=axs[i,j];items=[('fixed_N',None,'-',colors[0])]+[(s,e,ls,col) for s,col in [('plugin',colors[1]),('ons_gain',colors[2])] for e,ls in [('crossing','-'),('terminal','--')]]
            for s,e,ls,col in items:
                k=json.dumps([capture,target,'ridge',mult,s,.05]+([e] if e else []));v=real['natural'][k]
                ax.plot(grid,v['fraction'],ls,color=col,label=s+(' '+e if e else ''))
            ax.set_xscale('log',base=2);ax.set_ylim(-.02,1.02);ax.set_title(capture.split('_')[0]+'/'+target[-1]+' '+('raw' if mult==0 else f'{mult:g}x waveform SD'));ax.grid(alpha=.2)
            if j==0:ax.set_ylabel('Segment fraction')
            if i==3:ax.set_xlabel('Evaluation rows')
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.04))
    savefig(fig,'F4_natural',['real','cfg'],'All four natural targets, Ridge, raw and one frozen degradation; no confidence intervals across dependent segments')
    fig,axs=plt.subplots(1,3,figsize=(5.6,2.8));ks=[k for k in d['savings'] if '|ridge|' in k and k.endswith('|0.05')]
    for ax,B in zip(axs,[1024,2048,4096]):
        for i,k in enumerate(ks):
            v=d['savings'][k]['budgets'][str(B)];q=v['stop_over_budget_conditional_on_finite']
            if q['median'] is not None:ax.errorbar(q['median'],i,xerr=[[q['median']-q['q25']],[q['q75']-q['median']]],fmt='o',color=colors[1] if 'plugin' in k else colors[2])
        ax.axvline(1,color='black',ls=':');ax.set_title(f'Budget B = {B}');ax.set_xlabel('Stop row / B');ax.grid(axis='x',alpha=.2)
    axs[0].set_yticks(range(len(ks)),[k.replace('_variable','').replace('mult_','').replace('|ridge','').replace('|0.05','').replace('ons_gain','ONS') for k in ks]);axs[1].set_yticks([]);axs[2].set_yticks([])
    savefig(fig,'F5_savings',['savings'],'Raw Ridge alpha .05, conditional finite-stop median/IQR; full-denominator fractions retained in JSON')
    fig,ax=plt.subplots(figsize=(5.6,3.0));backgrounds=list(d['secondary']['backgrounds']);labels_=dict(peeking='Welch any look',terminal='Welch terminal',**{'e_Bonferroni_0.05':'e-Bonf .05','e_Bonferroni_0.01':'e-Bonf .01'})
    for j,(key,label) in enumerate(labels_.items()):
        vv=[d['secondary']['backgrounds'][b] for b in backgrounds];y=np.array([v['fractions'][key] for v in vv]);ci=np.array([v['pointwise_Wilson95'][key] for v in vv])
        ax.errorbar(np.arange(len(backgrounds))+(j-1.5)*.16,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt=['o','s','^','D'][j],ms=4,color=colors[j],label=label)
    ax.set_xticks(range(len(backgrounds)),backgrounds);ax.set_xlabel('Registered background ID');ax.set_ylabel('False-alarm frequency');ax.legend(ncol=2);ax.grid(axis='y',alpha=.2)
    savefig(fig,'F6_secondary',['secondary'],'Eight prespecified backgrounds, all columns, R256; pointwise Wilson95; 4.5 not nominal alpha')
    fig,ax=plt.subplots(figsize=(3.8,3.2));censored=0
    for r in d['syn_prediction']:
        if r['predicted_grid_N80'] is None or r['observed_grid_N80'] is None:censored+=1;continue
        ax.scatter(r['predicted_grid_N80'],r['observed_grid_N80'],marker='o' if r['payoff']=='raw' else '^',facecolors='none' if r['alpha']==.01 else colors[1],edgecolors=colors[1] if r['strategy']=='plugin' else colors[2],s=45)
    ax.plot([sg[0],sg[-1]],[sg[0],sg[-1]],'k--');ax.set_xscale('log',base=2);ax.set_yscale('log',base=2);ax.set_xlabel('Predicted terminal grid N80');ax.set_ylabel('Observed terminal grid N80');ax.set_title(f'{len(d["syn_prediction"])-censored} finite comparisons; {censored} censored omitted');ax.grid(alpha=.2)
    savefig(fig,'F7_prediction',['syn_prediction','synthetic'],'Stored sixteen-stream terminal Gaussian predictions only; censoring explicitly counted, not plotted as Nmax')
    n.save();dump(G/'FIGURE_PROVENANCE.json',figure_provenance)
    dump(G/'BUILD_ASSETS_STATUS.json',dict(status='PASS',input_files=len(hashes),result_macros=len(n.entries),vector_figures=len(figure_provenance),code_sha256=sha(__file__),waveforms_read=0))
    print(json.dumps(read(G/'BUILD_ASSETS_STATUS.json')))

if __name__=='__main__':main()
