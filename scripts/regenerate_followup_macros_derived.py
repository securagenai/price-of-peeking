#!/usr/bin/env python3
"""Add hash-bound follow-up macros using the v2 provenance convention; never rewrite v2."""
import argparse,json
from pathlib import Path
from regenerate_summary_macros_derived import sha
EXPECTED='982ce465244514373da286e059693e343933fa2cae5c7d496a001edd3275ee57'
def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('.'));a=p.parse_args()
    assert sha(a.results)==EXPECTED,'follow-up result identity mismatch'
    d=json.loads(a.results.read_text());assert d['status']=='COMPLETE';bs=d['backgrounds'];assert len(bs)==4
    macros={};prov={};identity=dict(file='results/followup.json',sha256=EXPECTED)
    def put(k,v,derivation,fmt='{}'):
        macros[k]=fmt.format(v);prov[k]=dict(value=v,formatted=macros[k],source=identity,derivation=derivation)
        return '\\'+k+'{}'
    eb=[b['comparators']['e_0.05'] for b in bs];welch=[b['comparators']['welch_any'] for b in bs]
    put('FollowBackgrounds',len(bs),'count registered backgrounds')
    for tag,vals in [('EB',eb),('Welch',welch)]:
        hit=[x['first_row'] for x in vals if x['rejected']];put('Follow'+tag+'Detected',len(hit),'count rejected '+tag)
        if hit:
            put('Follow'+tag+'RowMin',min(hit),'min first_row conditional on rejection '+tag)
            put('Follow'+tag+'RowMax',max(hit),'max first_row conditional on rejection '+tag)
    gaps=[x['terminal_gap_nats'] for x in eb if not x['rejected']]
    if gaps:
        put('FollowEBGapMin',min(gaps),'min terminal gap conditional on no crossing','{:.2f}')
        put('FollowEBGapMax',max(gaps),'max terminal gap conditional on no crossing','{:.2f}')
    rows=[];labels={'welch_any':'Welch any-look','welch_terminal':'Welch terminal','e_0.05':r'e-Bonferroni ($\alpha=.05$)','e_0.01':r'e-Bonferroni ($\alpha=.01$)'}
    for bi,b in enumerate(bs):
        bg=put('FollowBg'+chr(65+bi),b['background']['background_id'],'background ID')
        for ci,key in enumerate(labels):
            x=b['comparators'][key];tag=chr(65+bi)+chr(65+ci)
            first=put('FollowFirst'+tag,x['first_row'],'background '+str(bi)+' '+key+' first_row') if x['first_row'] is not None else '--'
            count=put('FollowCount'+tag,x['terminal_samples'],'background '+str(bi)+' '+key+' terminal_samples')
            rows.append(' & '.join([bg,labels[key],'yes' if x['rejected'] else 'no',first,count])+r'\\')
    table=r'''\begin{table}[tbp]\centering\small
\caption{Registered natural-label full-window follow-up. Background IDs identify previously exposed variable-key segments. First row is a stopping row count; sample counts are terminal exceedances, not cumulative. Descriptive results only.}\label{tab:followup}
\begin{tabular}{llllr}\toprule
Background & Comparator & Rejected & First row & Samples\\\midrule
'''+ '\n'.join(rows)+r'\bottomrule\end{tabular}\end{table}'+'\n'
    for sub in ['generated','provenance','tables']:(a.out/sub).mkdir(exist_ok=True)
    (a.out/'generated/macros_v3.tex').write_text('% Generated only from hash-bound follow-up aggregates.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    (a.out/'tables/followup.tex').write_text(table)
    (a.out/'provenance/MACRO_V3_PROVENANCE.json').write_text(json.dumps(prov,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
