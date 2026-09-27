"""Optional historical v0.2 synthetic reproduction, separately labeled new run."""
import argparse,sys,json
from pathlib import Path
from scripts.core.anytime import extension_v02 as e
from scripts.core.anytime.common import sha,atomic_json
from scripts.run_experiments_derived import binding,require_recorded_environment
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--init',action='store_true');p.add_argument('--max-records',type=int,default=1);a=p.parse_args()
    out=a.output_root.resolve()
    if out==ROOT or ROOT in out.parents:raise ValueError('output must be outside artifact')
    require_recorded_environment()
    cfg=ROOT/'configs/historical_extension.json';ident={'config_sha256':sha(cfg),'code':binding(),'scope':'NEW_DERIVED_RUN_NOT_HISTORICAL_FREEZE'}
    if a.init:
        out.mkdir(parents=True,exist_ok=False);atomic_json(out/'INTERNAL_SYNTHETIC_FREEZE.json',{'identity':ident});return
    e.identity=lambda _:ident
    sys.argv=['extension','--run',str(out),'--config',str(cfg),'--max-records',str(a.max_records)];e.main()
if __name__=='__main__':main()
