from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--screen',action='append',default=[]); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    screens=[]
    for spec in a.screen:
        name,path=spec.split('=',1); d=json.loads(Path(path).read_text()); screens.append({'name':name,'path':path,**d})
    eligible=[s for s in screens if s.get('valid')]
    promoted=[s for s in eligible if s.get('full_promotion')]
    def rank(s):
        d=s.get('deltas',{}); return (d.get('teacher') or -9,d.get('pairfull') or -9,s.get('pair_full_advantage_over_local') or -9,s.get('residual_intervention_net') or -9)
    winner=max(promoted,key=rank)['name'] if promoted else None
    mechanism=max(eligible,key=rank)['name'] if eligible else None
    report={
        'audit':'v64_3_7_darm_dbr_screen_comparison','winner':winner,'best_mechanism_arm':mechanism,
        'run_full_pipeline':bool(winner),'screens':screens,
        'note':'V64.3.7 removes the ruled-out BCHA/CCBR acquisition branches from optimization. DARM must first restore the strong direct selected-local anchor; DBR is promoted only by pair-full and final teacher-action gains without harmful residual interventions.'
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8'); print(json.dumps(report,indent=2,sort_keys=True)); return 0

if __name__=='__main__': raise SystemExit(main())
