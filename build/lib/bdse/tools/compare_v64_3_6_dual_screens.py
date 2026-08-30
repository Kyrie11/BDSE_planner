from __future__ import annotations
import argparse,json
from pathlib import Path

def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--screen',action='append',default=[]); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 screens=[]
 for spec in a.screen:
  name,path=spec.split('=',1); d=json.loads(Path(path).read_text()); d={'name':name,'path':path,**d}; screens.append(d)
 eligible=[s for s in screens if s.get('valid')]
 def rank(s):
  d=s.get('deltas',{}); return (int(s.get('full_promotion',False)),int(s.get('meaningful_acquisition_gain',False))+int(s.get('meaningful_value_gain',False)),d.get('teacher') or -9,d.get('pairfull') or -9,d.get('topm') or -9)
 promoted=[s for s in eligible if s.get('full_promotion')]
 winner=max(promoted,key=rank)['name'] if promoted else None
 report={'audit':'v64_3_6_dual_bottleneck_screen_comparison','winner':winner,'run_full_pipeline':bool(winner),'screens':screens,
         'note':'BCHA is justified only by a low frozen-family oracle; LBPR targets the independently proven pair-full value ceiling. Full promotion additionally requires a positive teacher-action signal.'}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,sort_keys=True),encoding='utf-8'); print(json.dumps(report,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
