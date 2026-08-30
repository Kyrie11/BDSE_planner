from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np, yaml
import bdse.tools.fit_v64_3_22_eaf_icer_tcr as v22

FOLDS=5
FOLD_SEED='v64.3.23-eaf-icer-rcr-scene-crossfit-v1'  # deliberately retained; no split search
KS=(32,64)
MIN_EDGES=1024; MIN_SCENES=256; MIN_FOLD_SCENES=40; MIN_SELECTED=8
ATTR_NAMES=[f'candidate_atom_signed_spectrum_{i:02d}' for i in range(16)]+[f'delta_atom_signed_spectrum_{i:02d}' for i in range(16)]


def _icer(cfg:dict[str,Any])->dict[str,Any]:
    return cfg['runtime']['decisive_frontier_value']['incumbent_contrastive_extremal_recovery']

def _fold(tok:str)->int:
    h=hashlib.sha256((FOLD_SEED+'::'+tok).encode()).digest(); return int.from_bytes(h[:8],'big')%FOLDS

def _row_attr(r:dict[str,Any])->list[float]|None:
    vals=[]
    for n in ATTR_NAMES:
        try:v=float(r.get(f'icer_attribution_resolved_{n}',np.nan))
        except Exception:return None
        if not np.isfinite(v): return None
        vals.append(v)
    return vals

def _build(by:dict[str,list[dict[str,Any]]], mode:str)->dict[str,Any]:
    X=[]; delta=[]; tok=[]; sup=[]; scalar=[]; action=[]
    eligible_before_feature=0; missing_attr=0
    for t,rs in by.items():
        if not rs: continue
        anchor=int(rs[0].get('anchor_action',-1)); legacy=int(rs[0].get('raw_top_action',-1))
        if legacy<0 or legacy==anchor: continue
        inc=next((r for r in rs if int(r.get('challenger_action',-999))==legacy),None)
        if inc is None or float(inc.get('icer_admissible',inc.get('dacer_admissible',0.0)))<.5: continue
        inc_tm=float(inc.get('teacher_margin',np.nan))
        if not np.isfinite(inc_tm): continue
        for r in rs:
            ch=int(r.get('challenger_action',-1))
            if ch<0 or ch in {anchor,legacy}: continue
            if float(r.get('icer_admissible',r.get('dacer_admissible',0.0)))<.5: continue
            tm=float(r.get('teacher_margin',np.nan)); s=float(r.get('icer_support_logit',np.nan)); d=float(r.get('icer_scalar_dominance_logit',np.nan))
            if not all(np.isfinite(x) for x in [tm,s,d]) or not(s>0 and d>0): continue
            eligible_before_feature+=1
            base=[]; bad=False
            for n in v22._ICER_REGRET_RISK_EVIDENCE_BASE_NAMES:
                q=float(r.get(f'icer_feature_{n}',np.nan))
                if not np.isfinite(q): bad=True; break
                base.append(q)
            if bad: continue
            if mode=='aggregate': feat=base
            elif mode=='attribution_resolved':
                ar=_row_attr(r)
                if ar is None: missing_attr+=1; continue
                feat=base+ar
            else: raise ValueError(mode)
            X.append(feat); delta.append(tm-inc_tm); tok.append(str(t)); sup.append(s); scalar.append(d); action.append(ch)
    if eligible_before_feature<MIN_EDGES: raise SystemExit(f'insufficient frozen replacement population: {eligible_before_feature}')
    if mode=='attribution_resolved' and missing_attr:
        raise SystemExit(f'STOP TRAIN INSTRUMENTATION: missing full attribution spectrum on {missing_attr}/{eligible_before_feature} eligible replacement edges')
    return dict(X=np.asarray(X,float),delta=np.asarray(delta,float),tok=np.asarray(tok,object),support=np.asarray(sup,float),scalar=np.asarray(scalar,float),action=np.asarray(action,int),eligible_before_feature=eligible_before_feature,missing_attr=missing_attr)

def _metric_weight(mode:str)->np.ndarray:
    if mode=='aggregate': return np.full(18,1/18,float)
    if mode=='attribution_resolved':
        return np.r_[np.full(18,1/18,float),np.full(16,1/16,float),np.full(16,1/16,float)]
    raise ValueError(mode)

def _memory(X:np.ndarray,mode:str):
    mean=X.mean(0); std=np.maximum(X.std(0),1e-6); mw=_metric_weight(mode)
    if len(mw)!=X.shape[1]: raise RuntimeError('metric schema mismatch')
    Z=((X-mean)/std)*np.sqrt(mw)
    return Z,mean,std,mw

def _score(trainX,trainY,qX,mode,cert):
    Z,mean,std,mw=_memory(trainX,mode); Q=((qX-mean)/std)*np.sqrt(mw)
    d2=np.maximum(np.sum(Q*Q,1)[:,None]+np.sum(Z*Z,1)[None,:]-2*Q@Z.T,0)
    rows=np.arange(len(Q))[:,None]; out=[]
    for k in KS:
        kk=min(k,len(Z)); nbr=np.argpartition(d2,kk-1,axis=1)[:,:kk]; dist=np.sqrt(d2[rows,nbr]); w=1/np.maximum(dist,1e-6); w/=np.maximum(w.sum(1,keepdims=True),1e-12); y=trainY[nbr]; mu=np.sum(w*y,1)
        if cert=='mean_se':
            var=np.sum(w*(y-mu[:,None])**2,1); neff=1/np.maximum(np.sum(w*w,1),1e-12); b=mu-np.sqrt(var/np.maximum(neff,1))
        elif cert=='downside_rms':
            dn=np.minimum(y,0); b=mu-np.sqrt(np.sum(w*dn*dn,1))
        else: raise ValueError(cert)
        out.append(b)
    return np.min(np.stack(out,1),1)

def _selection(data,score,hold):
    toks=data['tok']; sup=data['support']; scalar=data['scalar']; delta=data['delta']; action=data['action']; sel=[]; opp=cap=sc=0
    for t in sorted(hold):
        idx=np.flatnonzero(toks==t)
        if not len(idx): continue
        sc+=1; opp+=int(np.any(delta[idx]>0)); ok=idx[(sup[idx]>0)&(scalar[idx]>0)&(score[idx]>0)]
        if len(ok):
            j=sorted(ok.tolist(),key=lambda q:(-float(scalar[q]),-float(score[q]),int(action[q])))[0]; sel.append(float(delta[j])); cap+=int(delta[j]>0)
    a=np.asarray(sel,float)
    return dict(scene_count=sc,count=int(len(a)),precision=float(np.mean(a>0)) if len(a) else float('nan'),sum=float(a.sum()) if len(a) else 0.0,mean=float(a.mean()) if len(a) else float('nan'),worst=float(a.min()) if len(a) else float('nan'),opportunities=opp,capture=float(cap/opp) if opp else float('nan'))

def _crossfit(data,mode,cert):
    X=data['X']; y=data['delta']; toks=data['tok']; unique=sorted(set(map(str,toks)))
    if len(X)<MIN_EDGES or len(unique)<MIN_SCENES: raise SystemExit(f'insufficient TRAIN support {mode}/{cert}: edges={len(X)} scenes={len(unique)}')
    folds=[]
    for f in range(FOLDS):
        hold={t for t in unique if _fold(t)==f}
        if len(hold)<MIN_FOLD_SCENES: raise SystemExit(f'fold too small {f}: {len(hold)}')
        hm=np.array([str(t) in hold for t in toks]); fit=~hm; s=np.full(len(X),np.nan); s[hm]=_score(X[fit],y[fit],X[hm],mode,cert); m=_selection(data,s,hold); m['fold']=f; m['hold_scenes']=len(hold); m['path_safe']=bool(m['count']>=MIN_SELECTED and m['sum']>=-1e-9); folds.append(m)
    return dict(mode=mode,certificate=cert,folds=folds,all_folds_path_safe=all(x['path_safe'] for x in folds),fold_pass_count=sum(x['path_safe'] for x in folds),selected_count=sum(x['count'] for x in folds),teacher_improvement_sum=float(sum(x['sum'] for x in folds)),mean_precision=float(np.nanmean([x['precision'] for x in folds])),mean_capture=float(np.nanmean([x['capture'] for x in folds])))

def _save(path:Path,data,mode,cert):
    Z,mean,std,mw=_memory(data['X'],mode); names=[f'evidence::{n}' for n in v22._ICER_REGRET_RISK_EVIDENCE_BASE_NAMES]
    if mode=='attribution_resolved': names += [f'attribution::{n}' for n in ATTR_NAMES]
    path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,memory_metric_z=Z.astype('f4'),teacher_improvement=data['delta'].astype('f4'),feature_mean=mean.astype('f4'),feature_std=std.astype('f4'),feature_names=np.asarray(names,dtype='U128'),feature_metric_weight=mw.astype('f4'),neighbor_k_values=np.asarray(KS,'i4'),se_multiplier=np.asarray([1.0],'f4'),certificate_kind=np.asarray(['mean_minus_downside_rms' if cert=='downside_rms' else 'mean_minus_standard_error'],dtype='U64'),downside_multiplier=np.asarray([1.0],'f4'))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),row_count=len(data['delta']),feature_count=data['X'].shape[1],mode=mode,certificate=cert)

def _cfg(base,mem,mode,cert,tag):
    c=yaml.safe_load(yaml.safe_dump(base,sort_keys=False)); ic=_icer(c); ic.update(dict(model_type=('frozen_support_scalar_dominance_plus_attribution_resolved_local_regret_certificate' if mode=='attribution_resolved' else 'frozen_support_scalar_dominance_plus_evidence_local_regret_certificate'),dominance_policy='scalar_only',incumbent_retention_policy='preserve_admissible_incumbent',regret_risk_enabled=True,retention_regret_risk_enabled=False,replacement_regret_risk_enabled=True,regret_risk_model_type='local_multiscale_downside_regret_certificate' if cert=='downside_rms' else 'local_multiscale_regret_lower_bound',regret_risk_feature_mode='attribution_resolved' if mode=='attribution_resolved' else 'evidence_only',replacement_local_regret_memory_path=mem['path'],replacement_local_regret_memory_sha256=mem['sha256'],replacement_local_regret_neighbor_k_values=list(KS),replacement_local_regret_certificate='mean_minus_downside_rms' if cert=='downside_rms' else 'mean_minus_standard_error',replacement_regret_training_population='TRAIN_only_final_guard_admissible_support_positive_scalar_dominance_positive_alternatives',replacement_operator='preserve admissible incumbent by default; replace only if support>0 AND scalar_dominance>0 AND local_regret_certificate>0; rank by frozen scalar dominance; no signed-profile or transition hard gate',all_flagged_policy='preserve_legacy_for_structural_guard'))
    ver='V64.3.24-EAF-ICER-ARC-DARM-DBR'; c.setdefault('metadata',{})['algorithm_version']=ver;c.setdefault('provenance',{})['algorithm_version']=ver; e=c.setdefault('experiment',{}); e['name']='v64_3_24_'+tag;e['algorithm']='V64.3.24 EAF-ICER-ARC: Evidence-Attributed Incumbent-Contrastive Extremal Recovery with Attribution-Resolved Regret Certification';e['mechanism_chain']='fixed B<=16 -> frozen EAF frontier -> exact full selected-evidence attribution spectrum -> deployment-admissible frontier -> frozen support/scalar dominance -> local outcome-downside regret certificate -> incumbent-default extremal replacement -> unchanged guards'
    return c

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train-frontier-edges',required=True);ap.add_argument('--base-v20-dual-config',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--output-train-token-file',required=True);ap.add_argument('--output-report',required=True);a=ap.parse_args(); out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    by=v22._load_scenes(Path(a.train_frontier_edges)); ag=_build(by,'aggregate'); ar=_build(by,'attribution_resolved')
    if len(ag['delta'])!=len(ar['delta']): raise SystemExit(f'STOP TRAIN INSTRUMENTATION: aggregate/attribution populations differ {len(ag["delta"])} vs {len(ar["delta"])}')
    variants={
      'aggregate_meanse':(ag,'aggregate','mean_se'),
      'aggregate_downside':(ag,'aggregate','downside_rms'),
      'attribution_meanse':(ar,'attribution_resolved','mean_se'),
      'attribution_downside':(ar,'attribution_resolved','downside_rms'),
    }
    cf={k:_crossfit(*v) for k,v in variants.items()}; main=cf['attribution_downside']; gate=bool(main['all_folds_path_safe'] and main['selected_count']>=64 and main['teacher_improvement_sum']>=-1e-9)
    # Engineering fix carried in the V25 delivery: a legitimate fail-closed TRAIN
    # gate must still leave its cross-fit diagnostics and token manifest behind.
    # V24 previously raised before writing either artifact, which did not alter
    # actions/results but made a correct STOP unnecessarily hard to audit.
    toks=sorted(by); tp=Path(a.output_train_token_file);tp.parent.mkdir(parents=True,exist_ok=True);tp.write_text('\n'.join(toks)+'\n')
    cfgs={}; mems={}
    rep=dict(audit='v64_3_24_eaf_icer_arc_train_fit',algorithm='V64.3.24 EAF-ICER-ARC',train_scene_count=len(by),replacement_edges=len(ar['delta']),attribution_coverage=1.0,fold_seed=FOLD_SEED,neighbor_k_values=list(KS),crossfit=cf,train_gate_pass=gate,memories=mems,configs=cfgs,train_token_manifest=dict(path=str(tp),count=len(toks),sha256=hashlib.sha256(tp.read_bytes()).hexdigest()),fresh_validation_used=False)
    rp=Path(a.output_report); rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(rep,indent=2,sort_keys=True))
    if not gate: raise SystemExit('STOP TRAIN ARC: attribution-resolved downside certificate is not selected-path safe in all fixed scene folds; do not spend fresh validation GPU')
    base=yaml.safe_load(Path(a.base_v20_dual_config).read_text())
    for k,(d,m,c) in variants.items():
        mp=out/f'v64_3_24_{k}_memory.npz'; mem=_save(mp,d,m,c); cp=out/f'v64_3_24_{k}.yaml'; cp.write_text(yaml.safe_dump(_cfg(base,mem,m,c,k),sort_keys=False)); mems[k]=mem;cfgs[k]=str(cp)
    rep['memories']=mems; rep['configs']=cfgs; rp.write_text(json.dumps(rep,indent=2,sort_keys=True))
    print(json.dumps({'pass':True,'train_gate_pass':gate,'main_sum':main['teacher_improvement_sum'],'main_fold_pass_count':main['fold_pass_count'],'main_count':main['selected_count']},sort_keys=True))
if __name__=='__main__': main()
