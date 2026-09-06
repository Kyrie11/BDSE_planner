from __future__ import annotations

"""Collect the V64.3.56 paired realized interaction/safety constraint process.

The exact V50.5 treatment/control one-shot intervention is replayed only through
the first scheduled replan.  Final nuPlan metrics and outcome labels are not
recomputed.  This is state/mediator collection only.
"""
import argparse, concurrent.futures, json, os, shutil, subprocess, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import yaml

from bdse.planner.paired_constraint_process_retention import CONSTRAINT_PROFILE_SCHEMA, paired_constraint_profile
from bdse.tools import run_v64_3_50_pior_paired_closed_loop as base
from bdse.tools import run_v64_3_50_5_pior_paired_closed_loop as safe

NUPLAN_MODULE="bdse.tools.nuplan_v56_constraint_process_run_simulation"
SIDE_NAME="v56_constraint_process_events.jsonl"
CERT_NAME=".v56_constraint_batch_complete.json"


def _sha(path:Path)->str: return base._sha256(path)

def _config_replan_ticks(path:Path)->int:
    d=yaml.safe_load(path.read_text(encoding="utf-8")); n=int(((d.get("planner",{}) or {}).get("replan_interval_ticks",-1)))
    if n != 5: raise RuntimeError(f"V56 RCPR requires frozen exposure/replan ticks=5, got {n} from {path}")
    return n

def base54_planned_d(path:Path,tokens:list[str])->dict[str,float]:
    out={}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r=json.loads(line); tok=str(r.get("scenario_token",""))
        if tok in out: raise RuntimeError(f"V56 RCPR duplicate V53 profile token={tok}")
        out[tok]=float(r.get("execution_contrast_linf",float("nan")))
    if set(out)!=set(tokens) or any((not np.isfinite(v)) or v<0 for v in out.values()):
        raise RuntimeError(f"V56 RCPR V53 planned-profile population/value mismatch {len(out)}/{len(tokens)}")
    return out

def _read_trace(path:Path,tokens:list[str],arm:str,exposure:int)->dict[str,list[dict[str,Any]]]:
    expected=set(tokens); by:dict[str,dict[int,dict[str,Any]]]={}
    if not path.is_file(): raise RuntimeError(f"V56 RCPR missing sidecar {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r=json.loads(line); tok=str(r.get("scenario_token","")); a=str(r.get("arm","")); idx=int(r.get("iteration_index",-1))
        if tok not in expected or a!=arm or idx<0 or idx>exposure: raise RuntimeError(f"V56 RCPR invalid sidecar row token={tok} arm={a} idx={idx}")
        if int(r.get("exposure_ticks",-1))!=exposure: raise RuntimeError(f"V56 RCPR exposure drift token={tok}")
        vals=[float(v) for v in r.get("constraint_risk",[])]
        if len(vals)!=3 or any((not np.isfinite(v)) or v < -1e-12 for v in vals):
            raise RuntimeError(f"V56 RCPR constraint_risk invalid token={tok} idx={idx} values={vals}")
        if list(r.get("channel_order",[])) != ["agent_occupancy_risk","agent_ttc_risk","hard_offroute_excess_m"]:
            raise RuntimeError(f"V56 RCPR channel-order drift token={tok} idx={idx}")
        if str(r.get("state_source","")) != "current_simulated_runtime_only":
            raise RuntimeError(f"V56 RCPR state-source drift token={tok} idx={idx}")
        slot=by.setdefault(tok,{})
        if idx in slot: raise RuntimeError(f"V56 RCPR duplicate sample token={tok} idx={idx}")
        slot[idx]=r
    want=list(range(exposure+1)); out={}
    for tok in tokens:
        got=by.get(tok,{})
        if sorted(got)!=want: raise RuntimeError(f"V56 RCPR incomplete trace token={tok} indices={sorted(got)}")
        rows=[got[i] for i in want]; ts=[int(r["time_us"]) for r in rows]
        if any(b<=a for a,b in zip(ts,ts[1:])): raise RuntimeError(f"V56 RCPR non-increasing timestamps token={tok}")
        out[tok]=rows
    return out

def _cert_valid(root:Path,*,tokens:list[str],arm:str,cfg_sha:str,ckpt_sha:str,exposure:int,meta:dict[str,dict[str,Any]])->dict[str,Any]|None:
    cp=root/CERT_NAME; side=root/SIDE_NAME; diag=root/"pior_probe_events.jsonl"; target=root/"pior_probe_targets.json"
    if not all(p.is_file() for p in (cp,side,diag,target)): return None
    try:
        c=json.loads(cp.read_text(encoding="utf-8"))
        checks=[c.get("complete") is True,str(c.get("arm",""))==arm,int(c.get("scenario_count",-1))==len(tokens),
                str(c.get("scenario_token_sha256",""))==base._token_sha(tokens),str(c.get("config_sha256",""))==cfg_sha,
                str(c.get("checkpoint_sha256",""))==ckpt_sha,int(c.get("exposure_ticks",-1))==exposure,
                int(c.get("successful",-1))==len(tokens),int(c.get("failed",-1))==0,int(c.get("probe_fired_count",-1))==len(tokens),
                str(c.get("constraint_sidecar_sha256",""))==_sha(side),str(c.get("probe_events_sha256",""))==_sha(diag)]
        if not all(checks): return None
        base._validate_probe_events(diag,tokens=tokens,meta=meta,arm=arm); _read_trace(side,tokens,arm,exposure); return c
    except Exception: return None

def _run_batch(*,arm:str,gpu:str,cfg:Path,checkpoint:Path,tokens:list[str],meta:dict[str,dict[str,Any]],nuplan_root:Path,challenge:str,
               arm_root:Path,workers:int,batch_index:int,batch_count:int,exposure:int,heartbeat_seconds:float,resume:bool)->dict[str,Any]:
    root=arm_root/"batches"/f"batch_{batch_index:04d}"; cfg_sha=_sha(cfg); ckpt_sha=_sha(checkpoint)
    if resume:
        c=_cert_valid(root,tokens=tokens,arm=arm,cfg_sha=cfg_sha,ckpt_sha=ckpt_sha,exposure=exposure,meta=meta)
        if c is not None:
            print(f"[V56-RESUME] arm={arm} batch={batch_index+1}/{batch_count} n={len(tokens)}",flush=True); return c
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True,exist_ok=True)
    raw_files=base._batch_raw_files(tokens,meta); payload=base._probe_target_payload(tokens,meta); target=root/"pior_probe_targets.json"
    if base._write_probe_target_file(target,payload)!=base._payload_sha256(payload): raise RuntimeError("V56 RCPR target semantic hash drift")
    (root/"scenario_tokens.json").write_text(json.dumps(tokens,indent=2),encoding="utf-8")
    diag=root/"pior_probe_events.jsonl"; side=root/SIDE_NAME
    token_override="scenario_filter.scenario_tokens="+json.dumps(tokens,separators=(",",":"))
    cmd=[sys.executable,"-m","bdse.experiments.evaluate_closed_loop","--config",str(cfg),"--checkpoint",str(checkpoint),"--device","cuda",
         "--challenge",challenge,"--metric-aggregator",f"{challenge}_weighted_average","--output-dir",str(root),"--experiment-uid",f"v64_3_56_rcpr_{arm}_b{batch_index:04d}",
         "--nuplan-module",NUPLAN_MODULE,"--scenario-builder","nuplan","--worker","single_machine_thread_pool","--hydra-full-error",
         "--nuplan-data-root",str(nuplan_root),"--nuplan-map-root",str(nuplan_root/"maps"),"--nuplan-exp-root",str(nuplan_root/"exp"),"--nuplan-db-files",*raw_files,
         "--",token_override,f"scenario_filter.limit_total_scenarios={len(tokens)}","scenario_filter.shuffle=false","scenario_filter.log_names=null",
         *base._anchor_start_mapping_overrides(),f"worker.max_workers={int(workers)}","run_metric=false","~callback.simulation_log_callback"]
    env=os.environ.copy(); env.update({"PYTHONUNBUFFERED":"1","CUDA_VISIBLE_DEVICES":str(gpu),"BDSE_SHARE_MODEL_PER_PROCESS":"1","BDSE_SERIALIZE_GPU_INFERENCE":"0",
        "BDSE_SHARD_PLANNERS_ACROSS_GPUS":"0","BDSE_CLOSED_LOOP_DIAG":str(diag.resolve()),"BDSE_CLOSED_LOOP_DIAG_MODE":"pior_probe_events","BDSE_PIOR_TARGETS_FILE":str(target.resolve()),
        "BDSE_STRICT_CLOSED_LOOP_DIAG":"1","BDSE_PROFILE_CLOSED_LOOP":"0","BDSE_V54_DYNAMIC_RESPONSE":"1","BDSE_V54_EXPOSURE_TICKS":str(exposure),
        "BDSE_V56_CONSTRAINT_PROCESS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
    log=root/"run.log"; started=time.time(); print(f"[V56-BATCH-START] arm={arm} batch={batch_index+1}/{batch_count} gpu={gpu} n={len(tokens)}",flush=True)
    with log.open("w",encoding="utf-8") as f: proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    hb=max(5.0,float(heartbeat_seconds))
    while True:
        try: rc=proc.wait(timeout=hb); break
        except subprocess.TimeoutExpired:
            elapsed=time.time()-started; fires=max(0,base._count_probe_fires(diag)); util,mem=base._gpu_stats(gpu)
            print(f"[V56-TICK] arm={arm} batch={batch_index+1}/{batch_count} elapsed={elapsed/60:.1f}m probes={fires}/{len(tokens)} gpu={util}/{mem}",flush=True)
    text=log.read_text(encoding="utf-8",errors="replace"); succ,fail=base._parse_success(text); fired=base._count_probe_fires(diag)
    if rc!=0 or succ!=len(tokens) or fail!=0 or fired!=len(tokens):
        tail="\n".join(text.replace("\r","\n").splitlines()[-50:]); raise RuntimeError(f"V56 RCPR {arm} batch failed rc={rc} success={succ} failed={fail} probes={fired}/{len(tokens)}\n{tail}")
    base._validate_probe_events(diag,tokens=tokens,meta=meta,arm=arm); _read_trace(side,tokens,arm,exposure)
    wall=time.time()-started; c={"complete":True,"algorithm_version":"V64.3.56-EAF-ICER-RCPR","arm":arm,"batch_index":batch_index,"batch_count":batch_count,
        "scenario_count":len(tokens),"scenario_token_sha256":base._token_sha(tokens),"config_sha256":cfg_sha,"checkpoint_sha256":ckpt_sha,"challenge":challenge,
        "successful":succ,"failed":fail,"probe_fired_count":fired,"exposure_ticks":exposure,"run_metric":False,"outcome_labels_recollected":False,
        "constraint_sidecar":str(side),"constraint_sidecar_sha256":_sha(side),"probe_events_sha256":_sha(diag),"wall_time_s":wall}
    (root/CERT_NAME).write_text(json.dumps(c,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"[V56-BATCH-DONE] arm={arm} batch={batch_index+1}/{batch_count} wall={wall/60:.1f}m",flush=True); return c

def _run_arm(**kw)->dict[str,Any]:
    tokens=kw.pop("tokens"); meta=kw.pop("meta"); batch_size=int(kw.pop("batch_size")); output_root=kw.pop("output_root"); arm=kw["arm"]
    batches=base._collision_safe_batches(tokens,meta,batch_size=max(1,batch_size),first_batch_size=min(8,len(tokens))); ar=output_root/arm; ar.mkdir(parents=True,exist_ok=True)
    certs=[]
    for bi,bt in enumerate(batches): certs.append(_run_batch(tokens=bt,meta=meta,arm_root=ar,batch_index=bi,batch_count=len(batches),**kw))
    return {"arm":arm,"batch_count":len(certs),"scenario_count":sum(int(c["scenario_count"]) for c in certs),"total_batch_wall_time_s":sum(float(c["wall_time_s"]) for c in certs),"batches":certs}

def _collect(root:Path,arm:str,tokens:list[str],exposure:int)->dict[str,list[dict[str,Any]]]:
    out={}
    for p in sorted((root/arm/"batches").glob(f"batch_*/{SIDE_NAME}")):
        toks=sorted({str(json.loads(x).get("scenario_token","")) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()})
        part=_read_trace(p,toks,arm,exposure)
        for tok,rows in part.items():
            if tok in out: raise RuntimeError(f"V56 RCPR duplicate token across batches {tok}")
            out[tok]=rows
    if set(out)!=set(tokens): raise RuntimeError(f"V56 RCPR arm population mismatch {arm} {len(out)}/{len(tokens)}")
    return out

def main()->None:
    p=argparse.ArgumentParser();
    for name in ("manifest","treatment-config","control-config","checkpoint","nuplan-root","v53-operator-profiles","output-root","output-profiles","output-report"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--gpu-treatment",default="0");p.add_argument("--gpu-control",default="1");p.add_argument("--workers",type=int,default=4);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--heartbeat-seconds",type=float,default=30.0);p.add_argument("--challenge",default="closed_loop_nonreactive_agents");p.add_argument("--resume",action="store_true")
    a=p.parse_args(); safe._assert_frozen_base_runner(); tokens,meta,_=base._manifest(a.manifest)
    if len(tokens)!=502 or len(set(tokens))!=502: raise RuntimeError("V56 RCPR exact population must be 502")
    et=_config_replan_ticks(a.treatment_config); ec=_config_replan_ticks(a.control_config)
    if et!=ec: raise RuntimeError("V56 RCPR arm exposure mismatch")
    planned_d=base54_planned_d(a.v53_operator_profiles,tokens)
    kw=dict(checkpoint=a.checkpoint,tokens=tokens,meta=meta,nuplan_root=a.nuplan_root,challenge=str(a.challenge),output_root=a.output_root,workers=int(a.workers),batch_size=int(a.batch_size),exposure=et,heartbeat_seconds=float(a.heartbeat_seconds),resume=bool(a.resume))
    if str(a.gpu_treatment)!=str(a.gpu_control):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            ft=ex.submit(_run_arm,arm="treatment",gpu=str(a.gpu_treatment),cfg=a.treatment_config,**kw); fc=ex.submit(_run_arm,arm="control",gpu=str(a.gpu_control),cfg=a.control_config,**kw); ts,cs=ft.result(),fc.result()
    else:
        ts=_run_arm(arm="treatment",gpu=str(a.gpu_treatment),cfg=a.treatment_config,**kw); cs=_run_arm(arm="control",gpu=str(a.gpu_control),cfg=a.control_config,**kw)
    tr=_collect(a.output_root,"treatment",tokens,et); cr=_collect(a.output_root,"control",tokens,et); profiles=[]; zero_process=0; planned_equal=0
    for tok in sorted(tokens):
        rt,rc=tr[tok],cr[tok]; its=[int(x["iteration_index"]) for x in rt]; tts=[int(x["time_us"]) for x in rt]
        if its!=[int(x["iteration_index"]) for x in rc] or tts!=[int(x["time_us"]) for x in rc]: raise RuntimeError(f"V56 RCPR synchronization mismatch {tok}")
        prof=paired_constraint_profile([x["constraint_risk"] for x in rt],[x["constraint_risk"] for x in rc],iteration_indices=its,timestamps_us=tts)
        process_linf=max(abs(v) for v in prof["constraint_support_delta_process"])
        if process_linf<=1e-12: zero_process+=1
        if planned_d[tok] <= 1e-10:
            planned_equal += 1
            if process_linf > 1e-6:
                raise RuntimeError(f"V56 RCPR planned-physical-equal treatment/control constraint process diverged token={tok} linf={process_linf}")
        profiles.append({"scenario_token":tok,"planned_execution_contrast_linf":float(planned_d[tok]),**prof})
    if planned_equal != 38:
        raise RuntimeError(f"V56 RCPR frozen planned-equality signature drift {planned_equal} != 38")
    a.output_profiles.parent.mkdir(parents=True,exist_ok=True)
    with a.output_profiles.open("w",encoding="utf-8") as f:
        for r in profiles: f.write(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n")
    rep={"audit":"v64_3_56_paired_constraint_process_probe","algorithm_version":"V64.3.56-EAF-ICER-RCPR","pass":True,"scenario_count":502,"exposure_ticks":et,
         "paired_outcome_labels_recollected":False,"run_metric":False,"short_horizon_only":True,"profile_schema":CONSTRAINT_PROFILE_SCHEMA,"profile_sha256":_sha(a.output_profiles),
         "zero_process_count":zero_process,"planned_physical_equal_count":planned_equal,"planned_equal_process_tolerance":1e-6,
         "treatment_summary":ts,"control_summary":cs,"two_gpu_parallel":str(a.gpu_treatment)!=str(a.gpu_control)}
    a.output_report.write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps({"pass":True,"profiles":502,"sha256":rep["profile_sha256"]},sort_keys=True))
if __name__=="__main__": main()
