from __future__ import annotations

"""V56 process-local nuPlan wrapper.

It preserves the exact V54 short-horizon treatment/control intervention and ego
sidecar, and adds a second sidecar containing current realized interaction / TTC
/ route-constraint risk.  No final metric or outcome is recomputed.
"""
import json, os, runpy, threading
from pathlib import Path
from typing import Any

from bdse.tools import nuplan_v54_dynamic_response_run_simulation as v54
from bdse.planner.paired_constraint_process_retention import instantaneous_constraint_risk

_LOCK=threading.Lock(); _MARK="_bdse_v64_3_56_constraint_process_sidecar"

def install_constraint_sidecar() -> None:
    PlannerClass=v54._resolve_nuplan_planner_class()
    if getattr(PlannerClass,_MARK,False): return
    original=PlannerClass.compute_planner_trajectory
    def wrapped(self:Any,current_input:Any):
        exposure=int(os.environ.get("BDSE_V54_EXPOSURE_TICKS","-1"))
        idx=int(getattr(getattr(current_input,"iteration",None),"index",-1))
        if idx<0 or idx>exposure: return original(self,current_input)
        out=original(self,current_input)
        target=getattr(self.core,"_pior_bound_target",None)
        if not isinstance(target,dict): raise RuntimeError(f"V56 RCPR missing bound target iteration={idx}")
        tok=str(target.get("scenario_token",""));
        if not tok: raise RuntimeError("V56 RCPR target lacks scenario_token")
        pcfg=((self.core.cfg.get("selected_outcome_probe",{}) or {}) if isinstance(self.core.cfg,dict) else {})
        arm=str(pcfg.get("arm","")).strip().lower()
        if arm not in {"treatment","control"}: raise RuntimeError(f"V56 RCPR invalid arm {arm!r}")
        runtime=self._runtime_from_planner_input(current_input)
        risk=instantaneous_constraint_risk(runtime,self.core.cfg)
        diag_path=os.environ.get("BDSE_CLOSED_LOOP_DIAG","")
        if not diag_path: raise RuntimeError("V56 RCPR requires BDSE_CLOSED_LOOP_DIAG")
        path=Path(diag_path).with_name("v56_constraint_process_events.jsonl")
        it=current_input.iteration
        row={"scenario_token":tok,"arm":arm,"iteration_index":idx,"time_us":int(it.time_point.time_us),
             "constraint_risk":[float(v) for v in risk],"channel_order":["agent_occupancy_risk","agent_ttc_risk","hard_offroute_excess_m"],
             "state_source":"current_simulated_runtime_only","post_intervention":bool(idx>0),"exposure_ticks":exposure}
        path.parent.mkdir(parents=True,exist_ok=True)
        with _LOCK:
            with path.open("a",encoding="utf-8") as f: f.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
        return out
    PlannerClass.compute_planner_trajectory=wrapped; setattr(PlannerClass,_MARK,True)

def main()->None:
    if os.environ.get("BDSE_V56_CONSTRAINT_PROCESS")!="1": raise RuntimeError("V56 RCPR wrapper requires BDSE_V56_CONSTRAINT_PROCESS=1")
    v54.install_short_horizon(); v54.install_dynamic_response_sidecar(); install_constraint_sidecar()
    print("[BDSE-V56-RCPR] short-horizon ego + constraint-process sidecars enabled; final metrics disabled",flush=True)
    runpy.run_module("nuplan.planning.script.run_simulation",run_name="__main__")
if __name__=="__main__": main()
