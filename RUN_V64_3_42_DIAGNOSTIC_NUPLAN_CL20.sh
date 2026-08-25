#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR"
OUT_ROOT="${OUT_ROOT:-outputs_v64_3_42_eaf_icer_ovdr_screen_2gpu_v1}"
BDSE_VAL_CACHE="${BDSE_VAL_CACHE:-/data0/senzeyu2/dataset/nuplan/data/cache/bdse_val_v2}"
DESIGN_EXCLUDE_TOKENS="${DESIGN_EXCLUDE_TOKENS:-bdse/configs/v64_3_32_design_exclude_v64_3_30_3_screen_tokens.txt}"
EAF_V64_3_13_ROOT="${EAF_V64_3_13_ROOT:-outputs_v64_3_13_eaf_dmvr_screen_2gpu_v1}"
GPU="${GPU:-0}"; CHALLENGE="${CHALLENGE:-closed_loop_nonreactive_agents}"
NUPLAN_ROOT="${NUPLAN_ROOT:?set NUPLAN_ROOT to the nuPlan root containing maps/exp}"
NUPLAN_DB_ROOT="${NUPLAN_DB_ROOT:?set NUPLAN_DB_ROOT to the nuPlan DB root for these val scenarios}"
[[ -d "$BDSE_VAL_CACHE" && -s "$DESIGN_EXCLUDE_TOKENS" ]] || { echo 'STOP: missing val cache/design-exclusion tokens' >&2; exit 2; }
if [[ -z "${EAF_CKPT:-}" ]]; then
  # Prefer the exact checkpoint already used by the V42 main launcher if exported.
  echo 'STOP: export EAF_CKPT to the same V64.3.13 checkpoint used by V42' >&2; exit 2
fi
[[ -s "$EAF_CKPT" ]] || { echo "STOP: missing checkpoint $EAF_CKPT" >&2; exit 2; }
V20="bdse/configs/v64_3_20_icer_dc_dual.yaml"; PRESERVE="$OUT_ROOT/provenance/v64_3_42_preserve.yaml"; RSMR="$OUT_ROOT/provenance/v64_3_42_rsmr.yaml"; EPV="$OUT_ROOT/provenance/v64_3_42_epv_raw.yaml"; JOINT="$OUT_ROOT/provenance/v64_3_42_joint_raw.yaml"
for f in "$V20" "$PRESERVE" "$RSMR" "$EPV" "$JOINT"; do [[ -s "$f" ]] || { echo "STOP: missing diagnostic config $f; run the V42 main launcher through TRAIN fitting first" >&2; exit 2; }; done
DROOT="$OUT_ROOT/diagnostic_nuplan_cl20"; mkdir -p "$DROOT"
TOK="$DROOT/scenario_tokens.json"
python - "$BDSE_VAL_CACHE" "$DESIGN_EXCLUDE_TOKENS" "$TOK" <<'PY'
import hashlib,json,sys
from pathlib import Path
import numpy as np
from bdse.data.nuplan_dataset import PreprocessedBDSEDataset
cache=Path(sys.argv[1]); excluded={x.strip() for x in open(sys.argv[2]) if x.strip()}; out=Path(sys.argv[3])
ds=PreprocessedBDSEDataset(cache,split=['val'],max_scenarios=None)
paths=ds.build_index(); present=[]; seen=set()
for p in paths:
 try:
  with np.load(p,allow_pickle=False) as z:
   v=z['scenario_token']; t=str(v.item() if v.shape==() else v.reshape(-1)[0])
 except Exception: continue
 if t in excluded and t not in seen: seen.add(t); present.append(t)
# Label-free deterministic diagnostic subset from already design-excluded scenarios.
present.sort(key=lambda t: hashlib.sha256(('v64.3.42-diagnostic-cl20-v1|'+t).encode()).hexdigest())
sel=present[:20]
if len(sel)!=20: raise SystemExit(f'STOP: only {len(sel)} excluded val tokens found for diagnostic CL20')
out.write_text(json.dumps(sel,indent=2)+'\n')
print('diagnostic tokens',len(sel),hashlib.sha256(('\n'.join(sel)+'\n').encode()).hexdigest())
PY
TOKEN_OVERRIDE=$(python - "$TOK" <<'PY'
import json,sys
print('scenario_filter.scenario_tokens='+json.dumps(json.load(open(sys.argv[1])),separators=(',',':')))
PY
)
run_arm(){ local name="$1" cfg="$2"; local od="$DROOT/$name"; mkdir -p "$od"; CUDA_VISIBLE_DEVICES="$GPU" python -m bdse.experiments.evaluate_closed_loop --config "$cfg" --checkpoint "$EAF_CKPT" --device cuda --challenge "$CHALLENGE" --output-dir "$od" --experiment-uid "v42_diag_${name}_cl20" --scenario-builder nuplan --worker single_machine_thread_pool --hydra-full-error --nuplan-data-root "$NUPLAN_ROOT" --nuplan-map-root "$NUPLAN_ROOT/maps" --nuplan-exp-root "$NUPLAN_ROOT/exp" --nuplan-db-root "$NUPLAN_DB_ROOT" -- "$TOKEN_OVERRIDE" scenario_filter.limit_total_scenarios=20 scenario_filter.shuffle=false worker.max_workers=1 run_metric=true '~callback.simulation_log_callback' > "$od/run.log" 2>&1; }
for spec in "v20:$V20" "preserve:$PRESERVE" "rsmr:$RSMR" "epv_raw:$EPV" "ovdr_raw:$JOINT"; do name=${spec%%:*}; cfg=${spec#*:}; echo "Running diagnostic CL20 $name"; run_arm "$name" "$cfg"; done
python - "$DROOT" "$TOK" "$CHALLENGE" <<'PY'
import json,sys,hashlib
from pathlib import Path
from bdse.tools.run_external_closed_loop_comparison import final_metric_row
root=Path(sys.argv[1]); toks=json.load(open(sys.argv[2])); challenge=sys.argv[3]; arms=['v20','preserve','rsmr','epv_raw','ovdr_raw']; rep={'audit':'v64_3_42_diagnostic_closed_loop_only_not_promotion_evidence','challenge':challenge,'scenario_count':len(toks),'scenario_tokens_sha256':hashlib.sha256(('\n'.join(toks)+'\n').encode()).hexdigest(),'causal_role':'diagnostic_only_on_already_design_excluded_population; must not tune V42 or replace TRAIN/double-fresh gates','arms':{}}
for a in arms:
 metrics,path=final_metric_row(root/a); rep['arms'][a]={'metric_file':str(path),'metrics':metrics}
(root/'diagnostic_summary.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
keys=['score','no_ego_at_fault_collisions','time_to_collision_within_bound','drivable_area_compliance','ego_progress_along_expert_route','ego_is_comfortable']
lines=['# V64.3.42 diagnostic nuPlan CL20','', '**Diagnostic only. This population is already design-excluded and these results must not modify V42 gates or hyperparameters.**','', '| Arm | collision | TTC | drivable | progress | comfort |','|---|---:|---:|---:|---:|---:|']
for a in arms:
 m=rep['arms'][a]['metrics']; get=lambda *ks: next((float(m[k]) for k in ks if k in m),float('nan'))
 lines.append(f"| {a} | {get('no_ego_at_fault_collisions','collision_avoidance'):.4f} | {get('time_to_collision_within_bound','time_to_collision'):.4f} | {get('drivable_area_compliance'):.4f} | {get('ego_progress_along_expert_route','progress'):.4f} | {get('ego_is_comfortable','comfort'):.4f} |")
(root/'diagnostic_summary.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'pass':True,'summary':str(root/'diagnostic_summary.json')},indent=2))
PY
