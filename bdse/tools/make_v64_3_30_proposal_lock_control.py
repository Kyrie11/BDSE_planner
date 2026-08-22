from __future__ import annotations
import argparse
from pathlib import Path
import yaml


def main():
    ap=argparse.ArgumentParser(description='Create V64.3.30 same-proposal DRC causal control without evidence rebinding.')
    ap.add_argument('--v25-drc-config',required=True)
    ap.add_argument('--proposal-lock-v20-config',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    drc=yaml.safe_load(Path(a.v25_drc_config).read_text(encoding='utf-8'))
    lock=yaml.safe_load(Path(a.proposal_lock_v20_config).read_text(encoding='utf-8'))
    # Change only selector-side proposal locking metadata; keep V25 DRC memory,
    # heads, risk K/boundary and all planner/runtime contracts from the V25 arm.
    drc['selector']['proposal_conditioned_witness_rebinding']=lock['selector']['proposal_conditioned_witness_rebinding']
    drc['selector']['frontier_contrast_rebinding']=lock['selector']['frontier_contrast_rebinding']
    drc.setdefault('metadata',{}).update({
        'algorithm_version':'V64.3.30-EAF-ICER-PLOCK-DRC',
        'fixed_planner_interface_evidence_budget':16,'fixed_proposal_top_m':24,
        'proposal_lock_control':True,'pcwer_proposal_conditioned_rebinding':False,'fcr_post_eaf_rebinding':False,
    })
    drc.setdefault('provenance',{})['algorithm_version']='V64.3.30-EAF-ICER-PLOCK-DRC'
    drc['provenance']['screening_only']=True
    exp=drc.setdefault('experiment',{})
    exp['name']='v64_3_30_eaf_icer_proposal_lock_aggregate_downside'
    exp['algorithm']='V64.3.30 proposal-lock DRC control: unchanged V25 DRC on original AOCC evidence with same-proposal-only confirmation'
    exp['evaluation_role']='operator_control_separating_no_fallback_from_pcwer_evidence_rebinding'
    exp['mechanism_chain']='fixed B=16/M=24 -> original AOCC evidence -> risk-free unique direct proposal -> unchanged V25 DRC confirms/vetoes same proposal only -> incumbent preservation'
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(yaml.safe_dump(drc,sort_keys=False,allow_unicode=True),encoding='utf-8')

if __name__=='__main__': main()
