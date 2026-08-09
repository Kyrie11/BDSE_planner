from __future__ import annotations

import argparse, json
from pathlib import Path


def load(path: Path | None):
    return None if path is None else json.loads(path.read_text())


def viable(r):
    return bool(r and r.get('screen_instrumentation_valid') and r.get('adapter_parameter_activated') and r.get('adapter_forward_activated') and r.get('acra_wired'))


def improved(r):
    return bool(
        viable(r)
        and (r.get('delta_val_critical_topm_recall_micro') or 0.0) > 0.0
        and (r.get('delta_val_proposal_decisive_recall') is not None)
        and r['delta_val_proposal_decisive_recall'] >= -0.02
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--apwcca', type=Path, required=True)
    ap.add_argument('--apwrcca', type=Path)
    ap.add_argument('--value-probe', type=Path)
    ap.add_argument('--output', type=Path)
    args=ap.parse_args()
    a,b,c=load(args.apwcca),load(args.apwrcca),load(args.value_probe)
    if not viable(a):
        decision='engineering_or_gradient_failure'
        next_action='Do not change algorithm; repair instrumentation/gradient routing.'
    elif improved(a):
        decision='apwcca_positive'
        next_action='Promote AP-WCCA+ACRA to the full pipeline; AP-WRCCA is unnecessary unless used as an ablation.'
    elif b is None:
        decision='run_apwrcca'
        next_action='Run the matched AP-WRCCA+ACRA screen on the same data/subset.'
    elif not viable(b):
        decision='apwrcca_engineering_failure'
        next_action='Do not interpret AP-WRCCA; repair its instrumentation/gradient routing.'
    elif improved(b):
        decision='apwrcca_positive'
        next_action='Promote AP-WRCCA+ACRA; keep AP-WCCA as the winner-only ablation.'
    elif c is None:
        decision='run_literal_critical_value_probe'
        next_action='Both binary-target acquisition screens were valid but non-improving; run the literal-critical severity/value probe. Do not modify B/M/selector/certificate.'
    elif improved(c):
        decision='literal_critical_value_target_positive'
        next_action='Absorb literal-critical value alignment into the winning acquisition representation, then run full open-loop causal decomposition.'
    else:
        decision='acquisition_representation_bottleneck'
        next_action='All clean acquisition targets failed. The next algorithm should change evidence-to-boundary representation (e.g. multi-rival boundary representation), not selector/B/M/certificate. Atom-to-action value is only next if Top-M/selected critical recall improves but teacher action/regret does not.'
    out={'decision':decision,'next_action':next_action,'apwcca':a,'apwrcca':b,'value_probe':c}
    text=json.dumps(out,indent=2,sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(text)
    print(text)
    return 0

if __name__=='__main__': raise SystemExit(main())
