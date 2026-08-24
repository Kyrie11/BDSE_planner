#!/usr/bin/env bash
set -euo pipefail
# Historical compatibility entrypoint restored because the uploaded V64.30.3
# archive omitted this root file while retaining a repository regression test
# that requires it.  The original legacy pipeline is not reconstructed from
# memory: invoking this compatibility entrypoint fails closed and points to the
# retained, auditable V64.2 resume instructions instead of silently executing a
# different experiment.
cat >&2 <<'EOF'
V64_SAQA_BCC_NEXT_COMMANDS.sh is a historical compatibility entrypoint.
The exact legacy script bytes are absent from the uploaded archive, so this
restoration intentionally does not synthesize the old experiment. For the
historical V64.2 gate-fix workflow, follow:
  review_artifacts/NEXT_COMMANDS_V64_2_GATEFIX_HCBE.txt
For the current paper-mechanism experiment, run:
  bash RUN_V64_3_31_EAF_ICER_SCIR_SCREEN_2GPU.sh
EOF
exit 2
