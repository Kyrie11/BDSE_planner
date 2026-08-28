#!/usr/bin/env bash
set -euo pipefail
cat >&2 <<'MSG'
STOP: this is a fail-closed compatibility entrypoint for the historical pre-V64.3 pipeline.
The exact legacy launcher bytes are not present in the supplied V64.3.48 archive and are not reconstructed here.
Use the preserved review artifact `review_artifacts/NEXT_COMMANDS_V64_2_GATEFIX_HCBE.txt` only when intentionally reproducing that historical experiment.
This compatibility entrypoint deliberately refuses to invent or execute legacy experiment semantics.
MSG
exit 2
