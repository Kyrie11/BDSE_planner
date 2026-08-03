#!/usr/bin/env bash
set -euo pipefail
CL_LIMIT=50 CL_OUT="${CL50_OUT:-outputs/v60_external_compare/${CL_CHALLENGE:-closed_loop_nonreactive_agents}_cl50}" \
  bash RUN_V60_EXTERNAL_CLOSED_LOOP_SUITE_2GPU.sh
