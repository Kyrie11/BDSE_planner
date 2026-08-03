#!/usr/bin/env bash
set -euo pipefail
CL_LIMIT=20 CL_OUT="${CL20_OUT:-outputs/v60_external_compare/${CL_CHALLENGE:-closed_loop_nonreactive_agents}_cl20}" \
  bash RUN_V60_EXTERNAL_CLOSED_LOOP_SUITE_2GPU.sh
