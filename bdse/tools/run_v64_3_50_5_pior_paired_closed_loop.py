"""Engineering-safe launcher shim for the frozen V64.3.50 PIOR collector.

The scientific V50 collector remains byte-identical.  This shim rewrites only
its nuPlan child entrypoint so nuPlan metric computation is serialized inside
each arm process, eliminating stateful metric-engine races under the existing
threaded simulation worker.
"""

from __future__ import annotations

import hashlib
import os
import subprocess as _subprocess
from pathlib import Path
from typing import Any

from bdse.tools import run_v64_3_50_pior_paired_closed_loop as _base

_EXPECTED_BASE_SHA256 = "7c5472442e5a76ee6cbb6ef3189e086e4e800e8337deb14cacb26488b9050d53"
_OFFICIAL_MODULE = "nuplan.planning.script.run_simulation"
_SAFE_MODULE = "bdse.tools.nuplan_metric_safe_run_simulation"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_frozen_base_runner() -> None:
    p = Path(_base.__file__).resolve()
    got = _sha256(p)
    if got != _EXPECTED_BASE_SHA256:
        raise RuntimeError(
            "STOP V64.3.50.5: frozen V50 paired collector changed: "
            f"{p} sha256={got} expected={_EXPECTED_BASE_SHA256}"
        )


class _SubprocessProxy:
    """Proxy stdlib subprocess; rewrite only the PIOR evaluate_closed_loop child."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_subprocess, name)

    @staticmethod
    def Popen(cmd: Any, *args: Any, **kwargs: Any):  # noqa: N802 - mirror subprocess API
        if isinstance(cmd, (list, tuple)) and "bdse.experiments.evaluate_closed_loop" in cmd:
            rewritten = list(cmd)
            try:
                i = rewritten.index("--nuplan-module")
            except ValueError as exc:
                raise RuntimeError("STOP V64.3.50.5: PIOR child command lacks --nuplan-module") from exc
            if i + 1 >= len(rewritten) or rewritten[i + 1] != _OFFICIAL_MODULE:
                raise RuntimeError(
                    "STOP V64.3.50.5: unexpected frozen nuPlan entrypoint before metric-safety rewrite: "
                    f"{rewritten[i + 1] if i + 1 < len(rewritten) else '<missing>'}"
                )
            rewritten[i + 1] = _SAFE_MODULE
            env = dict(kwargs.get("env") or os.environ)
            env["BDSE_PIOR_METRIC_ENGINE_SERIALIZATION"] = "1"
            kwargs["env"] = env
            print(
                "[BDSE-PIOR-METRIC-SAFE-SPAWN] rewrote frozen nuPlan child to "
                f"{_SAFE_MODULE}; metric callback serialized, simulation workers unchanged",
                flush=True,
            )
            cmd = rewritten
        return _subprocess.Popen(cmd, *args, **kwargs)


def main() -> None:
    _assert_frozen_base_runner()
    _base.subprocess = _SubprocessProxy()  # engineering-only transport shim
    _base.main()


if __name__ == "__main__":
    main()
