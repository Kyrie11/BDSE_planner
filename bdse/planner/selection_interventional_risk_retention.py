from __future__ import annotations

"""V64.3.49 Selection-Interventional Invariant Retention (SIIR).

V48 showed that conditioning a selected-risk law on observed challenger
multiplicity can fit the frozen TRAIN population while failing to transport to
fresh selection regimes.  SIIR removes multiplicity from the deployment state
and changes *how the risk law is identified*: during TRAIN only, a deterministic
label-free intervention randomizes which prefix of a hash-permuted admissible
candidate set is exposed to the already frozen RSMR selector.

Runtime remains strictly monotone: the full candidate bank and RSMR winner are
unchanged, and SIIR can only retain that same winner or return to the incumbent.
The runtime state contains only consequence coordinates already validated before
V48:

    [QUALITY value,
     prospective-response increment,
     ego-reference increment].

No candidate-count/top-K value is consumed at deployment.
"""

from hashlib import sha256
from typing import Any, Sequence
import math
import numpy as np

SIIR_STATE_NAMES = [
    "quality_value",
    "prospective_response_increment",
    "ego_reference_increment",
]


def consequence_state(
    quality_value: float,
    plan_control_value: float,
    ego_reference_value: float,
) -> np.ndarray:
    z = np.asarray(
        [
            float(quality_value),
            float(plan_control_value) - float(quality_value),
            float(ego_reference_value) - float(plan_control_value),
        ],
        dtype=np.float64,
    )
    if z.shape != (3,) or not np.all(np.isfinite(z)):
        raise ValueError("V49 SIIR consequence state is non-finite")
    return z


def intervention_prefix(
    scenario_token: str,
    action_ids: Sequence[int],
    *,
    seed: str,
) -> tuple[list[int], int]:
    """Return a label-free hash permutation and one uniformly indexed prefix.

    This is a TRAIN-only intervention on the *selection event*.  It is not a
    runtime candidate-count sweep: every scene uses exactly one deterministic
    prefix chosen before labels/results are inspected, and deployment always
    returns to the original full candidate set.
    """
    acts = [int(a) for a in action_ids]
    if not acts:
        return [], 0
    if len(set(acts)) != len(acts):
        raise ValueError("V49 SIIR intervention requires unique action ids")
    ordered = sorted(
        acts,
        key=lambda a: (
            sha256(f"{seed}::perm::{scenario_token}::{a}".encode()).digest(),
            int(a),
        ),
    )
    h = sha256(f"{seed}::prefix::{scenario_token}".encode()).digest()
    m = 1 + (int.from_bytes(h[:8], "big") % len(ordered))
    return ordered, int(m)


def select_interventional_winner(
    scenario_token: str,
    action_ids: Sequence[int],
    scores: Sequence[float],
    supports: Sequence[float],
    margins: Sequence[float],
    utility_priors: Sequence[int],
    *,
    seed: str,
) -> tuple[int | None, int]:
    """Apply frozen RSMR tie-breaking after a label-free prefix intervention.

    Inputs intentionally contain no teacher outcome/value label.  The function
    only changes which candidates are exposed during TRAIN identification.
    """
    acts = [int(a) for a in action_ids]
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    sup = np.asarray(supports, dtype=np.float64).reshape(-1)
    mar = np.asarray(margins, dtype=np.float64).reshape(-1)
    pri = np.asarray(utility_priors, dtype=np.int64).reshape(-1)
    n = len(acts)
    if any(x.size != n for x in (score, sup, mar, pri)):
        raise ValueError("V49 SIIR intervention arrays have inconsistent length")
    if n == 0:
        return None, 0
    order, m = intervention_prefix(scenario_token, acts, seed=seed)
    exposed = set(order[:m])
    cand = [
        j for j, a in enumerate(acts)
        if a in exposed and math.isfinite(float(score[j])) and float(score[j]) > 0.0
    ]
    if not cand:
        return None, int(m)
    j = sorted(
        cand,
        key=lambda q: (
            -float(score[q]),
            -float(sup[q]),
            -float(mar[q]),
            -int(pri[q]),
            int(acts[q]),
        ),
    )[0]
    return int(j), int(m)


def _risk_score(z: np.ndarray, model: dict[str, Any]) -> float:
    names = [str(x) for x in model.get("feature_names", [])]
    if names != SIIR_STATE_NAMES:
        raise ValueError(f"V49 SIIR risk feature schema mismatch: {names}")
    mean = np.asarray(model.get("feature_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(model.get("feature_std", []), dtype=np.float64).reshape(-1)
    w = np.asarray(model.get("weights", []), dtype=np.float64).reshape(-1)
    if (
        mean.size != 3
        or std.size != 3
        or w.size != 3
        or np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(std))
        or np.any(std <= 0.0)
        or np.any(~np.isfinite(w))
    ):
        raise ValueError("V49 SIIR risk parameters are invalid")
    if abs(float(model.get("bias", 0.0))) > 1.0e-12:
        raise ValueError("V49 SIIR pairwise sign-risk ranker requires zero bias")
    raw = float(((z - mean) / np.maximum(std, 1.0e-6)) @ w)
    pmean = float(model.get("fit_positive_score_mean", float("nan")))
    pstd = float(model.get("fit_positive_score_std", float("nan")))
    if not math.isfinite(pmean) or not math.isfinite(pstd) or pstd <= 0.0:
        raise ValueError("V49 SIIR positive-score normalization is invalid")
    return float((raw - pmean) / max(pstd, 1.0e-6))


def runtime_risk(z: np.ndarray, cfg: dict[str, Any]) -> float:
    zz = np.asarray(z, dtype=np.float64).reshape(-1)
    if zz.size != 3 or not np.all(np.isfinite(zz)):
        raise ValueError("V49 SIIR runtime state must be finite 3-D")
    if str(cfg.get("aggregation", "sign_only")).strip().lower() != "sign_only":
        raise ValueError("V49 SIIR only permits sign_only risk")
    return _risk_score(zz, cfg.get("components", {}).get("sign_risk", {}))


def runtime_certificate(z: np.ndarray, cfg: dict[str, Any]) -> tuple[float, float]:
    risk = runtime_risk(z, cfg)
    tau = float(cfg.get("retention_threshold", float("nan")))
    if not math.isfinite(tau):
        raise ValueError("V49 SIIR retention threshold is invalid")
    return float(tau - risk), float(risk)
