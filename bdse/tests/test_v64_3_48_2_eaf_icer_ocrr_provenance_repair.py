from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "RUN_V64_3_48_2_EAF_ICER_OCRR_SCREEN_2GPU.sh"
LEDGER = ROOT / "bdse/configs/v64_3_48_consumed_fresh1000_tokens.txt"
SCIENCE_LOCK = ROOT / "V64_3_48_OCRR_SCIENCE_LOCK.sha256"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_consumed_v48_fresh_ledger_is_frozen_and_unique() -> None:
    tokens = [x.strip() for x in LEDGER.read_text().splitlines() if x.strip()]
    assert len(tokens) == 1000
    assert len(set(tokens)) == 1000
    assert _sha(LEDGER) == "c09becde1bca5a8d2d6fe2166980c4bb1a0923b02fdd69d598c5e1e3d4fc6d2e"


def test_ocrr_science_lock_preserves_preregistered_core() -> None:
    expected = {}
    for line in SCIENCE_LOCK.read_text().splitlines():
        h, rel = line.split(maxsplit=1)
        expected[rel.strip()] = h
    assert expected == {
        "bdse/planner/tournament.py": "291b3b77202974b74fe42431ee7954de8c401d927591c19a12a5837f18374044",
        "bdse/planner/operator_conditioned_risk_retention.py": "c35425114f8438d2c644da7aea7fae57916f17390ef49f93dfe614e1ae7179c3",
        "bdse/tools/fit_v64_3_48_eaf_icer_ocrr.py": "d33acbe4c6d41c86e1bc856f9c6dae2cf34b409205f11cd94bb04e9517981d54",
        "bdse/tools/check_v64_3_48_eaf_icer_ocrr_split.py": "ac3edeaea53965e17dc30db6fa845a3a2dbefdf1121e3db71ba3f807992ebf69",
        "bdse/tools/check_v64_3_48_eaf_icer_ocrr_screen.py": "1d5b46f80e18548f3f19622ae81acbeac4170f3f29112f8c4e8030eacc44b797",
    }
    for rel, h in expected.items():
        assert _sha(ROOT / rel) == h


def test_repair_launcher_hard_gates_source_and_old_fresh_consumption() -> None:
    text = LAUNCHER.read_text()
    assert "sha256sum -c V64_3_48_2_SOURCE_MANIFEST.sha256" in text
    assert "sha256sum -c V64_3_48_OCRR_SCIENCE_LOCK.sha256" in text
    assert "v64_3_48_consumed_fresh1000_tokens.txt" in text
    assert 'cat "$DESIGN_EXCLUDE_TOKENS" "$FROZEN_TRAIN_TOKENS" "$V48_CONSUMED_FRESH_TOKENS"' in text
    assert "v64.3.48.2-eaf-icer-ocrr-double-fresh-v1" in text
    assert "v64_3_48_2_split_A_screen.json" in text
    assert "v64_3_48_2_split_B_screen.json" in text
