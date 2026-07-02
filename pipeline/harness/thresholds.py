"""임계값 레지스트리 로더 + 판정.

thresholds.yaml 의 {min/max: x} 규칙으로 메트릭 dict 를 PASS/FAIL/미설정 판정.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_DEFAULT = Path(__file__).with_name("thresholds.yaml")


@dataclass
class Check:
    metric: str
    value: float
    rule: str            # "min" | "max" | "—"
    bound: Optional[float]
    status: str          # "PASS" | "FAIL" | "미설정"


def load_thresholds(path: str | Path | None = None) -> dict:
    p = Path(path) if path else _DEFAULT
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_stage(stage: str, metrics: dict, thresholds: dict) -> list[Check]:
    """한 단계의 메트릭 dict 를 임계값과 대조."""
    spec = thresholds.get(stage, {}) or {}
    checks: list[Check] = []
    for metric, value in metrics.items():
        if not isinstance(value, (int, float)) or value is None:
            continue
        rule_spec = spec.get(metric)
        if not rule_spec:
            checks.append(Check(metric, value, "—", None, "미설정"))
            continue
        if "min" in rule_spec and rule_spec["min"] is not None:
            bound = float(rule_spec["min"])
            status = "PASS" if value >= bound else "FAIL"
            checks.append(Check(metric, value, "min", bound, status))
        elif "max" in rule_spec and rule_spec["max"] is not None:
            bound = float(rule_spec["max"])
            status = "PASS" if value <= bound else "FAIL"
            checks.append(Check(metric, value, "max", bound, status))
        else:
            checks.append(Check(metric, value, "—", None, "미설정"))
    return checks


def overall(checks: list[Check]) -> str:
    """단계 종합 판정. FAIL 하나라도 있으면 FAIL, 판정 가능한 게 없으면 미설정."""
    statuses = {c.status for c in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "PASS" in statuses:
        return "PASS"
    return "미설정"
