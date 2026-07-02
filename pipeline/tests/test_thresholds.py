"""thresholds.py 검증 — min/max 규칙과 종합 판정."""

from pipeline.harness.thresholds import check_stage, overall


THRESHOLDS = {
    "m2": {
        "pair_precision": {"min": 0.95},
        "interventions": {"max": 3},
        "auto_relink_rate": {"min": None},  # 미설정
    }
}


def test_min_rule_pass_fail():
    checks = check_stage("m2", {"pair_precision": 0.96}, THRESHOLDS)
    assert next(c for c in checks if c.metric == "pair_precision").status == "PASS"
    checks = check_stage("m2", {"pair_precision": 0.90}, THRESHOLDS)
    assert next(c for c in checks if c.metric == "pair_precision").status == "FAIL"


def test_max_rule_pass_fail():
    checks = check_stage("m2", {"interventions": 2}, THRESHOLDS)
    assert next(c for c in checks if c.metric == "interventions").status == "PASS"
    checks = check_stage("m2", {"interventions": 5}, THRESHOLDS)
    assert next(c for c in checks if c.metric == "interventions").status == "FAIL"


def test_null_threshold_is_unset():
    checks = check_stage("m2", {"auto_relink_rate": 0.5}, THRESHOLDS)
    assert next(c for c in checks if c.metric == "auto_relink_rate").status == "미설정"


def test_overall_fail_dominates():
    checks = check_stage(
        "m2", {"pair_precision": 0.90, "interventions": 1}, THRESHOLDS
    )
    assert overall(checks) == "FAIL"


def test_overall_unset_when_no_active_rule():
    checks = check_stage("m2", {"auto_relink_rate": 0.5}, THRESHOLDS)
    assert overall(checks) == "미설정"
