"""M2~M5 메트릭 검증 — 손계산 답으로 대조."""

import math

from pipeline.harness.schemas import (
    ActionSegment,
    InterventionRecord,
    MatchEntry,
    ReIDResult,
    Segment,
)
from pipeline.harness.metrics.m2_reid import evaluate_m2
from pipeline.harness.metrics.m3_motion import evaluate_m3
from pipeline.harness.metrics.m4_action import evaluate_m4
from pipeline.harness.metrics.m5_match import evaluate_m5


# --------------------------------------------------------------------------- #
# M2 re-ID
# --------------------------------------------------------------------------- #
def test_m2_perfect_no_intervention():
    # track 1,2 = 개A / track 3 = 개B. 완벽 군집 + 개입 0.
    gt = {1: 100, 2: 100, 3: 200}
    pred = ReIDResult(mapping={1: 7, 2: 7, 3: 9})  # 라벨 달라도 군집 같으면 OK
    m = evaluate_m2(gt, pred)
    assert m.pair_precision == 1.0 and m.pair_recall == 1.0
    assert m.interventions == 0 and m.auto_relink_rate == 1.0
    assert m.num_global_gt == 2 and m.num_global_pred == 2


def test_m2_false_merge():
    # 두 다른 강아지(global 100,200)를 pred 가 한 ID(7)로 잘못 합침 → false merge.
    gt = {1: 100, 2: 200}
    pred = ReIDResult(mapping={1: 7, 2: 7})
    m = evaluate_m2(gt, pred)
    # 쌍 1개: same_gt=False, same_pred=True → fp=1, tp=0 → precision=0
    assert m.pair_precision == 0.0
    assert m.pair_recall == 1.0   # same_gt 쌍이 없으므로 분모0 → 1.0 관례


def test_m2_intervention_rate():
    # track 3개 중 1개는 사람 1탭으로 해결 → auto_relink_rate = 2/3.
    gt = {1: 100, 2: 100, 3: 100}
    pred = ReIDResult(
        mapping={1: 1, 2: 1, 3: 1},
        interventions=[InterventionRecord(track_id=3, t=5.0, reason="reappear")],
    )
    m = evaluate_m2(gt, pred)
    assert m.interventions == 1
    assert math.isclose(m.auto_relink_rate, 2 / 3, rel_tol=1e-9)
    assert m.pair_precision == 1.0 and m.pair_recall == 1.0


# --------------------------------------------------------------------------- #
# M3 모션
# --------------------------------------------------------------------------- #
def test_m3_perfect():
    gt = [Segment(0, 5, "static"), Segment(5, 10, "moving")]
    pred = [Segment(0, 5, "static"), Segment(5, 10, "moving")]
    m = evaluate_m3(gt, pred)
    assert m.accuracy == 1.0
    assert m.false_motion_rate == 0.0 and m.missed_motion_rate == 0.0


def test_m3_camera_shake_false_motion():
    # GT: 0~10 전부 static. pred 가 2~4 를 moving 으로 오검출(손떨림).
    gt = [Segment(0, 10, "static")]
    pred = [Segment(0, 2, "static"), Segment(2, 4, "moving"), Segment(4, 10, "static")]
    m = evaluate_m3(gt, pred)
    # 10초 중 2초 틀림 → accuracy 0.8, false_motion = 2/10 = 0.2
    assert math.isclose(m.accuracy, 0.8, rel_tol=1e-9)
    assert math.isclose(m.false_motion_rate, 0.2, rel_tol=1e-9)
    assert m.missed_motion_rate == 0.0


# --------------------------------------------------------------------------- #
# M4 동작
# --------------------------------------------------------------------------- #
def test_m4_uncertain_excluded_from_accuracy():
    # GT: 0~10 dynamic. pred: 0~6 dynamic(맞음), 6~8 static(틀림), 8~10 uncertain.
    gt = [ActionSegment(0, 10, "dynamic")]
    pred = [
        ActionSegment(0, 6, "dynamic"),
        ActionSegment(6, 8, "static"),
        ActionSegment(8, 10, "dynamic", uncertain=True),
    ]
    m = evaluate_m4(gt, pred)
    # committed = 8초(0~8), 그중 6초 정답 → group_accuracy = 6/8 = 0.75
    assert math.isclose(m.committed_time, 8.0, rel_tol=1e-9)
    assert math.isclose(m.group_accuracy, 0.75, rel_tol=1e-9)
    # uncertain = 2초 / 10초 = 0.2
    assert math.isclose(m.uncertain_rate, 0.2, rel_tol=1e-9)


def test_m4_low_motion_recall():
    # 저모션 동작 '웅크림' 구간(0~4)을 pred 가 맞춤, '팬팅'(4~6)은 static 으로 놓침.
    gt = [
        ActionSegment(0, 4, "dynamic", action="crouch"),
        ActionSegment(4, 6, "dynamic", action="panting"),
    ]
    pred = [
        ActionSegment(0, 4, "dynamic"),
        ActionSegment(4, 6, "static"),
    ]
    m = evaluate_m4(gt, pred, low_motion_actions={"crouch", "panting"})
    # 저모션 4초(crouch) 표집 성공 / 6초 전체 → 4/6
    assert math.isclose(m.low_motion_recall, 4 / 6, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# M5 매칭
# --------------------------------------------------------------------------- #
def test_m5_perfect():
    gt = [MatchEntry(0, "v1", 0, 3), MatchEntry(1, "v2", 3, 6)]
    pred = [MatchEntry(0, "v1", 0, 3), MatchEntry(1, "v2", 3, 6)]
    m = evaluate_m5(gt, pred)
    assert m.match_accuracy == 1.0 and m.uncertain_rate == 0.0


def test_m5_wrong_source_and_uncertain():
    # 구절0: 정답(v1). 구절1: pred 가 엉뚱한 source(v9). 구절2: uncertain.
    gt = [
        MatchEntry(0, "v1", 0, 3),
        MatchEntry(1, "v2", 3, 6),
        MatchEntry(2, "v3", 6, 9),
    ]
    pred = [
        MatchEntry(0, "v1", 0, 3),
        MatchEntry(1, "v9", 3, 6),
        MatchEntry(2, "v3", 6, 9, uncertain=True),
    ]
    m = evaluate_m5(gt, pred)
    # committed = 구절0,1 (2개), 정답 1개 → accuracy 0.5
    assert m.committed == 2
    assert math.isclose(m.match_accuracy, 0.5, rel_tol=1e-9)
    # uncertain = 1/3
    assert math.isclose(m.uncertain_rate, 1 / 3, rel_tol=1e-9)
