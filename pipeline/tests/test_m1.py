"""M1 추적 메트릭 검증 — 손계산 답이 정해진 합성 시나리오로 대조.

각 픽스처의 기대값은 주석에 유도 과정을 적어 둔다(메트릭 자체를 채점).
"""

import math

from pipeline.harness.schemas import BBox, Detection, FrameDetections
from pipeline.harness.metrics.m1_tracking import evaluate_m1


def _frame(idx, dets):
    return FrameDetections(idx, float(idx) * 0.25, dets)


def _dog(tid, x):
    return Detection(tid, BBox(x, 0, 10, 10), 1.0, "dog")


# 공통 GT: id=1 강아지가 x로 이동, 3프레임.
def _gt_single():
    return [_frame(0, [_dog(1, 0)]), _frame(1, [_dog(1, 1)]), _frame(2, [_dog(1, 2)])]


def test_perfect():
    gt = _gt_single()
    pred = _gt_single()  # 동일
    m = evaluate_m1(gt, pred)
    assert (m.fp, m.fn, m.idsw, m.fragmentation) == (0, 0, 0, 0)
    assert m.mota == 1.0 and m.idf1 == 1.0 and m.miss_rate == 0.0
    assert m.num_gt == 3 and m.num_pred == 3


def test_one_miss():
    # pred 가 frame1 검출을 놓침 → FN=1, fragmentation=1(끊겼다 재개), miss_rate=1/3
    gt = _gt_single()
    pred = [_frame(0, [_dog(1, 0)]), _frame(1, []), _frame(2, [_dog(1, 2)])]
    m = evaluate_m1(gt, pred)
    assert (m.fp, m.fn, m.idsw) == (0, 1, 0)
    assert m.fragmentation == 1
    assert math.isclose(m.mota, 1 - 1 / 3, rel_tol=1e-9)      # 0.6667
    assert math.isclose(m.miss_rate, 1 / 3, rel_tol=1e-9)
    assert math.isclose(m.idf1, 0.8, rel_tol=1e-9)            # 2·2/(4+0+1)


def test_one_false_positive():
    # 멀리 떨어진 유령 박스 1개 → FP=1 (예: 고양이 오검출 흉내)
    gt = _gt_single()
    pred = [
        _frame(0, [_dog(1, 0)]),
        _frame(1, [_dog(1, 1), Detection(99, BBox(100, 100, 10, 10), 0.9, "dog")]),
        _frame(2, [_dog(1, 2)]),
    ]
    m = evaluate_m1(gt, pred)
    assert (m.fp, m.fn, m.idsw) == (1, 0, 0)
    assert math.isclose(m.mota, 1 - 1 / 3, rel_tol=1e-9)
    assert math.isclose(m.idf1, 6 / 7, rel_tol=1e-9)          # 2·3/(6+1+0)


def test_id_switch():
    # pred 가 같은 개를 frame0-1 은 id10, frame2 는 id20 으로 → IDSW=1
    gt = _gt_single()
    pred = [
        _frame(0, [_dog(10, 0)]),
        _frame(1, [_dog(10, 1)]),
        _frame(2, [_dog(20, 2)]),
    ]
    m = evaluate_m1(gt, pred)
    assert (m.fp, m.fn, m.idsw) == (0, 0, 1)
    assert m.fragmentation == 0          # 매 프레임 매칭은 유지됨(끊김 아님)
    assert math.isclose(m.mota, 1 - 1 / 3, rel_tol=1e-9)
    assert math.isclose(m.idf1, 4 / 6, rel_tol=1e-9)          # 2·2/(4+1+1)


def test_two_dogs_perfect():
    # V3(개 2마리) 흉내: 두 트랙 모두 정확 추적 → 완벽
    gt = [
        _frame(0, [_dog(1, 0), _dog(2, 50)]),
        _frame(1, [_dog(1, 1), _dog(2, 51)]),
    ]
    # pred 는 track_id 가 달라도(7,8) 일관되기만 하면 완벽
    pred = [
        _frame(0, [_dog(7, 0), _dog(8, 50)]),
        _frame(1, [_dog(7, 1), _dog(8, 51)]),
    ]
    m = evaluate_m1(gt, pred)
    assert (m.fp, m.fn, m.idsw, m.fragmentation) == (0, 0, 0, 0)
    assert m.mota == 1.0 and m.idf1 == 1.0
    assert m.num_gt == 4
