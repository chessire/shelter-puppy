"""common.py 검증 — IoU/구간겹침/헝가리안. 손계산 답과 대조."""

import math

from pipeline.harness.schemas import BBox, Segment
from pipeline.harness.metrics.common import (
    iou,
    match_by_iou,
    interval_overlap,
    interval_iou,
)


def test_iou_identical():
    b = BBox(0, 0, 10, 10)
    assert iou(b, b) == 1.0


def test_iou_no_overlap():
    assert iou(BBox(0, 0, 10, 10), BBox(100, 100, 10, 10)) == 0.0


def test_iou_half_overlap():
    # 10x10 두 박스가 x로 5겹침 → inter=50, union=200-50=150 → 1/3
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 0, 10, 10)
    assert math.isclose(iou(a, b), 50 / 150, rel_tol=1e-9)


def test_match_by_iou_two_objects():
    gts = [BBox(0, 0, 10, 10), BBox(50, 0, 10, 10)]
    preds = [BBox(49, 0, 10, 10), BBox(1, 0, 10, 10)]  # 순서 뒤섞임
    matches, ug, up = match_by_iou(gts, preds, thr=0.5)
    # gt0↔pred1, gt1↔pred0 으로 올바르게 교차 매칭돼야 한다.
    assert set(matches) == {(0, 1), (1, 0)}
    assert ug == [] and up == []


def test_match_by_iou_below_threshold():
    gts = [BBox(0, 0, 10, 10)]
    preds = [BBox(8, 0, 10, 10)]  # IoU 작음
    matches, ug, up = match_by_iou(gts, preds, thr=0.5)
    assert matches == []
    assert ug == [0] and up == [0]


def test_interval_overlap_and_iou():
    a = Segment(0, 10, "moving")
    b = Segment(5, 15, "moving")
    assert interval_overlap(a, b) == 5.0
    # inter=5, union=20-5=15 → 1/3
    assert math.isclose(interval_iou(a, b), 5 / 15, rel_tol=1e-9)
