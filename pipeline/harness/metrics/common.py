"""메트릭 공통 유틸 — IoU(박스/시간구간), 헝가리안 매칭.

여기 함수들이 틀리면 모든 단계 채점이 틀어진다. 단위테스트로 못박는다.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..schemas import BBox, Segment


# --------------------------------------------------------------------------- #
# 박스 IoU
# --------------------------------------------------------------------------- #
def iou(a: BBox, b: BBox) -> float:
    """두 [x,y,w,h] 박스의 IoU. 겹침 없으면 0."""
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def iou_matrix(gts: list[BBox], preds: list[BBox]) -> np.ndarray:
    """gts × preds IoU 행렬."""
    m = np.zeros((len(gts), len(preds)), dtype=float)
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            m[i, j] = iou(g, p)
    return m


# --------------------------------------------------------------------------- #
# 헝가리안 매칭 (IoU 최대화)
# --------------------------------------------------------------------------- #
def match_by_iou(
    gts: list[BBox], preds: list[BBox], thr: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """IoU>=thr 제약에서 매칭을 최대화.

    반환: (matches[(gt_idx, pred_idx)], unmatched_gt_idx, unmatched_pred_idx)
    """
    if not gts or not preds:
        return [], list(range(len(gts))), list(range(len(preds)))

    m = iou_matrix(gts, preds)
    # linear_sum_assignment 는 비용 최소화 → IoU를 음수 비용으로.
    cost = -m
    rows, cols = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    matched_g, matched_p = set(), set()
    for r, c in zip(rows, cols):
        if m[r, c] >= thr:
            matches.append((int(r), int(c)))
            matched_g.add(int(r))
            matched_p.add(int(c))
    un_g = [i for i in range(len(gts)) if i not in matched_g]
    un_p = [j for j in range(len(preds)) if j not in matched_p]
    return matches, un_g, un_p


# --------------------------------------------------------------------------- #
# 시간 구간 겹침
# --------------------------------------------------------------------------- #
def interval_overlap(a: Segment, b: Segment) -> float:
    """두 구간의 겹치는 시간(초). 없으면 0."""
    return max(0.0, min(a.end_t, b.end_t) - max(a.start_t, b.start_t))


def interval_iou(a: Segment, b: Segment) -> float:
    """두 구간의 시간 IoU."""
    inter = interval_overlap(a, b)
    union = a.dur + b.dur - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def total_duration(segments: list[Segment]) -> float:
    return sum(s.dur for s in segments)
