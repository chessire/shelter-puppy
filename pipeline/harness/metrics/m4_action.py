"""M4 동작 판별 메트릭 — 군(동작/정지) 정확도 + uncertain율 + 저모션 표집률.

핵심 설계: '군 단위 판정'(dynamic vs static)으로 채점하고, uncertain 은
일반 클립으로 폴백되므로 정확도는 *committed(uncertain 아닌)* 구간에서만 잰다.

설계서 측정 지표:
  - 군 단위 정확도            → group_accuracy (committed 시간 가중)
  - uncertain율               → uncertain_rate
  - 저모션(웅크림·팬팅) 표집률 → low_motion_recall (저모션 동작을 안 놓쳤나)
  - 화질 하한선(px 스윕)       → 별도 실험 러너 소관(단일 eval 지표 아님)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..schemas import ActionSegment


@dataclass
class M4Metrics:
    total_time: float
    committed_time: float
    uncertain_rate: float
    group_accuracy: float          # committed 구간 중 군 일치 비율
    low_motion_recall: Optional[float]  # 저모션 동작 표집률(해당 GT 없으면 None)

    def as_dict(self) -> dict:
        return {
            "total_time": round(self.total_time, 4),
            "committed_time": round(self.committed_time, 4),
            "uncertain_rate": round(self.uncertain_rate, 4),
            "group_accuracy": round(self.group_accuracy, 4),
            "low_motion_recall": (
                None if self.low_motion_recall is None
                else round(self.low_motion_recall, 4)
            ),
        }


def _group_at(segs: list[ActionSegment], t: float):
    """반환 (group, uncertain) 또는 (None, None)."""
    for s in segs:
        if s.start_t <= t < s.end_t:
            return s.group, s.uncertain
    return None, None


def evaluate_m4(
    gt: list[ActionSegment],
    pred: list[ActionSegment],
    low_motion_actions: Optional[set[str]] = None,
) -> M4Metrics:
    bounds = set()
    for s in list(gt) + list(pred):
        bounds.add(s.start_t)
        bounds.add(s.end_t)
    pts = sorted(bounds)

    total = committed = uncertain_t = correct = 0.0
    for t0, t1 in zip(pts, pts[1:]):
        if t1 <= t0:
            continue
        mid = (t0 + t1) / 2
        gg, _ = _group_at(gt, mid)
        pg, punc = _group_at(pred, mid)
        if gg is None or pg is None:
            continue
        dur = t1 - t0
        total += dur
        if punc:
            uncertain_t += dur
        else:
            committed += dur
            if pg == gg:
                correct += dur

    # 저모션 표집률: GT 의 저모션 동작 구간에서 pred 가 (committed) 군을 맞춘 비율.
    low_recall: Optional[float] = None
    if low_motion_actions:
        lm_total = lm_hit = 0.0
        for s in gt:
            if s.action in low_motion_actions:
                lm_total += s.dur
                # 구간 중점에서 pred 가 committed 이고 군 일치면 표집 성공으로 본다.
                pg, punc = _group_at(pred, (s.start_t + s.end_t) / 2)
                if pg == s.group and not punc:
                    lm_hit += s.dur
        if lm_total > 0:
            low_recall = lm_hit / lm_total

    return M4Metrics(
        total_time=total,
        committed_time=committed,
        uncertain_rate=(uncertain_t / total) if total > 0 else 0.0,
        group_accuracy=(correct / committed) if committed > 0 else 1.0,
        low_motion_recall=low_recall,
    )
