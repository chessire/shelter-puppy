"""M3 모션 곡선 메트릭 — 동/정 분리 정확도 + 손떨림 오검출율.

GT/pred 모두 타임라인을 타일링하는 Segment(label=moving|static) 리스트.
시간(초) 가중으로 라벨 일치도를 잰다.

설계서 측정 지표:
  - 동/정 구간 분리 정확도        → accuracy
  - 손떨림이 동작으로 오검출되는 비율 → false_motion_rate (GT static 인데 pred moving)
  - (보조) 놓친 동작 비율           → missed_motion_rate (GT moving 인데 pred static)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..schemas import Segment


@dataclass
class M3Metrics:
    total_time: float
    accuracy: float
    false_motion_rate: float
    missed_motion_rate: float

    def as_dict(self) -> dict:
        return {
            "total_time": round(self.total_time, 4),
            "accuracy": round(self.accuracy, 4),
            "false_motion_rate": round(self.false_motion_rate, 4),
            "missed_motion_rate": round(self.missed_motion_rate, 4),
        }


def _label_at(segments: list[Segment], t: float) -> Optional[str]:
    for s in segments:
        if s.start_t <= t < s.end_t:
            return s.label
    return None


def _confusion_over_time(
    gt: list[Segment], pred: list[Segment]
) -> dict[tuple[str, str], float]:
    """겹치는 시간 구간을 잘게 쪼개 (gt_label, pred_label)별 초를 누적."""
    bounds = set()
    for s in gt + pred:
        bounds.add(s.start_t)
        bounds.add(s.end_t)
    pts = sorted(bounds)
    conf: dict[tuple[str, str], float] = {}
    for t0, t1 in zip(pts, pts[1:]):
        if t1 <= t0:
            continue
        mid = (t0 + t1) / 2
        gl = _label_at(gt, mid)
        pl = _label_at(pred, mid)
        if gl is None or pl is None:
            continue  # 양쪽 다 정의된 구간만 채점
        conf[(gl, pl)] = conf.get((gl, pl), 0.0) + (t1 - t0)
    return conf


def evaluate_m3(gt: list[Segment], pred: list[Segment]) -> M3Metrics:
    conf = _confusion_over_time(gt, pred)
    total = sum(conf.values())
    correct = sum(v for (g, p), v in conf.items() if g == p)
    gt_static = sum(v for (g, _), v in conf.items() if g == "static")
    gt_moving = sum(v for (g, _), v in conf.items() if g == "moving")
    static_as_moving = conf.get(("static", "moving"), 0.0)
    moving_as_static = conf.get(("moving", "static"), 0.0)
    return M3Metrics(
        total_time=total,
        accuracy=(correct / total) if total > 0 else 1.0,
        false_motion_rate=(static_as_moving / gt_static) if gt_static > 0 else 0.0,
        missed_motion_rate=(moving_as_static / gt_moving) if gt_moving > 0 else 0.0,
    )
