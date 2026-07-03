"""M2 re-ID 메트릭 ★게이트 — 군집 정확도 + 자동재연결률 + 영상당 개입수.

track_id 들을 global_dog_id 로 묶는 '군집' 문제로 본다. GT 는 정답 군집,
pred 는 시스템 군집. 핵심 위험은 'false merge'(다른 두 강아지를 한 ID로) 이므로
쌍(pair) 단위 precision/recall 로 잡는다.

설계서 측정 지표:
  - 자동 재연결률          → auto_relink_rate (사람 1탭 없이 묶인 track 비율)
  - 영상당 개입 N회         → interventions
  - (correctness) false merge → pair_precision (낮을수록 false merge 많음)
  - (correctness) 놓친 연결   → pair_recall
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from ..schemas import ReIDResult


@dataclass
class M2Metrics:
    num_tracks: int
    num_global_gt: int
    num_global_pred: int
    interventions: int
    auto_relink_rate: float
    pair_precision: float   # same_pred 중 실제로 same_gt 비율 (false merge ↓)
    pair_recall: float      # same_gt 중 실제로 same_pred 비율 (놓친 연결 ↓)

    def as_dict(self) -> dict:
        return {
            "num_tracks": self.num_tracks,
            "num_global_gt": self.num_global_gt,
            "num_global_pred": self.num_global_pred,
            "interventions": self.interventions,
            "auto_relink_rate": round(self.auto_relink_rate, 4),
            "pair_precision": round(self.pair_precision, 4),
            "pair_recall": round(self.pair_recall, 4),
        }


def evaluate_m2(gt_mapping: dict[int, int], pred: ReIDResult) -> M2Metrics:
    pred_mapping = pred.mapping
    # 공통으로 라벨된 track 만 채점(양쪽에 다 있는 것).
    tracks = sorted(set(gt_mapping) & set(pred_mapping))
    n = len(tracks)

    tp = fp = fn = 0
    for a, b in combinations(tracks, 2):
        same_gt = gt_mapping[a] == gt_mapping[b]
        same_pred = pred_mapping[a] == pred_mapping[b]
        if same_gt and same_pred:
            tp += 1
        elif same_pred and not same_gt:
            fp += 1   # false merge
        elif same_gt and not same_pred:
            fn += 1   # 놓친 연결

    pair_precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    pair_recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0

    # 자동재연결률: 개입(사람 1탭)으로 처리된 track 을 뺀 비율.
    intervened_tracks = {iv.track_id for iv in pred.interventions}
    auto = sum(1 for t in tracks if t not in intervened_tracks)
    auto_relink_rate = (auto / n) if n > 0 else 1.0

    return M2Metrics(
        num_tracks=n,
        num_global_gt=len({gt_mapping[t] for t in tracks}),
        num_global_pred=len({pred_mapping[t] for t in tracks}),
        interventions=len(pred.interventions),
        auto_relink_rate=auto_relink_rate,
        pair_precision=pair_precision,
        pair_recall=pair_recall,
    )
