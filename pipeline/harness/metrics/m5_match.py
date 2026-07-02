"""M5 TTS 연결 메트릭 — 구절↔클립 매칭 정확도 + 매칭 uncertain율.

GT/pred 모두 MatchEntry(phrase_id → source 클립 구간) 리스트.
한 구절을 '맞게' 연결했다 = pred 의 source 가 GT 와 같고 시간 IoU>=thr.
uncertain 으로 사람에게 넘긴 구절은 정확도 분모에서 빼고 따로 센다.

설계서 측정 지표:
  - 매칭 정확도        → match_accuracy (committed 구절 중 정답 비율)
  - 매칭 uncertain율    → uncertain_rate
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import MatchEntry, Segment
from .common import interval_iou


@dataclass
class M5Metrics:
    num_phrases: int
    committed: int
    uncertain_rate: float
    match_accuracy: float

    def as_dict(self) -> dict:
        return {
            "num_phrases": self.num_phrases,
            "committed": self.committed,
            "uncertain_rate": round(self.uncertain_rate, 4),
            "match_accuracy": round(self.match_accuracy, 4),
        }


def _as_segment(m: MatchEntry) -> Segment:
    return Segment(m.start_t, m.end_t, m.source)


def evaluate_m5(
    gt: list[MatchEntry], pred: list[MatchEntry], iou_thr: float = 0.5
) -> M5Metrics:
    gt_by_phrase = {m.phrase_id: m for m in gt}
    pred_by_phrase = {m.phrase_id: m for m in pred}

    num_phrases = len(gt_by_phrase)
    committed = correct = uncertain = 0
    for pid, gm in gt_by_phrase.items():
        pm = pred_by_phrase.get(pid)
        if pm is None:
            continue  # 예측 누락 → committed/uncertain 어디에도 안 듦(미연결)
        if pm.uncertain:
            uncertain += 1
            continue
        committed += 1
        same_source = pm.source == gm.source
        tiou = interval_iou(_as_segment(gm), _as_segment(pm))
        if same_source and tiou >= iou_thr:
            correct += 1

    return M5Metrics(
        num_phrases=num_phrases,
        committed=committed,
        uncertain_rate=(uncertain / num_phrases) if num_phrases > 0 else 0.0,
        match_accuracy=(correct / committed) if committed > 0 else 1.0,
    )
