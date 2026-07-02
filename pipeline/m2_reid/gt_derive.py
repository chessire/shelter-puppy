"""M2 정답 도출 — M1 GT 로부터 'pred 트랙 → 진짜 강아지' 매핑을 자동 추출.

새 라벨링 없이 M2 채점용 GT 를 만든다. 베이스라인 트래커가 한 강아지를 여러
pred 트랙으로 쪼갰으므로(fragmentation), 각 pred 트랙이 *실제로 어느 GT 강아지*
인지를 박스 겹침(IoU)으로 판정한다.

- pred 트랙 t 의 각 검출을, 같은 프레임 GT 박스들과 IoU 매칭 → 어느 GT 강아지인지 투표
- 다수결로 t 의 진짜 강아지(global_dog_id = GT track_id) 결정
- 어떤 GT 강아지와도 거의 안 겹치면(고양이 오검출 등) -1(=distractor) 로 표기

출력: data/gt/<video>_m2.json = {"mapping": {pred_track_id: true_dog_id}}
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from ..harness import io
from ..harness.metrics.common import iou


def derive_m2_gt(
    gt_frames, pred_frames, iou_thr: float = 0.5, min_overlap_ratio: float = 0.3
) -> dict[int, int]:
    """pred track_id → true_dog_id(GT track_id) 또는 -1(distractor)."""
    gt_by_frame = {f.frame_idx: f for f in gt_frames}
    # pred 트랙별 투표 집계: votes[pred_tid][gt_tid] = 겹친 횟수
    votes: dict[int, Counter] = defaultdict(Counter)
    totals: dict[int, int] = defaultdict(int)

    for pf in pred_frames:
        gf = gt_by_frame.get(pf.frame_idx)
        if gf is None:
            continue  # GT 라벨 없는 프레임은 판정 불가
        for pd in pf.detections:
            totals[pd.track_id] += 1
            best_gt, best_iou = None, 0.0
            for gd in gf.detections:
                v = iou(pd.bbox, gd.bbox)
                if v > best_iou:
                    best_iou, best_gt = v, gd.track_id
            if best_gt is not None and best_iou >= iou_thr:
                votes[pd.track_id][best_gt] += 1

    mapping: dict[int, int] = {}
    for tid, total in totals.items():
        c = votes.get(tid)
        if not c:
            mapping[tid] = -1  # GT 와 거의 안 겹침 = distractor(고양이 등)
            continue
        gt_tid, hits = c.most_common(1)[0]
        # 겹침이 트랙 길이의 일정 비율 미만이면 distractor 로 본다.
        mapping[tid] = gt_tid if hits >= max(1, total * min_overlap_ratio) else -1
    return mapping


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="m2_gt_derive")
    p.add_argument("--gt-dir", default="data/dev/gt")
    p.add_argument("--pred-dir", default="data/dev/preds")
    args = p.parse_args(argv)

    for gt_path in sorted(Path(args.gt_dir).glob("*_m1.jsonl")):
        name = gt_path.name.replace("_m1.jsonl", "")
        pred_path = Path(args.pred_dir) / f"{name}_m1.jsonl"
        if not pred_path.exists():
            continue
        gt = io.read_frames(gt_path)
        pred = io.read_frames(pred_path)
        mapping = derive_m2_gt(gt, pred)
        out = Path(args.gt_dir) / f"{name}_m2.json"
        io.write_json(out, {"mapping": {str(k): v for k, v in mapping.items()}})

        true_dogs = sorted({v for v in mapping.values() if v != -1})
        distractor = sum(1 for v in mapping.values() if v == -1)
        print(f"[OK] {name}: pred트랙 {len(mapping)}개 → 진짜강아지 {len(true_dogs)}마리"
              f"{true_dogs}  distractor {distractor}개  → {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
