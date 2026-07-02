"""합침 B — M2(임보견 식별) ⊕ M3(모션곡선).

모션곡선을 전체화면이 아니라 *M2가 임보견으로 묶은 박스* 기준으로 계산한다.
다견 영상에서도 지정한 임보견 한 마리의 동/정만 뽑힌다.

검증: ROI 박스가 임보견(global_dog_id)을 프레임 내내 따라가는지 — 시각화로 확인.
(임보견 트랙 집합은 M2 결과에서; 여기선 M2 도출 GT로 '식별 성공'을 전제)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ..harness import io
from ..harness.schemas import BBox
from .curve import compute_motion_curve, segment_motion


def foster_boxes(pred_frames, foster_tracks: set[int]) -> dict[int, BBox]:
    """임보견 트랙들의 박스를 프레임별로 합쳐 ROI 궤적 생성(겹치면 최고 conf)."""
    out: dict[int, BBox] = {}
    best: dict[int, float] = {}
    for pf in pred_frames:
        for d in pf.detections:
            if d.track_id in foster_tracks and d.conf >= best.get(pf.frame_idx, -1):
                out[pf.frame_idx] = d.bbox
                best[pf.frame_idx] = d.conf
    return out


def label_at(segments, t: float) -> str:
    for s in segments:
        if s.start_t <= t < s.end_t:
            return s.label
    return "static"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="m3_integrate")
    p.add_argument("name", help="영상 이름 (예: IMG_9980)")
    p.add_argument("--foster-dog", type=int, default=0, help="임보견 global id")
    p.add_argument("--thr", type=float, default=8.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--viz", default=None)
    args = p.parse_args(argv)

    mp4 = Path(f"data/dev/analysis/{args.name}.mp4")
    pred = io.read_frames(f"data/dev/preds/{args.name}_m1.jsonl")
    m2 = io.read_json(f"data/dev/gt/{args.name}_m2.json")["mapping"]
    foster_tracks = {int(k) for k, v in m2.items() if int(v) == args.foster_dog}
    other_tracks = {int(k) for k, v in m2.items() if int(v) not in (args.foster_dog, -1)}
    if not foster_tracks:
        print(f"임보견 dog={args.foster_dog} 트랙 없음. m2 매핑: {m2}")
        return 1

    boxes = foster_boxes(pred, foster_tracks)
    motion = compute_motion_curve(mp4, boxes)
    segs, _ = segment_motion(motion, args.fps, args.thr)
    print(f"[합침B] {args.name} 임보견 dog={args.foster_dog} "
          f"(트랙 {sorted(foster_tracks)}): 동/정 {len(segs)}구간")
    for s in segs:
        print(f"   {s.start_t:5.1f}~{s.end_t:5.1f}s  {s.label}")

    if args.viz:
        pred_by_frame = {f.frame_idx: f for f in pred}
        cap = cv2.VideoCapture(str(mp4))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wr = cv2.VideoWriter(args.viz, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in range(n):
            ok, img = cap.read()
            if not ok:
                break
            # 다른 개: 얇은 회색
            pf = pred_by_frame.get(idx)
            if pf:
                for d in pf.detections:
                    if d.track_id in other_tracks:
                        x, y, w, h = [int(v) for v in d.bbox.to_list()]
                        cv2.rectangle(img, (x, y), (x + w, y + h), (160, 160, 160), 1)
            # 임보견 ROI: 동=빨강/정=초록 굵게
            if idx in boxes:
                lab = label_at(segs, idx / args.fps)
                col = (0, 0, 255) if lab == "moving" else (0, 200, 0)
                x, y, w, h = [int(v) for v in boxes[idx].to_list()]
                cv2.rectangle(img, (x, y), (x + w, y + h), col, 3)
                cv2.putText(img, f"FOSTER {lab}", (x, max(15, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            wr.write(img)
        cap.release(); wr.release()
        print(f"   viz → {args.viz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
