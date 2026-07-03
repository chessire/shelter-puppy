"""M3 모션 곡선 — 강아지 박스 ROI 모션(카메라 보상) → 동/정 구간.

설계서:
- 박스 내부 ROI 모션(프레임차) 측정
- 글로벌 모션(카메라) 제거: 배경 특징점 추적 → RANSAC affine → 보상
  (손떨림·팬을 강아지 동작으로 오검출 방지)
- 고유 모션 = 보상 후 ROI 잔여 모션
- 시간축 스무딩 + 임계값 → 동/정 세그먼트

측정 격리: ROI 박스는 GT 키프레임을 dense 보간해 사용(M1/M2 오류와 분리).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ..harness import io
from ..harness.schemas import BBox, Segment


def interpolate_boxes(gt_frames, track_id: int) -> dict[int, BBox]:
    """GT 키프레임 박스를 프레임 단위로 선형 보간."""
    kf = []
    for f in gt_frames:
        for d in f.detections:
            if d.track_id == track_id:
                kf.append((f.frame_idx, d.bbox))
    kf.sort()
    dense: dict[int, BBox] = {}
    for (f0, b0), (f1, b1) in zip(kf, kf[1:]):
        for fi in range(f0, f1 + 1):
            a = (fi - f0) / (f1 - f0) if f1 > f0 else 0.0
            dense[fi] = BBox(
                b0.x + a * (b1.x - b0.x), b0.y + a * (b1.y - b0.y),
                b0.w + a * (b1.w - b0.w), b0.h + a * (b1.h - b0.h),
            )
    return dense


def _clamp_box(b: BBox, W: int, H: int):
    x1 = max(0, int(b.x)); y1 = max(0, int(b.y))
    x2 = min(W, int(b.x2)); y2 = min(H, int(b.y2))
    return x1, y1, x2, y2


def compute_motion_curve(analysis_mp4: Path, boxes: dict[int, BBox]) -> dict[int, float]:
    """프레임별 카메라 보상 ROI 모션 스칼라(0~255 평균차)."""
    cap = cv2.VideoCapture(str(analysis_mp4))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    motion: dict[int, float] = {}
    prev_gray = None

    idxs = sorted(boxes)
    start = idxs[0]
    # 순차 디코드 — 프레임마다 cap.set(랜덤 시크)하면 매번 키프레임부터 재디코드라
    # 수 배 느리다. 분석 mp4 는 P0 CFR 계약이라 순차 read 가 같은 프레임을 준다.
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for idx in range(start, idxs[-1] + 1):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and idx in boxes:
            x1, y1, x2, y2 = _clamp_box(boxes[idx], W, H)
            # 1) 글로벌 모션: 박스 밖(배경) 특징점으로 affine 추정
            bg_mask = np.full((H, W), 255, np.uint8)
            mx, my = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
            bg_mask[max(0, y1 - my):y2 + my, max(0, x1 - mx):x2 + mx] = 0
            warped_prev = prev_gray
            p0 = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 10, mask=bg_mask)
            if p0 is not None and len(p0) >= 8:
                p1, stt, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
                if p1 is not None:
                    g0 = p0[stt.ravel() == 1]; g1 = p1[stt.ravel() == 1]
                    if len(g0) >= 6:
                        M, _ = cv2.estimateAffinePartial2D(g0, g1, method=cv2.RANSAC)
                        if M is not None:
                            warped_prev = cv2.warpAffine(prev_gray, M, (W, H))
            # 2) 보상 후 ROI 잔여 모션
            roi_c = gray[y1:y2, x1:x2].astype(np.int16)
            roi_p = warped_prev[y1:y2, x1:x2].astype(np.int16)
            if roi_c.size > 0 and roi_c.shape == roi_p.shape:
                motion[idx] = float(np.abs(roi_c - roi_p).mean())
        prev_gray = gray
    cap.release()
    return motion


def segment_motion(
    motion: dict[int, float], fps: float, thr: float, smooth_win: int = 7,
    min_dur: float = 0.3,
) -> tuple[list[Segment], dict[int, float]]:
    """스무딩 + 임계값 → 동/정 세그먼트. 반환 (segments, smoothed_curve)."""
    idxs = sorted(motion)
    vals = np.array([motion[i] for i in idxs])
    if len(vals) == 0:
        return [], {}
    k = max(1, smooth_win)
    kernel = np.ones(k) / k
    sm = np.convolve(vals, kernel, mode="same")
    smoothed = {idxs[i]: float(sm[i]) for i in range(len(idxs))}

    labels = ["moving" if v >= thr else "static" for v in sm]
    segs: list[Segment] = []
    s = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            t0 = idxs[s] / fps
            t1 = (idxs[i - 1] + 1) / fps
            segs.append(Segment(t0, t1, labels[s]))
            s = i
    # min_dur 보다 짧은 구간은 이웃에 흡수(과분할 방지)
    merged: list[Segment] = []
    for seg in segs:
        if merged and seg.dur < min_dur:
            merged[-1] = Segment(merged[-1].start_t, seg.end_t, merged[-1].label)
        elif merged and merged[-1].label == seg.label:
            merged[-1] = Segment(merged[-1].start_t, seg.end_t, seg.label)
        else:
            merged.append(seg)
    return merged, smoothed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="m3_curve")
    p.add_argument("video")
    p.add_argument("--gt", required=True, help="M1 GT jsonl (ROI 박스 소스)")
    p.add_argument("--track", type=int, default=0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--thr", type=float, default=8.0)
    p.add_argument("--out", default=None, help="동/정 segments json 출력")
    args = p.parse_args(argv)

    boxes = interpolate_boxes(io.read_frames(args.gt), args.track)
    motion = compute_motion_curve(Path(args.video), boxes)
    segs, _ = segment_motion(motion, args.fps, args.thr)
    if args.out:
        io.write_segments(args.out, segs)
    vals = list(motion.values())
    print(f"[OK] {Path(args.video).name} track={args.track}: "
          f"프레임 {len(motion)}  모션 중앙값 {np.median(vals):.1f}  "
          f"세그먼트 {len(segs)}개")
    for s in segs:
        print(f"   {s.start_t:5.1f}~{s.end_t:5.1f}s  {s.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
