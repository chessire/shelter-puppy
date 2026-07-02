"""검출 실험 — 모델/해상도별 검출력(miss_rate·fp)을 GT 키프레임에서 빠르게 비교.

추적(ByteTrack) 없이 *검출만* GT 라벨 프레임에서 돌린다. 병목인 miss_rate 를
싸게 비교하려는 용도(정체성 지표는 M2 소관이라 여기선 안 봄).

좌표 정렬: 분석 mp4 가 GT(768) 와 다른 해상도면 --gt-scale 로 pred 를 GT 좌표로 환산.
예) 1536 분석에서 검출 → GT(768) 비교: pred × (768/1536)=0.5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ..harness import io
from ..harness.schemas import BBox, Detection, FrameDetections
from ..harness.metrics.common import match_by_iou

COCO_DOG = 16


def detection_miss_fp(
    gt: list[FrameDetections], pred: list[FrameDetections], iou_thr: float = 0.5
) -> tuple[int, int, int]:
    """프레임별 IoU 매칭만으로 검출 miss/fp (추적 정체성 무관). 반환 (num_gt, fn, fp)."""
    pmap = {f.frame_idx: f for f in pred}
    num_gt = fn = fp = 0
    for gf in gt:
        gb = [d.bbox for d in gf.detections]
        pb = [d.bbox for d in pmap.get(gf.frame_idx, FrameDetections(gf.frame_idx, 0, [])).detections]
        num_gt += len(gb)
        _, un_g, un_p = match_by_iou(gb, pb, iou_thr)
        fn += len(un_g)
        fp += len(un_p)
    return num_gt, fn, fp


def detect_on_gt_frames(
    model, analysis_mp4: Path, gt_frames: list[FrameDetections],
    conf: float, imgsz: int, gt_scale: float,
) -> list[FrameDetections]:
    cap = cv2.VideoCapture(str(analysis_mp4))
    out: list[FrameDetections] = []
    for gf in gt_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, gf.frame_idx)
        ok, img = cap.read()
        if not ok:
            out.append(FrameDetections(gf.frame_idx, gf.t, []))
            continue
        r = model.predict(img, classes=[COCO_DOG], conf=conf, imgsz=imgsz,
                          verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            for (x1, y1, x2, y2), cf in zip(
                r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
            ):
                dets.append(Detection(
                    track_id=-1,
                    bbox=BBox(float(x1) * gt_scale, float(y1) * gt_scale,
                              float(x2 - x1) * gt_scale, float(y2 - y1) * gt_scale),
                    conf=float(cf), cls="dog",
                ))
        out.append(FrameDetections(gf.frame_idx, gf.t, dets))
    cap.release()
    return out


def main(argv=None) -> int:
    from ultralytics import YOLO
    p = argparse.ArgumentParser(prog="detect_eval")
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--analysis-dir", default="data/dev/analysis")
    p.add_argument("--gt-dir", default="data/dev/gt")
    p.add_argument("--gt-scale", type=float, default=1.0,
                   help="pred→GT 좌표 환산 배율 (다른 해상도 분석영상 쓸 때)")
    args = p.parse_args(argv)

    model = YOLO(args.weights)
    print(f"# {args.weights}  conf={args.conf} imgsz={args.imgsz} scale={args.gt_scale}")
    print(f"  {'video':12}{'num_gt':>7}{'miss':>7}{'fp':>5}")
    print("  " + "-" * 31)
    tot_gt = tot_fn = tot_fp = 0
    for gt_path in sorted(Path(args.gt_dir).glob("*_m1.jsonl")):
        name = gt_path.name.replace("_m1.jsonl", "")
        mp4 = Path(args.analysis_dir) / f"{name}.mp4"
        if not mp4.exists():
            continue
        gt = io.read_frames(gt_path)
        pred = detect_on_gt_frames(model, mp4, gt, args.conf, args.imgsz, args.gt_scale)
        ng, fn, fp = detection_miss_fp(gt, pred)
        tot_gt += ng; tot_fn += fn; tot_fp += fp
        miss = fn / ng if ng else 0.0
        print(f"  {name:12}{ng:>7}{miss:>7.2f}{fp:>5}")
    micro_miss = tot_fn / tot_gt if tot_gt else 0.0
    print("  " + "-" * 31)
    print(f"  {'MICRO':12}{tot_gt:>7}{micro_miss:>7.2f}{tot_fp:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
