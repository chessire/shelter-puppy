"""M1 베이스라인 — YOLO + ByteTrack → 분석 mp4에서 pred(FrameDetections JSONL).

설계서 M1 '측정용 최소 구현'. 분석 mp4(768·CFR) 좌표계로 박스+track_id 를 뽑아
M0 하네스의 pred 입력을 만든다. GT(CVAT)와 같은 좌표계라 바로 채점 가능.

COCO dog=16 만 검출(고양이=15 는 제외). 모델이 고양이를 dog 로 오분류하면 그
검출이 dog 로 나와 dog-only GT 대비 FP 로 잡힌다(타종 오검출 측정).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..harness import io
from ..harness.schemas import BBox, Detection, FrameDetections

COCO_DOG = 16
COCO_CAT = 15


def _fps_from_map(mp4: Path, default: float = 30.0) -> float:
    mp = mp4.with_suffix(".map.json")
    if mp.exists():
        try:
            return float(json.loads(mp.read_text())["fps"])
        except Exception:
            pass
    return default


def run(
    mp4: str | Path,
    out_jsonl: str | Path,
    weights: str = "yolo11m.pt",  # [M0 측정 근거] 11n 대비 micro miss 0.32→0.21
    conf: float = 0.25,
    classes: list[int] | None = None,
    viz: str | Path | None = None,
) -> dict:
    from ultralytics import YOLO  # 지연 import (무거움)

    mp4 = Path(mp4)
    classes = classes if classes is not None else [COCO_DOG]
    fps = _fps_from_map(mp4)
    model = YOLO(weights)

    writer = None
    frames: list[FrameDetections] = []
    n_box = 0
    results = model.track(
        source=str(mp4),
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        classes=classes,
        conf=conf,
        verbose=False,
    )
    for idx, r in enumerate(results):
        dets: list[Detection] = []
        b = r.boxes
        if b is not None and b.id is not None:
            xyxy = b.xyxy.cpu().numpy()
            ids = b.id.cpu().numpy().astype(int)
            confs = b.conf.cpu().numpy()
            clss = b.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, cf, cl in zip(xyxy, ids, confs, clss):
                dets.append(Detection(
                    track_id=int(tid),
                    bbox=BBox(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                    conf=float(cf),
                    cls=model.names.get(int(cl), str(cl)),
                ))
            n_box += len(dets)
        frames.append(FrameDetections(idx, idx / fps, dets))

        if viz is not None:
            import cv2
            annotated = r.plot()
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(viz), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            writer.write(annotated)

    if writer is not None:
        writer.release()

    io.write_frames(out_jsonl, frames)
    n_tracks = len({d.track_id for f in frames for d in f.detections})
    summary = {
        "mp4": mp4.name,
        "out": str(out_jsonl),
        "frames": len(frames),
        "boxes": n_box,
        "tracks": n_tracks,
        "viz": str(viz) if viz else None,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="m1_track")
    p.add_argument("mp4", help="분석 mp4 경로")
    p.add_argument("out", help="출력 pred JSONL")
    p.add_argument("--weights", default="yolo11m.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--with-cat", action="store_true",
                   help="고양이(15)도 함께 검출(디버그·시각화용)")
    p.add_argument("--viz", default=None, help="검출 시각화 mp4 출력 경로")
    args = p.parse_args(argv)
    classes = [COCO_DOG, COCO_CAT] if args.with_cat else [COCO_DOG]
    s = run(args.mp4, args.out, args.weights, args.conf, classes, args.viz)
    print(f"[OK] {s['mp4']} → {s['out']}  "
          f"frames={s['frames']} boxes={s['boxes']} tracks={s['tracks']}"
          + (f"  viz={s['viz']}" if s['viz'] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
