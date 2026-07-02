"""놀이터 — 아무 영상이나 넣으면 정규화 + YOLO11m 추적 + 박스/ID 그려서 영상 출력.

사용:
  venv/bin/python -m pipeline.demo_track <영상경로> [--conf 0.25] [--with-cat]

예) 새 강아지 영상을 data/videos 에 넣고:
  venv/bin/python -m pipeline.demo_track data/videos/새영상.MOV
결과: data/demo/새영상_tracked.mp4  (더블클릭해서 보면 됨)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from .preprocess.normalize import normalize
from .m1_track.run import run, COCO_DOG, COCO_CAT


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="demo_track")
    p.add_argument("source", help="아무 영상 경로 (mov/mp4...)")
    p.add_argument("--out", default="data/dev/demo", help="출력 폴더")
    p.add_argument("--conf", type=float, default=0.25, help="검출 신뢰도 임계")
    p.add_argument("--with-cat", action="store_true", help="고양이도 함께 표시")
    args = p.parse_args(argv)

    src = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 분석용 정규화(768·CFR) → 임시 폴더
    with tempfile.TemporaryDirectory() as tmp:
        print(f"[1/2] 정규화 중… ({src.name})")
        meta = normalize(src, tmp)
        analysis_mp4 = Path(tmp) / meta["analysis_mp4"]

        # 2) YOLO11m + ByteTrack 추적 + 시각화
        print("[2/2] 추적·시각화 중… (CPU라 좀 걸려요)")
        viz = out_dir / f"{src.stem}_tracked.mp4"
        pred_jsonl = out_dir / f"{src.stem}_pred.jsonl"
        classes = [COCO_DOG, COCO_CAT] if args.with_cat else [COCO_DOG]
        s = run(analysis_mp4, pred_jsonl, weights="yolo11m.pt",
                conf=args.conf, classes=classes, viz=viz)

    print(f"\n✅ 완료! 더블클릭해서 보세요:\n   {viz}")
    print(f"   (프레임 {s['frames']} · 박스 {s['boxes']} · 트랙 {s['tracks']}개)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
