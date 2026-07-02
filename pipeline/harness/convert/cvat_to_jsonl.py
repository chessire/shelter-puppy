"""CVAT export → M1 GT JSONL 변환기.

입력: CVAT 의 'CVAT for video 1.1' XML.
  - <track id=.. label=..> 안의 <box ... keyframe="1" outside="0">만 취한다.
    keyframe 만 쓰는 이유 = 사람이 실제로 손으로 찍은 프레임이 곧 '정답 표본'.
    (CVAT 가 보간한 사이 프레임은 사람의 판단이 아니므로 GT 에서 뺀다 → 스파스 GT)
  - label 필터(기본 {"dog"}): 임보견만 GT 로. 고양이는 라벨 안 함 → 모델이 고양이를
    개로 잡으면 FP 로 잡히게 한다(타종 오검출 측정).

출력: FrameDetections JSONL (분석 mp4 좌표계, [x,y,w,h]).
  track id = CVAT track id (= M1 track_id). t = frame_idx / fps.

한 프레임이 어떤 track 의 keyframe 이지만 그 track 이 outside 면, 그 프레임은
'라벨했으나 해당 개 없음'으로 남아 빈/부분 프레임이 된다(true negative 채점).
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from ..schemas import BBox, Detection, FrameDetections
from .. import io


def _original_size(root) -> tuple[int, int] | None:
    """CVAT XML meta 의 <original_size> (라벨 당시 해상도)."""
    el = root.find(".//original_size")
    if el is None:
        return None
    w = el.findtext("width")
    h = el.findtext("height")
    if w is None or h is None:
        return None
    return int(w), int(h)


def parse_cvat_video_xml(
    xml_path: str | Path,
    fps: float = 30.0,
    labels: set[str] | None = None,
    target_size: tuple[int, int] | None = None,
) -> list[FrameDetections]:
    """CVAT XML → FrameDetections.

    target_size 가 주어지고 라벨 당시 해상도(<original_size>)와 다르면 박스를
    target 좌표계로 균일 환산한다. 사용자가 실수로 원본(.MOV, 1920×1080)을
    CVAT 에 올려도 분석 mp4(768×432) 좌표로 자동 정렬된다.
    """
    labels = labels or {"dog"}
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # 좌표 환산 배율 결정.
    sx = sy = 1.0
    orig = _original_size(root)
    if target_size is not None:
        if orig is None:
            raise ValueError(
                "target_size 를 줬으나 XML 에 <original_size> 가 없어 환산 불가"
            )
        if orig != tuple(target_size):
            sx = target_size[0] / orig[0]
            sy = target_size[1] / orig[1]

    # frame_idx -> list[Detection]
    by_frame: dict[int, list[Detection]] = {}
    # 라벨된 프레임 집합(빈 프레임 포함): 어떤 track 이든 keyframe 인 프레임.
    labeled_frames: set[int] = set()

    for track in root.findall(".//track"):
        label = track.get("label", "")
        tid = int(track.get("id"))
        for box in track.findall("box"):
            if box.get("keyframe") != "1":
                continue
            frame = int(box.get("frame"))
            labeled_frames.add(frame)
            if box.get("outside") == "1":
                continue
            if label not in labels:
                continue
            xtl = float(box.get("xtl")) * sx
            ytl = float(box.get("ytl")) * sy
            xbr = float(box.get("xbr")) * sx
            ybr = float(box.get("ybr")) * sy
            det = Detection(
                track_id=tid,
                bbox=BBox(xtl, ytl, xbr - xtl, ybr - ytl),
                conf=1.0,
                cls=label,
            )
            by_frame.setdefault(frame, []).append(det)

    frames = [
        FrameDetections(idx, idx / fps, by_frame.get(idx, []))
        for idx in sorted(labeled_frames)
    ]
    return frames


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cvat_to_jsonl")
    p.add_argument("xml", help="CVAT for video 1.1 XML")
    p.add_argument("out", help="출력 JSONL 경로")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--labels", default="dog", help="쉼표구분 라벨 화이트리스트")
    p.add_argument("--map", default=None,
                   help="분석 mp4 의 .map.json — 좌표를 분석 크기로 자동 환산")
    p.add_argument("--target", default=None, help="환산 목표 크기 'WxH' (--map 대신)")
    args = p.parse_args(argv)
    labels = {s.strip() for s in args.labels.split(",") if s.strip()}

    target = None
    if args.map:
        import json
        target = tuple(json.loads(Path(args.map).read_text())["analysis_size"])
    elif args.target:
        w, h = args.target.lower().split("x")
        target = (int(w), int(h))

    frames = parse_cvat_video_xml(
        args.xml, fps=args.fps, labels=labels, target_size=target
    )
    io.write_frames(args.out, frames)
    n_box = sum(len(f.detections) for f in frames)
    print(f"[OK] {args.xml} → {args.out}  labeled_frames={len(frames)} boxes={n_box}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
