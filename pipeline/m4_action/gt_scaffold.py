"""M4 GT 라벨링 스캐폴드 — 사람 라벨 부담 최소화.

측정-우선 원칙: GT(정답)는 사람이 확정한다. 단 백지에서 시작하지 않는다.
  1) M3 모션 곡선(강아지 박스 기준)으로 동/정 *구간 경계 후보*를 자동 생성.
  2) 각 구간의 강아지 크롭을 0.25초 간격으로 뽑아 컨택트시트(PNG)로 렌더.
  3) draft JSON 에 group 힌트(모션 기반)와 action="TODO" 를 채워 둠.
→ 사람은 시트를 보고 action 만 채우고 경계/군을 고치면 된다. group 은 최종적으로
  action 에서 파생(group_of)되며, 모션 힌트는 참고일 뿐 정답이 아니다.

측정 격리: ROI 박스는 M1 GT 키프레임을 dense 보간해 쓴다(M1/M2 오류와 분리). M3 와 동일.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from . import foster_track
from ..harness import io
from ..harness.schemas import BBox
from ..m3_motion.curve import compute_motion_curve, interpolate_boxes, segment_motion
from ..workspace import Workspace


def foster_boxes_gt(name: str, foster_dog: int, ws: Workspace | None = None) -> dict[int, BBox]:
    """강아지 GT 박스를 프레임별로 dense 보간해 ROI 궤적 생성.

    global_dog_id 는 곧 GT track_id 다(m2 gt_derive: pred→GT track id 매핑이므로).
    GT 는 fragmentation 이 없어 강아지가 단일 GT track == foster_dog 로 존재한다.
    """
    ws = ws or Workspace.dev()
    gt = io.read_frames(ws.gt_m1(name))
    boxes = interpolate_boxes(gt, foster_dog)
    if not boxes:
        present = sorted({d.track_id for f in gt for d in f.detections})
        raise SystemExit(f"{name}: 강아지 GT track={foster_dog} 박스 없음. 존재 트랙={present}")
    return boxes


def foster_boxes_pred(name: str, ws: Workspace | None = None,
                      foster_track_id: int | None = None) -> dict[int, BBox]:
    """프로덕션용 — GT 없는 영상의 강아지 박스를 M1 pred 에서 도출.

    단독 강아지(고객 영상은 보통 토리 한 마리) 가정: 프레임마다 가장 큰 dog 박스 =
    강아지. 고양이/distractor 는 cls 로 배제. pred 는 매 프레임이라 보간 불필요.
    다견 잡에서 고객이 track 을 고르면(foster_track_id) 그 트랙만 남겨 정확히 좁힌다.
    """
    ws = ws or Workspace.dev()
    frames = io.read_frames(ws.preds_m1(name))
    # 사진 앵커가 추적 조각들을 합쳐 리스트로 줄 수 있다(같은 강아지 = 트랙 여러 개).
    wanted = None
    if foster_track_id is not None:
        wanted = (set(foster_track_id) if isinstance(foster_track_id, (list, tuple, set))
                  else {foster_track_id})
    boxes: dict[int, BBox] = {}
    for f in frames:
        dogs = [d for d in f.detections if d.cls == "dog"]
        if wanted is not None:
            dogs = [d for d in dogs if d.track_id in wanted]
        if dogs:
            boxes[f.frame_idx] = max(dogs, key=lambda d: d.bbox.area).bbox
    return boxes


def foster_boxes(name: str, ws: Workspace | None = None) -> dict[int, BBox]:
    """강아지 ROI 통합 제공: GT 있으면 GT(측정 격리), 없으면 PRED(프로덕션).

    측정용 5영상은 손라벨 GT 로 정밀하게, 고객/신규 영상은 검출 결과로 자동 처리 —
    설계서의 '개발 채점 vs 운영 자동화' 분리를 코드로 구현한 지점. 강아지 track 은
    ws(개발=foster_map, 잡=meta.json)에서 결정한다.
    """
    ws = ws or Workspace.dev()
    if ws.gt_m1(name).exists():
        return foster_boxes_gt(name, ws.foster_track(name), ws)
    return foster_boxes_pred(name, ws, ws.foster_track(name))


def _sample_indices(f0: int, f1: int, fps: float, step_s: float = 0.25,
                    max_n: int = 8) -> list[int]:
    """[f0, f1) 구간에서 step_s 간격 프레임 인덱스(최대 max_n개, 균등)."""
    step = max(1, int(round(fps * step_s)))
    idxs = list(range(f0, f1, step))
    if not idxs:
        idxs = [f0]
    if len(idxs) > max_n:
        sel = np.linspace(0, len(idxs) - 1, max_n).round().astype(int)
        idxs = [idxs[i] for i in sel]
    return idxs


def _crop(frame, b: BBox, thumb_h: int = 160):
    H, W = frame.shape[:2]
    x1 = max(0, int(b.x)); y1 = max(0, int(b.y))
    x2 = min(W, int(b.x2)); y2 = min(H, int(b.y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((thumb_h, thumb_h, 3), np.uint8)
    c = frame[y1:y2, x1:x2]
    s = thumb_h / c.shape[0]
    return cv2.resize(c, (max(1, int(c.shape[1] * s)), thumb_h))


def render_contact_sheet(name: str, segs, boxes: dict[int, BBox], fps: float,
                         out_png: Path, thumb_h: int = 160,
                         ws: Workspace | None = None) -> None:
    """구간별 강아지 크롭 가로 스트립을 세로로 쌓아 한 장 PNG 로."""
    ws = ws or Workspace.dev()
    cap = cv2.VideoCapture(str(ws.analysis(name)))
    frame_cache: dict[int, np.ndarray] = {}

    def get_frame(idx: int):
        if idx not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            frame_cache[idx] = img if ok else None
        return frame_cache[idx]

    header_h = 26
    pad = 4
    rows = []
    max_w = 0
    for i, s in enumerate(segs):
        f0, f1 = int(s.start_t * fps), int(s.end_t * fps)
        thumbs = []
        for idx in _sample_indices(f0, f1, fps):
            img = get_frame(idx)
            if img is None or idx not in boxes:
                continue
            t = cv2.copyMakeBorder(_crop(img, boxes[idx], thumb_h), 0, 0, 0, pad,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
            thumbs.append(t)
        if not thumbs:
            continue
        strip = np.hstack(thumbs)
        hint = "DYN" if s.label == "moving" else "STA"
        bar = np.full((header_h, strip.shape[1], 3), 40, np.uint8)
        cv2.putText(bar, f"#{i}  {s.start_t:.1f}-{s.end_t:.1f}s  [{hint}]  action=?",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        row = np.vstack([bar, strip])
        rows.append(row)
        max_w = max(max_w, row.shape[1])
    cap.release()
    if not rows:
        print(f"  ! {name}: 렌더할 구간 없음")
        return
    rows = [cv2.copyMakeBorder(r, 0, pad, 0, max_w - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(20, 20, 20)) for r in rows]
    sheet = np.vstack(rows)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), sheet)
    print(f"  → 컨택트시트 {out_png}  ({len(rows)}구간, {sheet.shape[1]}x{sheet.shape[0]})")


def build(name: str, foster_dog: int, thr: float, fps: float, sheet_dir: Path,
          ws: Workspace | None = None) -> None:
    ws = ws or Workspace.dev()
    boxes = foster_boxes_gt(name, foster_dog, ws)
    motion = compute_motion_curve(ws.analysis(name), boxes)
    segs, _ = segment_motion(motion, fps, thr)

    # draft JSON: group 은 모션 힌트, action 은 사람이 채울 자리.
    draft = {
        "foster_dog": foster_dog,
        "_note": "action 을 점프/달리기/걷기/앉기/엎드림 중 하나로 채우세요. "
                 "group 은 비워두면 action 에서 자동 파생됩니다. 경계도 자유롭게 수정 가능.",
        "segments": [
            {
                "start_t": round(s.start_t, 2),
                "end_t": round(s.end_t, 2),
                "group_hint": "dynamic" if s.label == "moving" else "static",
                "action": "TODO",
            }
            for s in segs
        ],
    }
    out_json = ws.gt_dir / f"{name}_m4.draft.json"
    io.write_json(out_json, draft)
    print(f"[{name}] dog={foster_dog}: {len(segs)}구간 → {out_json}")
    render_contact_sheet(name, segs, boxes, fps, sheet_dir / f"{name}_m4_sheet.png", ws=ws)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="m4_gt_scaffold")
    p.add_argument("names", nargs="*", default=None,
                   help="영상 이름들 (기본: 테스트 5영상 전부)")
    p.add_argument("--foster-dog", type=int, default=None,
                   help="강아지 track 강제 지정(기본: foster_map.json)")
    p.add_argument("--thr", type=float, default=8.0, help="모션 동/정 임계(경계 후보용)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--sheet-dir", default="data/dev/m4_sheets")
    args = p.parse_args(argv)

    names = args.names or ["IMG_0004", "IMG_0008", "IMG_9980", "IMG_0069", "IMG_0066"]
    sheet_dir = Path(args.sheet_dir)
    for n in names:
        foster = args.foster_dog if args.foster_dog is not None else foster_track(n)
        build(n, foster, args.thr, args.fps, sheet_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
