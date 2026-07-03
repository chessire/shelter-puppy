"""Workspace(root) — 모든 산출물을 root 기준으로 해석하는 경로 리졸버.

설계서 '개발 골든셋 vs 잡 런타임 분리'를 코드로 구현하는 단일 지점. 하드코딩
`data/dev/...` 경로 대신 이 리졸버를 받게 해서, 같은 파이프라인 코드가 두 루트에서
돈다:

  개발(측정) = Workspace.dev()      → data/dev   (GT·라벨영상·글로벌 메타, 영구·공유)
  잡(런타임) = Workspace.job(id)    → $DATA_ROOT/<id> (요청당 1개, GT 불필요·자족·격리)

scene_tags·foster_track 은 출처가 둘로 갈린다(설계서: '개발은 글로벌 파일, 잡은
per-job meta.json'). 여기선 meta.json 을 먼저 보고 없으면 글로벌 파일로 폴백하는
cascade 로 통일해, dev/job 양쪽이 같은 호출부를 쓴다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# 원본(편집용) 영상 확장자 탐색 순서 — 분석 mp4(저해상도)가 아니라 출력용 고해상도 원본.
_VIDEO_EXTS = ("MOV", "mov", "mp4", "MP4")


class Workspace:
    """root 하위 표준 레이아웃(analysis/preds/input/gt/videos/cards/out + meta.json)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"Workspace({str(self.root)!r})"

    # --- 생성자 ----------------------------------------------------------- #
    @classmethod
    def dev(cls) -> "Workspace":
        """개발 골든셋(data/dev) — GT·채점·라벨영상이 있는 측정용 루트."""
        return cls("data/dev")

    @classmethod
    def job(cls, job_id: str, data_root: str | Path | None = None) -> "Workspace":
        """잡 런타임 루트($DATA_ROOT/<id>, 기본 ./jobs/<id>).

        DATA_ROOT 환경변수로 저장 위치 이전(디스크/볼륨/S3 마운트) 용이.
        """
        base = Path(data_root or os.environ.get("DATA_ROOT", "jobs"))
        return cls(base / job_id)

    # --- 디렉토리 --------------------------------------------------------- #
    @property
    def analysis_dir(self) -> Path:
        return self.root / "analysis"

    @property
    def preds_dir(self) -> Path:
        return self.root / "preds"

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def gt_dir(self) -> Path:
        return self.root / "gt"

    @property
    def videos_dir(self) -> Path:
        return self.root / "videos"

    @property
    def cards_dir(self) -> Path:
        return self.root / "cards"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    # --- 파일 리졸버 ------------------------------------------------------ #
    def analysis(self, name: str) -> Path:
        """P0 정규화 분석 mp4 (768·CFR)."""
        return self.analysis_dir / f"{name}.mp4"

    def analysis_map(self, name: str) -> Path:
        """분석→원본 공간/시간 매핑(.map.json)."""
        return self.analysis_dir / f"{name}.map.json"

    def preds_m1(self, name: str) -> Path:
        """M1 검출·트랙 pred (FrameDetections JSONL)."""
        return self.preds_dir / f"{name}_m1.jsonl"

    def preds_m4(self, name: str) -> Path:
        """M4 동작 태그 (ActionSegment JSON)."""
        return self.preds_dir / f"{name}_m4.json"

    def gt_m1(self, name: str) -> Path:
        """M1 GT (개발 측정용; 잡엔 없음)."""
        return self.gt_dir / f"{name}_m1.jsonl"

    def out(self, *parts: str) -> Path:
        """렌더 출력 경로."""
        return self.out_dir.joinpath(*parts)

    def source_video(self, name: str) -> Path | None:
        """편집용 원본 고해상도 영상 — 잡은 input/, 개발은 videos/ 에서 찾는다.

        설계서 '분석용 ≠ 편집용, 출력은 원본 그대로'. 둘 다 탐색해 dev/job 통일.
        """
        for d in (self.input_dir, self.videos_dir):
            for ext in _VIDEO_EXTS:
                p = d / f"{name}.{ext}"
                if p.exists():
                    return p
        return None

    # --- meta.json (잡 상태머신 + per-job 메타) --------------------------- #
    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    def read_meta(self) -> dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        return {}

    def write_meta(self, meta: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def update_meta(self, **kw) -> dict:
        """meta.json 부분 갱신(읽고 병합 후 저장). 갱신된 dict 반환."""
        meta = self.read_meta()
        meta.update(kw)
        self.write_meta(meta)
        return meta

    @property
    def state(self) -> str | None:
        return self.read_meta().get("state")

    def set_state(self, state: str) -> None:
        self.update_meta(state=state)

    # --- per-job vs 글로벌 메타 cascade ---------------------------------- #
    def scene_tags(self) -> dict[str, set]:
        """영상별 장면 태그맵. 잡 = 사람(scene_tags) > 요청주도 추론(scene_tags_auto),
        영상 단위로 사람 태그가 있으면 그 영상은 사람 것만. 개발 = videos/scene_tags.json."""
        meta = self.read_meta()
        human = meta.get("scene_tags") or {}
        auto = meta.get("scene_tags_auto") or {}
        if human or auto:
            raw = {**auto, **human}     # 같은 영상 키는 사람이 덮는다
        else:
            p = self.videos_dir / "scene_tags.json"
            raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        return {k: set(v) for k, v in raw.items() if not k.startswith("_")}

    def foster_track(self, name: str) -> int | None:
        """임보견 track id. 잡=meta.json['foster_track'], 개발=gt/foster_map.json.

        잡은 단독견이면 None(=foster_boxes_pred 가 최대 dog 박스로 자동 처리),
        다견이면 고객이 고른 track id 가 meta 에 저장된다. 개발은 GT track id.
        """
        meta = self.read_meta()
        # 영상별 맵(다견 잡 — 사진 앵커/카드가 채움)이 단일 값보다 우선.
        # 값이 리스트면 추적 조각들(같은 개가 트랙 여러 개로 갈라진 것) — 그대로 전달,
        # foster_boxes_pred 가 집합 필터로 처리한다.
        fmap = meta.get("foster_track_map") or {}
        if fmap.get(name) is not None:
            v = fmap[name]
            return [int(t) for t in v] if isinstance(v, list) else int(v)
        if meta.get("foster_track") is not None:
            return int(meta["foster_track"])
        p = self.gt_dir / "foster_map.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            if name in m:
                return int(m[name])
        return None
