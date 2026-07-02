"""M0 데이터 계약 — 파이프라인 전 단계의 출력/정답(GT) 스키마.

모든 모듈(M1~M5)은 여기 정의된 형태로 결과를 쓰고, 모든 메트릭은 여기서 읽는다.
스키마를 한 곳에 동결해 두면 모듈을 병렬로 만들어도 핸드오프가 깨지지 않는다.

설계서 매핑:
  - M1 검출·추적   → FrameDetections (프레임당 박스 + track_id)
  - M2 re-ID       → ReIDResult     (track_id → global_id 매핑 + 개입 로그)
  - M3 모션 곡선   → Segment(label=moving|static)
  - M4 동작 판별   → ActionSegment  (동작군/정지군 + uncertain)
  - M5 TTS 연결    → MatchEntry      (구절 → 클립 구간)

직렬화 규약:
  - 좌표계는 bbox = [x, y, w, h] (좌상단 x,y / 너비,높이), 픽셀 단위. 설계서 스키마 준수.
  - 프레임열(FrameDetections)은 JSONL: 한 줄 = 한 프레임.
  - 그 외 단건 산출물은 단일 JSON 객체.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------- #
# 공통 기본형
# --------------------------------------------------------------------------- #
@dataclass
class BBox:
    """축 정렬 박스. [x, y, w, h] (좌상단 기준, 픽셀)."""
    x: float
    y: float
    w: float
    h: float

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.w, self.h]

    @classmethod
    def from_list(cls, v: list[float]) -> "BBox":
        x, y, w, h = v
        return cls(float(x), float(y), float(w), float(h))

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


# --------------------------------------------------------------------------- #
# M1 — 검출·추적
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """프레임 내 단일 검출. GT는 conf=1.0 로 둔다."""
    track_id: int
    bbox: BBox
    conf: float = 1.0
    cls: str = "dog"  # COCO 클래스. 고양이 distractor 구분용으로 보존.

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox.to_list(),
            "conf": self.conf,
            "cls": self.cls,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        return cls(
            track_id=int(d["track_id"]),
            bbox=BBox.from_list(d["bbox"]),
            conf=float(d.get("conf", 1.0)),
            cls=str(d.get("cls", "dog")),
        )


@dataclass
class FrameDetections:
    """한 프레임의 검출 묶음. JSONL 한 줄에 대응."""
    frame_idx: int
    t: float  # 초 단위 timestamp (원본 타임코드)
    detections: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "t": self.t,
            "detections": [d.to_dict() for d in self.detections],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrameDetections":
        return cls(
            frame_idx=int(d["frame_idx"]),
            t=float(d["t"]),
            detections=[Detection.from_dict(x) for x in d.get("detections", [])],
        )


# --------------------------------------------------------------------------- #
# M2 — re-ID
# --------------------------------------------------------------------------- #
@dataclass
class InterventionRecord:
    """사람 1탭 개입 1건. 사용성 지표(영상당 개입수)의 원천."""
    track_id: int
    t: float
    reason: str  # "new_video" | "reappear" | "ambiguous" 등


@dataclass
class ReIDResult:
    """track_id → global_dog_id 매핑 + 재연결 근거 + 개입 로그.

    GT(정답)는 mapping 만 채우면 된다(개입/유사도는 자동출력에만 존재).
    """
    mapping: dict[int, int]  # track_id -> global_dog_id
    interventions: list[InterventionRecord] = field(default_factory=list)
    # 자동 재연결 근거 (track_id -> {"sim": float, "anchor": global_id, "auto": bool})
    relink_evidence: dict[int, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mapping": {str(k): v for k, v in self.mapping.items()},
            "interventions": [asdict(i) for i in self.interventions],
            "relink_evidence": {str(k): v for k, v in self.relink_evidence.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReIDResult":
        return cls(
            mapping={int(k): int(v) for k, v in d.get("mapping", {}).items()},
            interventions=[InterventionRecord(**i) for i in d.get("interventions", [])],
            relink_evidence={int(k): v for k, v in d.get("relink_evidence", {}).items()},
        )


# --------------------------------------------------------------------------- #
# M3 / M4 — 시간 구간 (모션 / 동작)
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """M3 모션 구간. label ∈ {moving, static}."""
    start_t: float
    end_t: float
    label: str

    @property
    def dur(self) -> float:
        return max(0.0, self.end_t - self.start_t)

    def to_dict(self) -> dict:
        return {"start_t": self.start_t, "end_t": self.end_t, "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(float(d["start_t"]), float(d["end_t"]), str(d["label"]))


@dataclass
class ActionSegment:
    """M4 동작 구간. group ∈ {dynamic, static}; action 은 선택(점프/달리기/웅크림…)."""
    start_t: float
    end_t: float
    group: str
    action: Optional[str] = None
    conf: float = 1.0
    uncertain: bool = False

    @property
    def dur(self) -> float:
        return max(0.0, self.end_t - self.start_t)

    def to_dict(self) -> dict:
        return {
            "start_t": self.start_t,
            "end_t": self.end_t,
            "group": self.group,
            "action": self.action,
            "conf": self.conf,
            "uncertain": self.uncertain,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActionSegment":
        return cls(
            start_t=float(d["start_t"]),
            end_t=float(d["end_t"]),
            group=str(d["group"]),
            action=d.get("action"),
            conf=float(d.get("conf", 1.0)),
            uncertain=bool(d.get("uncertain", False)),
        )


# --------------------------------------------------------------------------- #
# M5 — TTS 연결 (구절 → 클립 구간)
# --------------------------------------------------------------------------- #
@dataclass
class MatchEntry:
    """내레이션 구절 ↔ 영상 클립 구간 매칭 1건."""
    phrase_id: int
    source: str       # 어느 소스 영상의 구간인지 (clip 식별자)
    start_t: float
    end_t: float
    uncertain: bool = False

    def to_dict(self) -> dict:
        return {
            "phrase_id": self.phrase_id,
            "source": self.source,
            "start_t": self.start_t,
            "end_t": self.end_t,
            "uncertain": self.uncertain,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MatchEntry":
        return cls(
            phrase_id=int(d["phrase_id"]),
            source=str(d["source"]),
            start_t=float(d["start_t"]),
            end_t=float(d["end_t"]),
            uncertain=bool(d.get("uncertain", False)),
        )
