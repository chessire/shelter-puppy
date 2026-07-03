"""M6 — 편집 실행 (Layer 3). 모드 B(편집만, TTS 없음) 우선.

설계 원칙: LLM 은 자연어 → 편집 인텐트 JSON *번역만*, 파이썬 실행기가 인텐트를
ffmpeg/OpenCV 로 결정론 컴파일. M6 는 정확도 게이트가 아니라 *엔지니어링 검증*.

구성형(compositional) 인텐트:
  EditPlan = 전역 title + 순서 있는 EditBlock 리스트.
  각 EditBlock 이 영상의 한 '구간 연출'(select·길이·pace·전환·크롭·줌·배속).
  "먼저 노는 거 → 마지막 얼굴 줌" 같은 순서·구성을 블록 시퀀스로 표현한다.
  단일 요청은 블록 1개짜리 plan (전역 모델과 통일).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# 블록 1개의 JSON 스키마(Gemma format= 강제). 고수준 의도만, 실제 구간은 컴파일러가.
_BLOCK_PROPS = {
    "select": {"type": "string", "enum": ["dynamic", "static", "all", "묘기"]},
    "target_dur": {"type": "number"},   # 이 블록 목표 길이(초). 0/누락이면 가능한 만큼.
    "pace": {"type": "string", "enum": ["fast", "calm"]},
    "transition": {"type": "string", "enum": ["cut", "xfade"]},
    "subject": {"type": "string", "enum": ["foster", "full"]},
    "zoom": {"type": "string", "enum": ["none", "gradual"]},
    "speed": {"type": "number"},
    "caption": {"type": "string"},      # 이 블록 동안 아래에 띄울 한글 자막. 없으면 빈 문자열.
    "keywords": {"type": "array", "items": {"type": "string"}},  # 장면 키워드(사용자 표현 그대로)
    # ⚠️ 함의 키워드("keywords_implied" — 카페→실내 같은 유추 속성)는 기각(2026-07-03,
    # 사용자 반례): ①틀린 함의(실내 수영장인데 실외 유추 → 오배정) ②블록 간 공유
    # 함의(수영장·계곡 둘 다 '물')는 변별력 없음 ③실내/실외 같은 속성은 요청 단어가
    # 아니라 영상만 아는 사실 — 속성은 픽셀에서 관찰해야(차기: 캡셔닝+텍스트 매칭).
    # ⚠️ 장면 문구 자유텍스트 필드("scene")는 넣지 말 것(2026-07-03 실측 기각 2중):
    # ①문구 대조 추론이 복합 문구에서 구조적 실패 ②자유 텍스트 필드가 요청 복창
    # 반복 루프를 재유발(초소형 출력 원칙 위반). EditBlock.scene 은 향후용 예비.
    # 모드 A: 이 블록 동안 읽을 내레이션 구절(없으면 빈 문자열 = 무음 블록).
    # 내레이션이 타임라인 주인 — 있으면 블록 길이는 구절 길이가 정한다(m5_tts.render).
    "narration": {"type": "string"},
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},     # 전역 한글 제목/자막. 없으면 빈 문자열.
        "blocks": {
            "type": "array",
            "items": {"type": "object", "properties": _BLOCK_PROPS, "required": ["select"]},
        },
    },
    "required": ["blocks"],
}


@dataclass
class EditBlock:
    """영상의 한 구간 연출 단위."""
    select: str = "all"          # dynamic|static|all
    target_dur: Optional[float] = None
    pace: str = "fast"           # fast|calm
    transition: str = "cut"      # cut|xfade (블록 내부 클립 잇기)
    subject: str = "full"        # foster=임보견 중심 크롭, full=전체
    zoom: str = "none"           # none|gradual (정적 구간 권장)
    speed: float = 1.0           # 재생속도 배율
    caption: str = ""            # 이 블록 동안 띄울 자막(빈값=없음)
    keywords: list = None        # 장면 키워드(사용자 표현) — 매칭+핀. 비면 전체 소스.
    scene: str = ""              # 장면 묘사 문구(요청 표현 그대로) — 추론 대조용(예비)
    narration: str = ""          # 모드 A: 이 블록 동안 읽을 내레이션 구절(빈값=무음 블록)
    # 저작 모드 전용(author.py): 소재 이름 직접 지정 — 구성 작가가 관찰 프로필을 보고
    # 골랐으므로 키워드 매칭 왕복 없이 그 소스만 쓴다(핀·예약 의미론 동일). 번역
    # 프롬프트(_BLOCK_PROPS)에는 없다 — interpret 가 이름을 지어내면 안 되므로.
    sources: list = None

    @classmethod
    def from_dict(cls, d: dict) -> "EditBlock":
        td = d.get("target_dur"); sp = d.get("speed")
        return cls(
            select=str(d.get("select", "all")),
            target_dur=(float(td) if td else None),
            pace=str(d.get("pace", "fast")),
            transition=str(d.get("transition", "cut")),
            subject=str(d.get("subject", "full")),
            zoom=str(d.get("zoom", "none")),
            speed=(float(sp) if sp else 1.0),
            caption=str(d.get("caption", "") or ""),
            keywords=[str(k) for k in (d.get("keywords") or [])],
            scene=str(d.get("scene", "") or ""),
            narration=str(d.get("narration", "") or ""),
            sources=[str(s) for s in (d.get("sources") or [])],
        )


@dataclass
class EditPlan:
    """전역 title + 순서 있는 블록들."""
    blocks: list[EditBlock] = field(default_factory=list)
    title: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "EditPlan":
        blocks = [EditBlock.from_dict(b) for b in d.get("blocks", [])]
        if not blocks:
            blocks = [EditBlock()]   # 빈 plan 방어
        return cls(blocks=blocks, title=str(d.get("title", "") or ""))
