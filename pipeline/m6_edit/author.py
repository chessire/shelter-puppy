"""저작(authoring) — 구조 없는 한 줄 요청을 관찰 프로필 기반 구성으로.

배경(2026-07-03 사용자 결정): "우리 토리 소개 영상 만들어줘" 같은 목적 문장은 구조
정보가 0비트라 번역기(interpret)는 기본값 블록 1개 = 소스 나열밖에 못 만든다
(simple-demo 실측: 9소스 × 한 컷 = "영상의 나열"). 구조의 원천은 요청이 아니라
**소재(관찰 프로필) + 목적(요청의 느낌)** — Gemma 가 구성 작가로서 소재를 보고
블록 시퀀스·자막·대본을 창작한다. 모드 A 의 기존 작문("대본 없이 취지만 줬으면
새로 써라")은 소재를 못 보는 장님 작문이었다 — 이 경로가 대체한다.

사용자 확정 원칙:
  · 완전히 주관적이어도, 돌릴 때마다 달라도 됨 — AUTHOR_TEMP 상향, 파이프라인에서
    유일하게 *의도된* 비결정. --rerender 가 "구성 복권"이 된다.
  · 환각 허용 — 단 요청의 말투·호칭·목적(느낌)은 보존한다.
  · **프롬프트를 제외한 상수 문자열 절대 금지** — 연출 템플릿·장면 어휘·자막 문구를
    코드에 두지 않는다. 저작의 재료는 요청 원문 + 관찰 프로필뿐이고, 구성(흐름·순서)
    은 Gemma 가 정한다.
실행은 그대로 결정론 — 저작 출력은 소독(이름 enum 재검증·길이 클램프·speed 강제)
후 기존 EditPlan 으로 흘러 M5/M6 기계를 탄다("LLM 은 해석·창작만, 픽셀은 결정론").
"""

from __future__ import annotations

import json
import re

from . import EditBlock, EditPlan
from ..workspace import Workspace

MODEL = "gemma4:26b-a4b-it-q4_K_M"
AUTHOR_TEMP = 0.9    # 창작 다양성 — 사용자: "돌릴 때마다 달라도 상관없다"
MAX_BLOCKS = 6
DUR_MIN, DUR_MAX = 2.0, 12.0

# 이모지·기호 — PIL 기본 한글 폰트가 못 그려 화면에 두부(□)로 박힌다(렌더러 제약의
# 계기 보정, 어휘 아님). 실측: 저작 title 에 🐾 유출.
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF️‍]")


def _clean_text(t: str, request: str) -> str:
    """저작 텍스트 소독 — 이모지 제거 + 요청 원문 복창 제거(런타임 문자열 대조).

    실측: 마지막 블록 caption 에 요청 전문이 그대로 유출("우리 토리 소개 영상
    만들어줘 (가족이 되어주세요!)") — 복창 부분만 걷어내고 창작 부분은 살린다.
    """
    t = _EMOJI.sub("", t or "").strip()
    req = request.strip()
    if t and req and req in t:
        t = t.replace(req, "").strip(" ()[]{}·—–\-,:;'\"")
    return t


def is_unstructured(plan: EditPlan) -> bool:
    """요청이 구성 정보를 안 줬다는 신호 — 번역 결과가 '기본값 블록 1개'.

    문자열 검사가 아니라 구조 검사(요청에 순서·장면·자막·길이·연출 지시가 하나라도
    있었으면 번역기가 어느 필드든 채웠을 것). select 지정(노는 것만 등)도 구조로 본다.
    """
    if len(plan.blocks) != 1 or plan.title:
        return False
    b = plan.blocks[0]
    return (not b.keywords and not b.caption and not b.narration
            and not b.target_dur and b.zoom == "none" and b.speed == 1.0
            and b.select == "all")


def script_invented(plan: EditPlan, request: str) -> bool:
    """모드 A 대본이 요청 원문에 없음 = Gemma 장님 작문(소재 모름) → 저작으로 교체.

    attribute_directives 와 같은 앵커 원리의 역이용 — 고객이 대본을 줬으면 문장이
    원문 안에 그대로 있다(런타임 문자열 대조, 상수 아님).
    """
    sents = [b.narration for b in plan.blocks if (b.narration or "").strip()]
    return bool(sents) and all(request.find(s) < 0 for s in sents)


def _schema(names: list[str]) -> dict:
    block = {"type": "object", "properties": {
        "sources": {"type": "array", "items": {"enum": names}},
        "select": {"type": "string", "enum": ["dynamic", "static", "all", "묘기"]},
        "dur": {"type": "number"},
        "zoom": {"type": "string", "enum": ["none", "gradual"]},
        "caption": {"type": "string"},
        "narration": {"type": "string"},
    }, "required": ["sources", "select", "dur", "caption"]}   # dur 필수 — 미기입 시
    # 블록이 pace 기본값(컷당 2초)으로 흘러 총 길이가 목표 미달(실측 12.2s)
    return {"type": "object",
            "properties": {"title": {"type": "string"},
                           "blocks": {"type": "array", "items": block}},
            "required": ["blocks"]}


def author_plan(request: str, ws: Workspace, names: list[str],
                narration: bool) -> EditPlan | None:
    """요청 + 관찰 프로필 → 저작 EditPlan. 실패(빈 블록 등)면 None(호출부 폴백)."""
    import ollama
    from ..m4_action.observe import ensure_profiles, motion_summary, profile_text
    from .run import _probe_dur
    profiles = ensure_profiles(ws, names)
    if not profiles:
        return None
    avail = [n for n in names if n in profiles]
    # 소재 길이는 실측 관찰 — 없으면 작가가 2초짜리 영상을 두 블록에 배치하는
    # 실수를 한다(실측: 인트로·엔딩에 같은 2.3초 소재).
    records = "\n".join(
        f"[{n}] 길이 {max(_probe_dur(str(ws.analysis(n))), 0):.0f}초 | "
        f"{profile_text(profiles[n], motion_summary(ws, n))}"
        for n in avail)
    prompt = (
        # ⚠️ 프레이밍 상수 금지(2026-07-03 사용자): '임시보호/입양/홍보' 같은 목적을
        # 프롬프트가 주입하면 요청에 없는 문구("Adopt Me!"·"[임시보호]")가 제목·자막에
        # 샌다(실측). 영상의 목적·톤은 오직 고객 요청에서 온다.
        "너는 강아지 영상의 구성 작가다. 아래는 고객 요청과, 쓸 수 "
        "있는 소재(영상별 관찰 기록 — 기계 측정이라 오류 가능)다.\n\n"
        f"고객 요청: {request}\n\n소재:\n{records}\n\n"
        "이 소재들만으로 요청의 목적에 맞는 세로 숏폼(총 20~35초)을 기획하라. "
        "구성(흐름·순서·분위기)은 네가 정한다. 시간 순서대로 블록 3~6개, 각 블록:\n"
        "- sources: 그 블록에 어울리는 소재 이름 1~3개(관찰 기록을 근거로 고른다)\n"
        "- select: 그 소재에서 쓸 구간 — 활발한 움직임=dynamic, 차분함=static, "
        "재주 장면=묘기, 무관=all\n"
        "- dur: 블록 길이(초, 3~10)\n"
        "- zoom: 얼굴·첫인상을 천천히 당겨 보여줄 블록만 gradual(차분한 구간에서), "
        "아니면 none\n"
        "- caption: 화면 하단 자막 한 문장 — 고객 요청의 말투·호칭·이름을 그대로 "
        "살려서. 매 블록 반드시 채운다(빈 문자열 금지).\n"
        "- narration: 그동안 목소리로 읽을 한 문장(자막과 같아도 된다)\n"
        "- title: 화면에 박을 *영상 자체의* 짧은 제목(요청문을 설명하는 '~만들기' "
        "같은 문구가 아니다), 필요 없으면 빈 문자열\n"
        "자막·내레이션이 소재의 관찰 기록과 어긋나지 않게 하라.")
    # 재시도 조건 = JSON 실패 + 구조 무효(블록 0개 또는 자막·대본 전무 — 텍스트 0인
    # 소개 영상은 창작 다양성이 아니라 결함, 실측: 전 블록 caption 빈 값 복권).
    # 비결정은 의도이므로 재시도도 같은 온도(새 추첨).
    for _ in range(2):
        r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                        options={"temperature": AUTHOR_TEMP, "num_predict": 2048,
                                 "repeat_penalty": 1.3},
                        format=_schema(avail), think=False)
        try:
            raw = json.loads(r.message.content)
        except json.JSONDecodeError:
            continue
        plan = _to_plan(raw, request, avail, narration)
        if plan is not None:
            return plan
    return None


def _to_plan(raw: dict, request: str, avail: list[str],
             narration: bool) -> EditPlan | None:
    """저작 응답 → 소독된 EditPlan. 구조 무효면 None(호출부가 재추첨)."""
    blocks = []
    for d in (raw.get("blocks") or [])[:MAX_BLOCKS]:
        srcs = [s for s in dict.fromkeys(d.get("sources") or []) if s in avail]
        dur = float(d.get("dur") or 0)
        b = EditBlock.from_dict({
            "select": d.get("select", "all"),
            "target_dur": (min(max(dur, DUR_MIN), DUR_MAX) if dur > 0 else None),
            "zoom": d.get("zoom", "none"),
            "caption": _clean_text(d.get("caption", ""), request),
            "narration": (_clean_text(d.get("narration", ""), request)
                          if narration else ""),
            "sources": srcs,
        })
        # 결정론 강제: 저작은 speed 를 못 만진다(배속은 "요청이 명시할 때만" 가드),
        # 줌 블록은 강아지 중심(기존 클로즈업 귀속과 같은 구조 매핑).
        b.speed = 1.0
        if b.zoom == "gradual":
            b.subject = "foster"
        blocks.append(b)
    blocks = [b for b in blocks if b.sources or b.select != "all" or b.caption or b.narration]
    if not blocks or not any(b.caption or b.narration for b in blocks):
        return None
    return EditPlan(blocks=blocks, title=_clean_text(str(raw.get("title") or ""), request))
