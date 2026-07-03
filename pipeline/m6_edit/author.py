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


# 부정 핀 — 자막 거부 명시(결정론, m5_tts._EDIT_PINS 와 같은 결: 부정을 먼저 본다).
# 빈 caption 이 "생각 안 함"(→채움)인지 "원치 않음"(→금지)인지는 요청만이 안다.
_NO_TEXT_PINS = ("자막 없이", "자막없이", "텍스트 없이", "텍스트없이",
                 "글자 없이", "글자없이", "자막은 빼", "자막 빼")


def caption_forbidden(request: str) -> bool:
    low = request.lower()
    return any(p in low for p in _NO_TEXT_PINS)


def _records(ws: Workspace, avail: list[str], profiles: dict) -> str:
    """소재 관찰 기록(작가 입력). 길이는 실측 관찰 — 없으면 작가가 2초짜리 영상을
    두 블록에 배치하는 실수를 한다(실측: 인트로·엔딩에 같은 2.3초 소재)."""
    from ..m4_action.observe import motion_summary, profile_text
    from .run import _probe_dur
    return "\n".join(
        f"[{n}] 길이 {max(_probe_dur(str(ws.analysis(n))), 0):.0f}초 | "
        f"{profile_text(profiles[n], motion_summary(ws, n))}"
        for n in avail)


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
    from ..m4_action.observe import ensure_profiles
    profiles = ensure_profiles(ws, names)
    if not profiles:
        return None
    avail = [n for n in names if n in profiles]
    records = _records(ws, avail, profiles)
    allow_caption = not caption_forbidden(request)
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
        plan = _to_plan(raw, request, avail, narration, allow_caption)
        if plan is not None:
            return plan
    return None


def _to_plan(raw: dict, request: str, avail: list[str], narration: bool,
             allow_caption: bool = True) -> EditPlan | None:
    """저작 응답 → 소독된 EditPlan. 구조 무효면 None(호출부가 재추첨)."""
    blocks = []
    for d in (raw.get("blocks") or [])[:MAX_BLOCKS]:
        srcs = [s for s in dict.fromkeys(d.get("sources") or []) if s in avail]
        dur = float(d.get("dur") or 0)
        b = EditBlock.from_dict({
            "select": d.get("select", "all"),
            "target_dur": (min(max(dur, DUR_MIN), DUR_MAX) if dur > 0 else None),
            "zoom": d.get("zoom", "none"),
            "caption": (_clean_text(d.get("caption", ""), request)
                        if allow_caption else ""),
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
    if not blocks:
        return None
    # 텍스트 전무 = 결함(재추첨) — 단 고객이 자막을 거부했으면 무자막이 곧 의도.
    if allow_caption and not any(b.caption or b.narration for b in blocks):
        return None
    return EditPlan(blocks=blocks, title=_clean_text(str(raw.get("title") or ""), request))


# --------------------------------------------------------------------------- #
# 부분 저작 — 유저 뼈대의 빈 필드만 채움 (그라디언트 요청, 2026-07-03 설계)
# --------------------------------------------------------------------------- #
# "구조 소유권은 이진, 빈칸 채움이 그라디언트": 요청 디테일은 천차만별이지만
# (한 줄 목적 ~ 풀 스펙), 번역 뼈대의 *빈 필드 수*가 곧 그 그라디언트다. 병합은
# 결정론 — 저작 출력에서 유저 명시 필드를 아예 읽지 않으므로 구조적으로 불변
# (대본 삼킴·요청 복창 사고 계보의 재발 통로 차단).
# 모드 A 는 sources 만 채운다 — 내레이션이 타임라인 주인이라 dur 을 채우면 무음이
# 늘어지고(렌더가 max(dur, 구절길이)), 자막은 wants_caption 관례가 담당.

def field_gaps(plan: EditPlan, narration: bool, allow_caption: bool = True) -> list[str]:
    """블록별 빈 필드 스캔(결정론). 반환 = 갭 요약 리스트(로그용), 비면 갭 없음."""
    gaps = []
    for i, b in enumerate(plan.blocks):
        need = []
        if not (b.keywords or b.sources):
            need.append("sources")
        if not narration:
            if allow_caption and not b.caption:
                need.append("caption")
            if not b.target_dur:
                need.append("dur")
        if need:
            gaps.append(f"블록{i}:{'+'.join(need)}")
    return gaps


def fill_plan(request: str, ws: Workspace, names: list[str],
              plan: EditPlan, narration: bool) -> EditPlan:
    """유저가 정한 구성(뼈대)은 그대로, 빈 필드만 저작으로 채운다.

    갭이 없으면 LLM 0호출로 plan 그대로(풀 스펙 요청 = 저작 기여 0으로 수렴).
    채움 실패(JSON 2회)도 plan 그대로 — 기본값 렌더가 폴백(안전한 저하).
    """
    allow_caption = not caption_forbidden(request)
    gaps = field_gaps(plan, narration, allow_caption)
    if not gaps:
        return plan
    import ollama
    from ..m4_action.observe import ensure_profiles
    profiles = ensure_profiles(ws, names)
    avail = [n for n in names if n in profiles]
    if not avail:
        return plan
    print(f"[저작] 부분 채움 — 빈 필드 {gaps}")

    skel = []
    for i, b in enumerate(plan.blocks):
        parts = [f"select={b.select}",
                 f"장면 키워드={', '.join(b.keywords) if b.keywords else '(없음→sources 채울 것)'}",
                 (f"자막={b.caption!r}" if b.caption
                  else ("자막=(채울 것)" if allow_caption and not narration else "자막=(없음)")),
                 (f"길이={b.target_dur:.0f}초" if b.target_dur
                  else ("길이=(채울 것)" if not narration else "길이=(내레이션이 정함)"))]
        if b.narration:
            parts.append(f"내레이션={b.narration!r}")
        skel.append(f"블록{i}: " + " · ".join(parts))

    # required 로 전 키 강제 — 안 그러면 Gemma 가 dur 을 건너뛴다(전체 저작에서
    # 12.2s, 부분 저작에서 13.5s 실측 재발). 갭 아닌 필드 출력은 병합이 무시한다.
    fill_block = {"type": "object", "properties": {
        "sources": {"type": "array", "items": {"enum": avail}},
        "dur": {"type": "number"},
        "caption": {"type": "string"}, "narration": {"type": "string"}},
        "required": ["sources", "dur", "caption"]}
    schema = {"type": "object",
              "properties": {"blocks": {"type": "array", "items": fill_block}},
              "required": ["blocks"]}
    prompt = (
        "너는 강아지 영상의 구성 작가다. 고객이 이미 영상 구성을 정했고, 일부 "
        "항목만 비어 있다.\n"
        f"고객 요청: {request}\n\n확정된 구성(순서·내용을 바꿀 수 없다):\n"
        + "\n".join(skel) +
        "\n\n소재(영상별 관찰 기록 — 기계 측정이라 오류 가능):\n"
        + _records(ws, avail, profiles) +
        "\n\n'(채울 것)' 표시된 빈 항목만 채워라. 블록 순서 그대로 blocks 배열로 "
        "출력하고, 이미 값이 있는 항목의 출력은 무시된다.\n"
        "- caption: 화면 하단 자막 한 문장 — 고객 요청의 말투·호칭·이름 그대로\n"
        "- dur: 그 블록 길이(초, 3~10)\n"
        "- sources: 키워드 없는 블록만 — 그 장면에 어울리는 소재 이름 1~3개"
        "(관찰 기록을 근거로)")
    for _ in range(2):
        r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                        options={"temperature": AUTHOR_TEMP, "num_predict": 1024,
                                 "repeat_penalty": 1.3},
                        format=schema, think=False)
        try:
            raw = json.loads(r.message.content)
        except json.JSONDecodeError:
            continue
        _merge_fill(plan, raw, request, avail, narration, allow_caption)
        return plan
    return plan


def _merge_fill(plan: EditPlan, raw: dict, request: str, avail: list[str],
                narration: bool, allow_caption: bool) -> None:
    """결정론 병합 — *빈 슬롯에만* 기입. 유저 명시 필드는 저작 출력에서 읽지 않는다."""
    for i, (b, d) in enumerate(zip(plan.blocks, raw.get("blocks") or [])):
        filled = []
        if not (b.keywords or b.sources):
            srcs = [s for s in dict.fromkeys(d.get("sources") or []) if s in avail]
            if srcs:
                b.sources = srcs
                filled.append(f"sources={srcs}")
        if not narration:
            if allow_caption and not b.caption:
                cap = _clean_text(d.get("caption", ""), request)
                if cap:
                    b.caption = cap
                    filled.append(f"caption={cap!r}")
            if not b.target_dur:
                dur = float(d.get("dur") or 0)
                if dur > 0:
                    b.target_dur = min(max(dur, DUR_MIN), DUR_MAX)
                    filled.append(f"dur={b.target_dur:.0f}")
        if filled:
            print(f"     블록{i} ← {' '.join(filled)}")
