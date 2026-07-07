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

from . import AUTO_POS, EditBlock, EditPlan, PlanText
from .layout import LEVEL3_MAX_CHARS, READ_CPS
from ..workspace import Workspace

MODEL = "gemma4:26b-a4b-it-q4_K_M"
AUTHOR_TEMP = 0.9    # 창작 다양성 — 사용자: "돌릴 때마다 달라도 상관없다"
MAX_BLOCKS = 6
DUR_MIN, DUR_MAX = 2.0, 12.0
MAX_TEXTS = 2        # 블록 걸침 카피 상한 — 동시에 떠 있는 글이 많으면 못 읽는다

# 저작 프롬프트의 자막 글자 예산(자/초) — 진짜 한계(READ_CPS)의 절반. LLM 은 글자
# 수를 못 세므로 예산은 마진이 본체다: 2배를 넘겨도 아직 읽을 수 있는 값을 준다
# (앞단 예산 = 생성 유도, 뒷단 layout.required_secs = 경고 위주 안전망 — 사용자
# 확정 2026-07-06 "가드는 앞에"). 숫자 임계는 내용 상수가 아니라 지시라 허용.
BUDGET_CPS = READ_CPS / 2

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
        # 복창을 걷어낸 잔여가 구두점뿐이면 통째로 비움(실측: caption='!').
        # strip 집합에 !? 를 넣으면 정상 자막의 문장부호까지 먹는다 — 잔여 검사로만.
        if not re.search(r"[가-힣a-zA-Z0-9]", t):
            t = ""
    return t


def _watch_echo(t: str, request: str, where: str) -> str:
    """복창 안전망 + 관측 — 발동은 프롬프트 오염 신호다(원인은 프롬프트에서 고친다).

    사용자 지적(2026-07-03): 소독으로 잔여물('!')을 다듬는 건 증상 치료 — 병인은
    '말투를 그대로 살려서' 류의 인용 초대 문구였다(프롬프트에서 제거). 이 안전망이
    자주 발동하면 프롬프트가 다시 오염된 것이니 경고를 남긴다.
    """
    req = request.strip()
    if t and req and req in t:
        print(f"     [저작] ⚠️ 요청문 복창 감지({where}) → 제거 — 프롬프트 점검 신호")
    return _clean_text(t, request)


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


def _record_lines(ws: Workspace, avail: list[str], profiles: dict) -> dict:
    """소재별 관찰 기록 한 줄 — 작가 입력(_records)과 자막 검수(_verify_captions)의
    단일 출처. 길이는 실측 관찰 — 없으면 작가가 2초짜리 영상을 두 블록에 배치하는
    실수를 한다(실측: 인트로·엔딩에 같은 2.3초 소재). 확정 장면 태그(사람+자동,
    데이터이지 상수 아님)도 합류 — 없으면 검수가 근거 있는 자막을 오탐한다
    (실측: 목줄+야외 기록만으로는 '산책' 자막이 지어냄 판정)."""
    from ..m4_action.observe import motion_summary, profile_text
    from .run import _probe_dur
    tags = ws.scene_tags()
    # 행동 관찰(behavior)은 저작·검수 기록에만 합류 — 매칭(profile_text) 합류는
    # 골든 재채점 5/7 회귀로 기각(observe.profile_text 주석, 2026-07-07 실측).
    return {n: (f"[{n}] 길이 {max(_probe_dur(str(ws.analysis(n))), 0):.0f}초 | "
                f"{profile_text(profiles[n], motion_summary(ws, n))}"
                + (f" | 행동: {profiles[n]['behavior']}"
                   if profiles[n].get("behavior") else "")
                + (f" | 확정 장면: {', '.join(sorted(tags[n]))}"
                   if tags.get(n) else ""))
            for n in avail}


def _records(ws: Workspace, avail: list[str], profiles: dict) -> str:
    return "\n".join(_record_lines(ws, avail, profiles).values())


def _schema(names: list[str], voice_choice: bool = False) -> dict:
    # 레벨 계약(2026-07-07): L2 위치(하단)·L3 자리/순간(피사체 옆·모션 피크)은
    # 렌더러 몫 — 저작 출력에서 위치·타이밍 필드를 제거(재량 축소 = 실패 모드 축소).
    block = {"type": "object", "properties": {
        "sources": {"type": "array", "items": {"enum": names}},
        "select": {"type": "string", "enum": ["dynamic", "static", "all", "묘기"]},
        "dur": {"type": "number"},
        "zoom": {"type": "string", "enum": ["none", "gradual"]},
        "caption": {"type": "string"},
        "caption_span": {"type": "array", "items": {"type": "number"}},
        "narration": {"type": "string"},
    }, "required": ["sources", "select", "dur", "caption"]}   # dur 필수 — 미기입 시
    # 블록이 pace 기본값(컷당 2초)으로 흘러 총 길이가 목표 미달(실측 12.2s)
    text = {"type": "object", "properties": {
        "text": {"type": "string"},
        "blocks": {"type": "array", "items": {"type": "integer"}},
    }, "required": ["text", "blocks"]}
    schema = {"type": "object",
              "properties": {"title": {"type": "string"},
                             "blocks": {"type": "array", "items": block},
                             "texts": {"type": "array", "items": text}},
              "required": ["blocks"]}
    if voice_choice:
        # 재량 TTS 는 *영상 단위* 장르 결정(2026-07-07 체셔 "위화감" — 내레이터는
        # 등장하는 순간 존재가 성립, 블록별 on/off 는 연출이 아니라 고장으로 읽힘).
        # 필수화 = dur 미기입 습성과 같은 방어.
        schema["properties"]["voice"] = {"type": "boolean"}
        schema["required"] = schema["required"] + ["voice"]
    return schema


def author_plan(request: str, ws: Workspace, names: list[str],
                narration: bool, voice_choice: bool = False) -> EditPlan | None:
    """요청 + 관찰 프로필 → 저작 EditPlan. 실패(빈 블록 등)면 None(호출부 폴백).

    voice_choice — 저작 재량 TTS(2026-07-07 사용자): TTS *무언급* 요청에서만
    (voice_discretion_allowed 게이트) 저작이 블록별로 "자막을 목소리로 읽을지"를
    고른다. 새 대본을 짓는 게 아니라 **화면 자막을 그대로 읽는다**(narration :=
    caption 결정론 복사 — 대본 발명 축 원천 차단). 유저가 TTS 를 언급하면(모드 A·
    부정 핀·카드) 이 재량은 완전히 꺼진다 — TTS 소유권 이진.
    """
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
        # ⚠️ 요청문 = 지시문(2026-07-03 사용자): "말투를 그대로 살려서" 류의 문구는
        # 요청문 인용 초대장이 된다(실측: caption 에 요청 전문 복창). 요청은 해석
        # 대상이고 자막·제목은 시청자에게 말하는 새 문장이라고 명시한다.
        "너는 강아지 영상의 구성 작가다. 아래 '고객 요청'은 너에게 주는 *지시문*"
        "이고, 요청문 자체는 영상에 들어갈 내용이 아니다 — 요청이 말하는 목적과 "
        "말투를 파악해 영상을 기획하라.\n\n"
        f"고객 요청: {request}\n\n"
        f"소재(영상별 관찰 기록 — 기계 측정이라 오류 가능):\n{records}\n\n"
        "이 소재들만으로 요청의 목적에 맞는 세로 숏폼(총 20~35초)을 기획하라. "
        "구성(흐름·순서·분위기)은 네가 정한다.\n"
        "화면 글은 3계층이다 — 제목 title(전체 1개, 상단), 설명 caption(장면당 "
        "한 문장, 하단), 감탄 texts(순간의 짧은 외침, 화면 중간). 계층을 섞지 "
        "마라: 문장은 caption 에만, 감탄은 texts 에만.\n"
        "시간 순서대로 블록 3~6개, 각 블록:\n"
        "- sources: 그 블록에 어울리는 소재 이름 1~3개(관찰 기록을 근거로 고른다)\n"
        "- select: 그 소재에서 쓸 구간 — 활발한 움직임=dynamic, 차분함=static, "
        "재주 장면=묘기, 무관=all\n"
        "- dur: 블록 길이(초, 3~10)\n"
        "- zoom: 얼굴·첫인상을 천천히 당겨 보여줄 블록만 gradual(차분한 구간에서), "
        "아니면 none\n"
        "- caption: 이 장면의 설명(Level 2) — 하단 자막 한 문장. 영상을 *보는 "
        "사람*에게 말하는 새 문장으로 써라(요청문을 옮겨 적지 마라). 강아지 "
        "이름과 고객의 말투는 따른다. 매 블록 반드시 채운다(빈 문자열 금지). "
        f"눈으로 읽는 글이다 — 블록 길이 1초당 {BUDGET_CPS:.0f}자를 넘기지 마라"
        f"(예: 5초 블록 ≤ {BUDGET_CPS * 5:.0f}자). 길면 문장을 줄여라. "
        "**사실 주장은 그 블록 소재의 관찰 기록이 보여주는 범위까지만** — 기록에 "
        "물건·배경이 보인다는 이유로 그와 관련된 사건·행동을 지어내지 마라. "
        "기록이 뒷받침하지 않으면 장면 서술 대신 강아지의 감정·매력을 말하라.\n"
        "- caption_span: 자막이 장면에 맞춰 떴다 사라지게 하고 싶으면 블록 길이 "
        "대비 [시작,끝] 0~1 비율(예: [0.2,0.8]), 내내 표시면 생략\n"
        + ("- narration: 그동안 목소리로 읽을 한 문장(자막과 같아도 된다)\n"
           if narration else "")
        + ("- 최상위 voice: 이 영상에 내레이션이 어울리는지의 *장르 결정* — "
           "true 면 모든 블록의 자막을 목소리가 그대로 읽고, false 면 목소리 "
           "없는 자막 영상이 된다. 새 문장을 짓는 게 아니다 — 읽는 것은 화면 "
           "자막 그 문장들이다. 블록별로 읽었다 말았다 할 수 없다(시청자에게 "
           "고장으로 들린다). 목소리를 쓰기로 했다면 화면 글은 더 절제하라.\n"
           if voice_choice else "")
        + "- title: 제목(Level 1) — 화면 상단에 박을 *영상 자체의* 짧은 제목"
        "(요청문을 설명하는 '~만들기' 같은 문구가 아니다), 필요 없으면 빈 문자열\n"
        "- (선택) 최상위 texts: 감탄(Level 3) — 장면의 *순간*에 터지는 외침·"
        "의성어. {text, blocks:[시작,끝 블록 번호]} 형식, 최대 "
        f"{MAX_TEXTS}개. **공백 제외 {LEVEL3_MAX_CHARS}자 이내 — 넘으면 통째로 "
        "버려진다. 문장을 쓰지 마라(문장은 caption 계층).** 등장하는 순간과 "
        "자리는 렌더러가 정한다(움직임이 터지는 프레임에, 강아지 곁에) — 너는 "
        "어느 장면(blocks)에서 무슨 감탄이 터질지만 정한다.\n"
        "자막·내레이션이 소재의 관찰 기록과 어긋나지 않게 하라.")
    # 재시도 조건 = JSON 실패 + 구조 무효(블록 0개 또는 자막·대본 전무 — 텍스트 0인
    # 소개 영상은 창작 다양성이 아니라 결함, 실측: 전 블록 caption 빈 값 복권).
    # 비결정은 의도이므로 재시도도 같은 온도(새 추첨).
    for _ in range(2):
        r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                        options={"temperature": AUTHOR_TEMP, "num_predict": 2048,
                                 "repeat_penalty": 1.3},
                        format=_schema(avail, voice_choice), think=False)
        try:
            raw = json.loads(r.message.content)
        except json.JSONDecodeError:
            continue
        plan = _to_plan(raw, request, avail, narration, allow_caption, voice_choice)
        if plan is not None:
            # 검수는 부가 안전망 — 실패해도 저작을 죽이지 않는다(안전한 저하).
            # 자막 + (모드 A) 저작이 쓴 내레이션 — 저작 소유 텍스트는 전부 검수.
            for field, on in (("caption", allow_caption), ("narration", narration)):
                if not on:
                    continue
                try:
                    _verify_captions(plan, request, ws, avail, profiles, field=field)
                except Exception as e:
                    print(f"     [검수] {field} 건너뜀({type(e).__name__}: {e})")
            return plan
    return None


_JUDGE_CONF = 0.70   # '지어냄' 판정 확신 하한(logprob 정규화) [잠정]


def _caption_claims(caption: str) -> list[str]:
    """자막의 *사실 주장* 추출 — 검수 1단(주장이 없으면 심판 불필요).

    단일 이진 판정은 기각 계보 2건(simple-demo3 3연속 사고의 교훈): 관대 기준은
    사물→행동 추론('식기→밥')과 조건절 숨김('밥 먹을 때*만큼은* 활발')을 통과시키고,
    엄격 기준은 감정·이름·동의어까지 오탐(12/21 실측). 부풀림은 문장 *구성* 속에
    숨는다 → 주장 단위로 분해해 하나씩 대조한다(추출은 기록을 안 보므로 판정
    오염 없음).
    """
    import ollama
    prompt = (
        "문장: " + caption + "\n이 문장이 단정하는 구체적 행동·사건·장소·상황을 "
        "짧은 구절로 전부 뽑아라 — 조건절·수식으로 스치듯 말한 것도 포함한다. "
        "감정·매력·이름·호칭 표현은 주장이 아니다. 없으면 빈 배열.")
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0, "num_predict": 128},
                    format={"type": "object",
                            "properties": {"claims": {"type": "array",
                                                      "items": {"type": "string"}}},
                            "required": ["claims"]}, think=False)
    try:
        return [str(c) for c in json.loads(r.message.content).get("claims") or []
                if str(c).strip()]
    except json.JSONDecodeError:
        return []


def _claim_supported(claim: str, record: str) -> bool:
    """주장 1개 ↔ 기록 대조 — 단일 토큰 + logprob(M4·모드판정의 검증된 레시피)."""
    import math
    import ollama
    prompt = (
        "관찰 기록: " + record + "\n주장: " + claim +
        "\n기록의 서술(또는 그 명백한 동의어·직접적 표현)이 이 주장을 뒷받침하면 "
        "'근거있음', 기록에 없는 행동·상황이면 '지어냄'. 물건이 보인다는 서술은 "
        "그 물건을 쓰는 행동의 근거가 아니다. 다른 말 없이 한 단어만 출력.")
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0, "num_predict": 4},
                    think=False, logprobs=True, top_logprobs=10)
    lp = getattr(r, "logprobs", None)
    if not lp:
        return True                     # 판정 불가 = 무혐의(안전한 저하)
    first = lp[0] if isinstance(lp[0], dict) else lp[0].model_dump()
    fab = ok = 0.0
    for c in first.get("top_logprobs", []):
        tok = (c["token"] or "").strip()
        if tok.startswith("지"):
            fab += math.exp(c["logprob"])
        elif tok.startswith("근"):
            ok += math.exp(c["logprob"])
    return not ((fab + ok) > 0 and fab / (fab + ok) >= _JUDGE_CONF)


def _caption_fabricated(caption: str, record: str,
                        _claims_cache: dict | None = None) -> bool:
    """자막 1개 사실 검수 = 추출(1단) + 주장별 대조(2단). 주장 0개면 무혐의."""
    if _claims_cache is not None and caption in _claims_cache:
        claims = _claims_cache[caption]
    else:
        claims = _caption_claims(caption)
        if _claims_cache is not None:
            _claims_cache[caption] = claims
    return any(not _claim_supported(c, record) for c in claims)


def _verify_captions(plan: EditPlan, request: str, ws: Workspace,
                     avail: list[str], profiles: dict, field: str = "caption",
                     only: set | None = None) -> None:
    """저작 텍스트 사실 검수 — 주장 분해 판정 + 걸린 블록만 증거 제한 재작성.

    병인(simple-demo3 실측 2026-07-07): 저작이 프로필의 사물 단서를 사건 서사로
    부풀림('식기 주변에서 움직임'→'밥 먹을 때가 제일 신나요', '문틈 사이로 손을
    향해 다가옴'→'문틈 사이로 살짝 나타난') + 그 자막이 다른 소재 블록에 얹힘.
    텍스트를 못 비우므로(L2=서사 등뼈) 드롭이 아니라 재작성. 검수 기준은 *그 블록
    소재의 기록*이라 부풀림과 소재-자막 결속 끊김을 한 심판이 잡는다.

    field — caption(모드 공통) | narration(모드 A 저작 작문; 자막 복사본은 제외).
    only — 검수 대상 블록 제한(부분 저작이 채운 블록만 — **유저 텍스트는 검수
    금지**, 유저는 자기 영상을 알고 우리 관찰이 틀릴 수 있다. 소유권 이진).
    """
    import ollama
    lines = _record_lines(ws, avail, profiles)
    entries = [(i, b, [s for s in (b.sources or []) if s in lines])
               for i, b in enumerate(plan.blocks)]
    entries = [(i, b, srcs) for i, b, srcs in entries
               if getattr(b, field) and srcs and (only is None or i in only)
               and not (field == "narration" and b.narration == b.caption)]
    if not entries:
        return
    by_idx = {i: (b, srcs) for i, b, srcs in entries}
    label = "하단 자막(설명)" if field == "caption" else "목소리로 읽을 내레이션 문장"

    claims_cache: dict = {}             # 같은 자막의 주장 추출은 소스 수와 무관 1회

    def _sweep(idxs) -> list[int]:
        # 소스별 각각 판정 — 텍스트는 그 블록의 *모든* 소스 클립 위에 뜨므로 하나라도
        # 못 받치면 그 클립에서 화면 불일치다(simple-demo3 2차 실측: '밥' 자막이
        # 0195 클립 위에). 합본 기록 한 줄 판정은 희석돼 통과시킴(3회 실측) → 기각.
        return [i for i in idxs
                if any(_caption_fabricated(getattr(by_idx[i][0], field), lines[s],
                                           claims_cache)
                       for s in by_idx[i][1])]

    def _rewrite(idxs, extra: str) -> None:
        rew = (
            f"다음 블록들의 {label}이 검수에서 '소재 기록에 없는 사실을 지어냄'"
            "으로 판정됐다. 각각 다시 써라.\n규칙: 그 블록 기록이 보여주는 것 "
            "안에서만 사실을 말하고, 기록에 물건·배경이 보인다는 이유로 사건을 "
            "지어내지 마라. 블록에 소재가 여러 개면 문장은 그 *모두*에 맞아야 "
            "한다 — 한 소재에만 있는 사실 대신 모두에 해당하는 것을 말하라. "
            "마땅한 사실이 없으면 장면 서술 대신 강아지의 감정·매력을 말하라. "
            f"{extra}한 문장, 1초당 {BUDGET_CPS:.0f}자 이내. 요청의 말투·호칭은 "
            "따르되 요청문 자체를 옮겨 적지 마라.\n"
            f"고객 요청: {request}\n" +
            "\n".join(f"블록{i} 기록: " + " / ".join(lines[s] for s in by_idx[i][1])
                      for i in idxs) +
            "\n블록 순서대로 captions 배열로만 출력.")
        r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": rew}],
                        options={"temperature": 0.4, "num_predict": 512,
                                 "repeat_penalty": 1.3},
                        format={"type": "object",
                                "properties": {"captions": {
                                    "type": "array", "items": {"type": "string"}}},
                                "required": ["captions"]}, think=False)
        try:
            caps = json.loads(r.message.content).get("captions") or []
        except json.JSONDecodeError:
            return
        _apply_rewrites(plan, dict(zip(idxs, caps)), request, field)

    # 라운드 1 = 증거 제한 재작성, 라운드 2 = 재판정 잔존만 감정 전용(사실 금지 —
    # 실측: 1차 재작성이 다른 한쪽 소스의 사실('문틈')로 도망). 그래도 남으면 경고.
    suspects = _sweep([i for i, _, _ in entries])
    for extra in ("", "이번에는 사실 서술을 아예 하지 말고 강아지의 감정·매력·"
                      "분위기만 말하라. "):
        if not suspects:
            return
        _rewrite(suspects, extra)
        suspects = _sweep(suspects)
    if suspects:
        print(f"     [검수] ⚠️ 재작성 2회 후에도 잔존: 블록 {suspects} — "
              "저작 증거 프롬프트 점검 신호")


def _apply_rewrites(plan: EditPlan, fixes: dict, request: str,
                    field: str = "caption") -> None:
    """검수 재작성 병합(결정론) — 소독 후 교체, 재량 TTS(자막 그대로 읽기) 동기화."""
    for i, new in fixes.items():
        if not (isinstance(i, int) and 0 <= i < len(plan.blocks)):
            continue
        b = plan.blocks[i]
        old = getattr(b, field)
        new = _clean_text(str(new or ""), request)
        if not new or new == old:
            continue
        print(f"     [검수] 블록{i} {field} 재작성: {old!r} → {new!r}")
        if field == "caption" and b.narration and b.narration == b.caption:
            b.narration = new               # 재량 TTS = 자막 그대로 읽기 동기화
        setattr(b, field, new)


def _to_plan(raw: dict, request: str, avail: list[str], narration: bool,
             allow_caption: bool = True, voice_choice: bool = False) -> EditPlan | None:
    """저작 응답 → 소독된 EditPlan. 구조 무효면 None(호출부가 재추첨)."""
    blocks = []
    for d in (raw.get("blocks") or [])[:MAX_BLOCKS]:
        srcs = [s for s in dict.fromkeys(d.get("sources") or []) if s in avail]
        dur = float(d.get("dur") or 0)
        b = EditBlock.from_dict({
            "select": d.get("select", "all"),
            "target_dur": (min(max(dur, DUR_MIN), DUR_MAX) if dur > 0 else None),
            "zoom": d.get("zoom", "none"),
            "caption": (_watch_echo(d.get("caption", ""), request, "caption")
                        if allow_caption else ""),
            # L2 위치는 계약(하단 고정) — 저작 출력에 위치 필드가 없다(레벨 기획).
            "caption_span": d.get("caption_span"),
            "narration": (_watch_echo(d.get("narration", ""), request, "narration")
                          if narration else ""),
            "sources": srcs,
        })
        # 결정론 강제: 저작은 speed 를 못 만진다(배속은 "요청이 명시할 때만" 가드),
        # 줌 블록은 강아지 중심(기존 클로즈업 귀속과 같은 구조 매핑).
        b.speed = 1.0
        if b.zoom == "gradual":
            b.subject = "foster"
        # 읽는 자막(내레이션=자막)은 블록 내내 표시 — 저작 span 이 자막을 발화보다
        # ~1초 늦춰 AV 스큐(simple-demo3 실측 2026-07-07). 동기가 저작 재량에 우선.
        if b.narration and b.narration == b.caption:
            b.caption_span = None
        blocks.append(b)
    blocks = [b for b in blocks if b.sources or b.select != "all" or b.caption or b.narration]
    if not blocks:
        return None
    # 재량 TTS = *영상 단위* 장르 결정 — 읽으면 자막 있는 모든 블록, 아니면 전부
    # 침묵(2026-07-07 체셔 "위화감": 블록별 on/off 는 내레이터 존재의 일관성을 깨
    # 고장으로 들린다). 읽기는 자막 결정론 복사(대본 발명 축 없음) + span 동기.
    if voice_choice and raw.get("voice"):
        for b in blocks:
            if b.caption:
                b.narration = b.caption
                b.caption_span = None
    title = _watch_echo(str(raw.get("title") or ""), request, "title")
    # 감탄(L3, texts) 소독 — 복창 감시 + 인덱스 클램프 + 상한 + **글자 계약**:
    # 공백 제외 LEVEL3_MAX_CHARS 초과는 자르지 않고 통째로 드롭(자르면 뜻이
    # 깨진다 — 레벨 기획 2026-07-07). 자막 거부 요청이면 감탄도 텍스트이므로
    # 통째로 버린다(부정 핀 우선). 자리·순간 필드는 스키마에 없다(렌더러 몫).
    texts = []
    if allow_caption:
        for d in (raw.get("texts") or [])[:MAX_TEXTS]:
            if not isinstance(d, dict):
                continue
            t = PlanText.from_dict(
                dict(d, text=_watch_echo(str(d.get("text") or ""), request, "texts")))
            if not (t.text and t.blocks and t.blocks[0] < len(blocks)):
                continue
            if len(t.text.replace(" ", "")) > LEVEL3_MAX_CHARS:
                print(f"     [저작] L3 감탄 {len(t.text)}자 > 계약 "
                      f"{LEVEL3_MAX_CHARS}자 → 드롭: {t.text[:20]!r}")
                continue
            t.blocks = [min(t.blocks[0], len(blocks) - 1),
                        min(t.blocks[1], len(blocks) - 1)]
            t.pos = AUTO_POS                   # 저작 감탄의 자리는 전부 렌더러 몫
            texts.append(t)
    # 목소리를 쓰면 화면 글은 절제(사용자 2026-07-07) — 감탄은 최대 1개.
    if voice_choice and any(b.narration for b in blocks):
        texts = texts[:1]
    # 텍스트 전무 = 결함(재추첨) — 단 고객이 자막을 거부했으면 무자막이 곧 의도.
    if allow_caption and not any(b.caption or b.narration for b in blocks) and not texts:
        return None
    return EditPlan(blocks=blocks, texts=texts, title=title)


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
    # 텍스트 소유권 이진(2026-07-03 사용자): 요청에 텍스트 번인 지시가 하나라도
    # 있으면(= 번역 결과에 유저 자막 존재) 자막은 통째로 유저 소유 — 저작은 빈
    # 블록의 자막도 채우지 않는다("지시가 하나라도 있으면 저작은 아예 없음").
    # 지시가 전혀 없을 때만 자막·위치·표시구간이 저작 재량이 된다.
    user_texted = any((b.caption or "").strip() for b in plan.blocks)
    allow_caption = not caption_forbidden(request) and not user_texted
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
        "항목만 비어 있다. 아래 '고객 요청'은 지시문이고 요청문 자체는 영상에 "
        "들어갈 내용이 아니다.\n"
        f"고객 요청: {request}\n\n확정된 구성(순서·내용을 바꿀 수 없다):\n"
        + "\n".join(skel) +
        "\n\n소재(영상별 관찰 기록 — 기계 측정이라 오류 가능):\n"
        + _records(ws, avail, profiles) +
        "\n\n'(채울 것)' 표시된 빈 항목만 채워라. 블록 순서 그대로 blocks 배열로 "
        "출력하고, 이미 값이 있는 항목의 출력은 무시된다.\n"
        "- caption: 자막 한 문장 — 영상을 *보는 사람*에게 말하는 새 "
        "문장으로(요청문을 옮겨 적지 마라). 강아지 이름·고객의 말투는 따른다. "
        f"눈으로 읽는 글이니 블록 길이 1초당 {BUDGET_CPS:.0f}자를 넘기지 마라. "
        "사실 주장은 관찰 기록이 보여주는 범위까지만 — 기록이 뒷받침하지 않는 "
        "사건·행동을 지어내지 마라(불확실하면 감정·분위기로).\n"
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
        filled = _merge_fill(plan, raw, request, avail, narration, allow_caption)
        if filled:
            # 부분 저작이 채운 자막도 저작 소유 → 같은 검수. only= 가 소유권 경계 —
            # 유저가 쓴 자막은 검수 대상이 아니다(유저는 자기 영상을 안다).
            try:
                _verify_captions(plan, request, ws, avail, profiles,
                                 only=set(filled))
            except Exception as e:
                print(f"     [검수] 건너뜀({type(e).__name__}: {e})")
        return plan
    return plan


def _merge_fill(plan: EditPlan, raw: dict, request: str, avail: list[str],
                narration: bool, allow_caption: bool) -> list[int]:
    """결정론 병합 — *빈 슬롯에만* 기입. 유저 명시 필드는 저작 출력에서 읽지 않는다.
    반환 = 저작이 자막을 채운 블록 인덱스(검수 대상 경계)."""
    filled_caps: list[int] = []
    for i, (b, d) in enumerate(zip(plan.blocks, raw.get("blocks") or [])):
        filled = []
        if not (b.keywords or b.sources):
            srcs = [s for s in dict.fromkeys(d.get("sources") or []) if s in avail]
            if srcs:
                b.sources = srcs
                filled.append(f"sources={srcs}")
        if not narration:
            if allow_caption and not b.caption:
                cap = _watch_echo(d.get("caption", ""), request, "caption")
                if cap:
                    b.caption = cap        # L2 위치는 계약(하단) — pos 필드 없음
                    b.caption_span = EditBlock._clean_span(d.get("caption_span"))
                    filled.append(f"caption={cap!r}")
                    filled_caps.append(i)
            if not b.target_dur:
                dur = float(d.get("dur") or 0)
                if dur > 0:
                    b.target_dur = min(max(dur, DUR_MIN), DUR_MAX)
                    filled.append(f"dur={b.target_dur:.0f}")
        if filled:
            print(f"     블록{i} ← {' '.join(filled)}")
    return filled_caps
