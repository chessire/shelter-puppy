"""모드 판정 + 모드 A 대본 분해 — Gemma 는 해석만, 확정은 3단 구조.

모드 판정(2026-07-02 설계): TTS 를 넣을지는 텍스트 의도 분류 = 순수 의미 해석이라
Gemma 적합(밤 크롭 실패와 달리 입력에 정보가 있는 축). 단 이진 판정을 통째로 맡기지
않고 3단으로 쌓는다:
  ① 명시 키워드 = 결정론 핀 ("리터럴 값은 결정론 파싱" 원칙 — 부정 핀을 먼저 본다:
     "음성 없이"에는 "음성"이 포함되므로)
  ② 나머지 = Gemma logprob 2지선다 (M4 검증 레시피: think=False + 첫 토큰 top_logprobs)
  ③ 애매(저확신) = uncertain — 억지로 확정하지 않고 카드 후보("자막인지 음성인지")

대본 분해(모드 A): 기존 EditPlan 스키마에 narration 필드를 얹어 두 모드를 통일 —
블록 = 장면 연출 + 그동안 읽을 구절. 순서·장면 매칭·자막은 모드 B 기계를 그대로 탄다.
"""

from __future__ import annotations

import json
import math
import re

from ..m6_edit import _BLOCK_PROPS, EditBlock, EditPlan

MODEL = "gemma4:26b-a4b-it-q4_K_M"
MODE_CONF = 0.70   # 2지선다 확신 하한 — 미만이면 uncertain [잠정]

# 부정 핀(편집만)을 긍정 핀보다 먼저 검사한다.
_EDIT_PINS = ("음성 없이", "목소리 없이", "내레이션 없이", "나레이션 없이",
              "무음으로", "소리 없이", "자막만")
_NARR_PINS = ("내레이션", "나레이션", "tts", "목소리", "음성", "보이스",
              "읽어줘", "읽어 줘", "대본", "멘트", "성우", "말해줘", "말해 줘")


def pin_mode(request: str) -> str | None:
    """결정론 키워드 핀. 매칭 없으면 None → Gemma 로 넘어간다."""
    low = request.lower()
    if any(p in low for p in _EDIT_PINS):
        return "edit"
    if any(p in low for p in _NARR_PINS):
        return "narration"
    return None


def decide_mode(request: str) -> tuple[str, float]:
    """(mode, confidence). mode ∈ narration|edit|uncertain."""
    pinned = pin_mode(request)
    if pinned:
        return pinned, 1.0

    import ollama
    prompt = (
        "강아지 영상 편집 요청이다. 고객이 *음성 내레이션(목소리로 읽어주는 것)*을 "
        "원하는지 판정해라. 화면에 글자만 띄우는 자막은 내레이션이 아니다.\n"
        "다른 말 없이 한 단어만 출력 — 원하면 '음성', 아니면 '편집'.\n요청: " + request)
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0, "num_predict": 4},
                    think=False, logprobs=True, top_logprobs=10)
    lp = r.logprobs
    if not lp:
        return "uncertain", 0.0
    first = lp[0] if isinstance(lp[0], dict) else lp[0].model_dump()
    probs = {"narration": 0.0, "edit": 0.0}
    for cand in first.get("top_logprobs", []):
        tok = (cand["token"] or "").strip()
        if tok.startswith("음"):
            probs["narration"] += math.exp(cand["logprob"])
        elif tok.startswith("편"):
            probs["edit"] += math.exp(cand["logprob"])
    total = sum(probs.values())
    if total <= 0:
        return "uncertain", 0.0
    mode = max(probs, key=probs.get)
    conf = probs[mode] / total
    return (mode, conf) if conf >= MODE_CONF else ("uncertain", conf)


def _chat_json(prompt: str, schema: dict, num_predict: int = 1024) -> dict:
    """format 강제 JSON 호출 + 실측 실패 2종 방어.

    ① 생성 한도로 JSON 잘림 → num_predict 명시(한도 = 실패 바운드)
    ② temperature 0 탐욕 디코딩이 긴 문자열 안에서 반복 루프(요청 꼬리 무한 복창,
       실측 3768자) → repeat_penalty 로 탈출, 재시도는 온도를 올려 경로를 바꾼다.
    """
    import ollama
    for attempt, temp in enumerate((0, 0.3)):
        r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                        options={"temperature": temp, "num_predict": num_predict,
                                 "repeat_penalty": 1.3},
                        format=schema, think=False)
        try:
            return json.loads(r.message.content)
        except json.JSONDecodeError as e:
            if attempt:
                raise RuntimeError(f"Gemma JSON 파싱 2회 실패: {e}\n"
                                   f"응답 꼬리: …{r.message.content[-120:]!r}") from e


_SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"},
                   "sentences": {"type": "array", "items": {"type": "string"}}},
    "required": ["sentences"],
}

# 장면 속성만(자유 텍스트 최소) — narration 은 여기 없다. 단일 호출로 대본+장면을 다
# 시키면 Gemma 가 대본 전체를 한 블록에 삼키거나 문장을 중복하는 실측 실패 → 2단 분리.
# 배열 일괄 호출도 실측 실패(블록 1개만 + keywords 에 프롬프트 조각 유출) →
# M4 크롭당 1호출처럼 *문장당 1호출*(초소형 출력이 제일 강건).
_SCENE_PROPS = {k: v for k, v in _BLOCK_PROPS.items() if k != "narration"}
_SCENE_SCHEMA = {"type": "object", "properties": _SCENE_PROPS, "required": ["select"]}

_KW_OK = re.compile(r"^[가-힣a-zA-Z0-9 ]{1,10}$")
# 스키마 enum 단어가 keywords 로 새는 실측 유출(['all'] 등) 차단.
_KW_BLOCK = {"all", "dynamic", "static", "foster", "full", "none",
             "cut", "xfade", "fast", "calm", "gradual", "묘기"}


def _clean_keywords(kws: list) -> list:
    """키워드 결정론 소독 — LLM 이 흘린 프롬프트 조각(콜론·백틱·장문·enum 단어) 차단."""
    out = [k.strip() for k in kws if _KW_OK.match(k.strip())]
    return [k for k in out if k.lower() not in _KW_BLOCK]


def _match_scene(sentence: str, request: str) -> EditBlock:
    d = _chat_json(
        "강아지 입양영상 편집기다. 이 내레이션 문장이 나오는 동안 보여줄 장면 속성을 정해라.\n"
        "- select: 활발/노는 장면=dynamic, 잔잔/앉은/쉬는=static, 무관=all, "
        "하이파이브·손·빵야·구르기 같은 재주=묘기\n"
        "- keywords: *이 문장 자체*에 나오는 장소/상황만, 각각 한 단어"
        "(카페·산책·비·밤·하이파이브 등). 강아지 이름·나이·품종 같은 내용 단어 금지. "
        "편집 지시는 이 문장을 콕 집어 장면을 지정할 때만 반영. "
        "확실치 않으면 빈 배열(빈 배열이 안전하다).\n"
        "- target_dur: 편집 지시가 *이 문장 장면*의 초를 명시한 경우만. 아니면 0.\n"
        "- zoom: 클로즈업/확대/얼굴을 보여줘 지시면 gradual, 아니면 none. "
        "pace: 빠른컷=fast, 잔잔=calm.\n"
        "- speed: 지시가 슬로우/배속을 *명시*한 경우만(슬로우=0.5, 빠르게=2), 아니면 1.\n"
        "- subject: 임보견 당겨찍기·얼굴 중심=foster, 전체=full.\n"
        "- caption: 지시가 이 문장을 '텍스트로/자막으로 띄워라' 하면 그 문구"
        "(대개 내레이션 문장 그대로), 아니면 빈 문자열.\n"
        "예1) 지시 '얼굴을 천천히 클로즈업으로 5초 보여주고' + 이 문장이 그 장면 → "
        "zoom=gradual, subject=foster, target_dur=5\n"
        "예2) 지시 '카페에서 노는 모습 10초' + 이 문장이 카페 장면 → "
        "select=dynamic, keywords=[\"카페\"], target_dur=10\n"
        f"내레이션 문장: {sentence}\n편집 지시(참고): {request}",
        _SCENE_SCHEMA, num_predict=256)
    b = EditBlock.from_dict(d)
    b.keywords = _clean_keywords(b.keywords or [])
    return b


def interpret_narration(request: str) -> EditPlan:
    """모드 A: 요청/대본 → 블록(장면 연출 + narration 구절) 시퀀스. 2단 분리:

    ①대본 추출 — 출력은 문장 리스트뿐(짧음). 고객 대본은 그대로, 없으면 Gemma 작성.
    ②장면 매칭 — 문장마다 select/keywords 등 *속성만*(enum 위주, 자유 텍스트 0).
    narration 텍스트는 파이썬이 ①의 문장을 블록에 꽂는다 — LLM 이 대본을 삼키거나
    중복 복창할 통로 자체를 없앤 구조("리터럴은 결정론" 원칙).
    내레이션이 타임라인 주인 — target_dur 는 사용자가 초를 명시한 경우만.
    """
    d = _chat_json(
        "강아지 입양영상 내레이션 요청이다. *목소리로 읽을 대본 문장*만 순서대로 뽑아라.\n"
        "- 고객이 대본을 줬으면 그 문장들을 그대로(새로 짓지도, 합치지도 마라).\n"
        "- 대본 없이 취지만 줬으면 짧고 따뜻한 문장 2~4개를 새로 써라.\n"
        "- 편집 지시('~장면으로 끝내줘', '~는 10초' 등)는 대본이 아니다 — 빼라.\n"
        "- 숫자·단위(3살, 7.2kg)는 바꾸지 말고 그대로.\n"
        "- title: 고객이 제목/전체 자막을 *명시적으로* 요청할 때만. '~영상 만들자'는 "
        "목적 문장이지 제목 요청이 아니다 — 그 경우 빈 문자열.\n요청: " + request,
        _SCRIPT_SCHEMA)
    sents = [s.strip() for s in d.get("sentences", []) if s and s.strip()]
    if not sents:
        raise RuntimeError("대본 문장 추출 실패(빈 리스트) — 요청을 확인하세요.")

    # 자막 결정론 폴백: 요청이 텍스트 표시를 언급하는데 Gemma 가 caption 을 놓치면
    # (실측: 전 블록 누락) 내레이션 문장을 그대로 자막으로 — "띄워주면서 읽어줘" 관례.
    wants_caption = ("텍스트" in request) or ("자막" in request)
    blocks = []
    for s in sents:                       # 문장당 1호출 — 문장이 곧 블록(내레이션이 주인)
        b = _match_scene(s, request)
        b.narration = s
        if wants_caption and not b.caption:
            b.caption = s
        blocks.append(b)
    attribute_directives(request, sents, blocks)
    return EditPlan(blocks=blocks, title=str(d.get("title", "") or ""))


_DUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*초")


def attribute_directives(request: str, sents: list, blocks: list) -> None:
    """리터럴 지시(N초·클로즈업)를 결정론으로 블록에 귀속 — 거리 기반 앵커.

    "카페 5초" 교훈: 명시 숫자는 LLM 에 맡기면 놓친다(실측: 예시를 줘도 전부 0).
    'X를 띄워주면서 읽어줘' 패턴상 대본 문장이 요청 원문 안에 그대로 있으므로, 문장
    위치를 앵커 삼아 각 지시("5초"·"클로즈업")를 가장 가까운 문장의 블록에 귀속한다.
    Gemma 가 이미 값을 채웠으면 존중(여긴 폴백). 문장이 원문에 없으면(대본을 Gemma 가
    작성한 경우) 귀속 불가 — 그대로 둔다.
    """
    sent_pos = [(request.find(s), i) for i, s in enumerate(sents) if request.find(s) >= 0]
    if not sent_pos:
        return

    def nearest(pos: int) -> int:
        return min(sent_pos, key=lambda t: abs(t[0] - pos))[1]

    for m in _DUR_RE.finditer(request):
        bi = nearest(m.start())
        if not blocks[bi].target_dur:
            blocks[bi].target_dur = float(m.group(1))
    for m in re.finditer(r"클로즈\s*업", request):
        bi = nearest(m.start())
        blocks[bi].zoom = "gradual"
        blocks[bi].subject = "foster"
