"""요청 주도 장면 추론 — 고정 어휘 없이, 유저 프롬프트의 키워드가 곧 보기.

설계 전환(2026-07-03 사용자 결정): 이전엔 고정 VOCAB enum 으로 영상을 선태깅했는데,
임보자가 무슨 영상을 줄지 알 수 없으므로 어떤 고정 어휘도 구조적 사각(카페·수영장…)을
남긴다. 대신 **요청에서 Gemma 가 뽑은 장면 키워드들을 그대로 vision 의 보기로** 쓴다 —
"이 장면에 카페/산책/밤 중 확실히 보이는 것은?" 어휘가 요청마다 동적 생성되므로 상수
결합이 없고, 요청 키워드와 태그가 같은 어휘 공간이라 매칭 어긋남(폴백)도 구조적으로
사라진다. 무에서의 추론이 아니라 프롬프트에서의 유추(사용자 표현 그대로).

유지되는 보수 게이트: 프레임 3장 다수결(2/3), "확실한 것만"(빈 배열 허용) — 틀린
태그(핀 오염) < 없는 태그(전체 폴백). 사람 태그(meta.scene_tags)는 항상 우선이며,
추론 결과는 meta.scene_tags_auto 에 키워드별로 누적 캐시(재렌더 시 새 키워드만 질문).
"""

from __future__ import annotations

import json

import cv2

from ..workspace import Workspace

MODEL = "gemma4:26b-a4b-it-q4_K_M"
_SAMPLE_FRACS = (0.2, 0.5, 0.8)
_MIN_AGREE = 2      # 다수결 하한 — 프레임 3장 중 2장


def _sample_jpegs(mp4: str) -> list[bytes]:
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for frac in _SAMPLE_FRACS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n * frac)))
        ok, frame = cap.read()
        if not ok:
            continue
        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            out.append(buf.tobytes())
    cap.release()
    return out


# 질문 방식 측정(2026-07-03, 골든 9영상 × 기대 6건):
#   다지선다("보이는 것 전부, 확실한 것만") = 고정밀·저재현 — 밤·하이파이브만 잡고
#     산책·비·카페는 빈 배열(빈 답이 최안전 답이 되는 어트랙터). 가짜 태그 0.
#   키워드당 예/아니오 = 재현 상승(산책·비 복구)하나 가짜 태그 유출(집 놀이→'카페',
#     밤 질주→'산책') + 하이파이브 소실. 가짜 태그는 핀 오염(가장 위험한 방향).
# → 정밀도 우선으로 다지선다 채택. 재현 부족은 매칭 의미론(키워드별 폴백 합집합,
#   m6_edit._scene_filter)이 흡수 — 못 잡은 키워드는 배제 없이 전체 폴백된다.
#   카페는 두 방식 모두 실패(스틸 신호 자체가 약함) — 사람 태그·재요청이 처방. [잠정]


def _frame_tags(jpeg: bytes, keywords: list[str]) -> list[str]:
    import ollama
    schema = {"type": "object",
              "properties": {"tags": {"type": "array", "items": {"enum": keywords}}},
              "required": ["tags"]}
    prompt = ("강아지 영상의 한 장면이다. 다음 보기 중 이 장면에 *확실히* 보이는 "
              "것만 골라라 — 애매하면 넣지 마라(빈 배열 가능).\n"
              "보기: " + " ".join(keywords))
    r = ollama.chat(model=MODEL,
                    messages=[{"role": "user", "content": prompt, "images": [jpeg]}],
                    options={"temperature": 0}, format=schema, think=False)
    try:
        tags = json.loads(r.message.content).get("tags", [])
    except json.JSONDecodeError:
        return []
    return [t for t in tags if t in keywords]


def consensus(per_frame: list[list[str]], order: list[str],
              min_agree: int = _MIN_AGREE) -> list[str]:
    """프레임별 태그 → 다수결 통과분만(order = 보기 순서 유지)."""
    from collections import Counter
    counts = Counter(t for tags in per_frame for t in set(tags))
    thr = min(min_agree, max(1, len(per_frame)))   # 샘플이 1~2장뿐이면 하한 완화
    return [t for t in order if counts.get(t, 0) >= thr]


def infer_scene_tags(ws: Workspace, names: list[str],
                     keywords: list[str]) -> dict[str, list[str]]:
    """요청 키워드를 보기로 영상별 장면 추론. 키워드가 없으면 물을 것도 없다."""
    kws = [k for k in dict.fromkeys(keywords) if k and k.strip()]
    if not kws:
        return {n: [] for n in names}
    result = {}
    for name in names:
        jpegs = _sample_jpegs(str(ws.analysis(name)))
        result[name] = consensus([_frame_tags(j, kws) for j in jpegs], kws) if jpegs else []
    return result


# ── 장면 문구 대조(2026-07-03) — 원자화된 키워드('카페')는 문맥이 잘려 스틸 판별이
# 약하다(실측: 애견카페 미인식). 요청의 장면 묘사 문구 통째("애견카페에서 뛰어 노는
# 모습")로 물으면 배경+행동 결합 신호를 쓸 수 있다. 예/아니오 초소형 + 다수결 유지.
_YESNO = {"type": "object", "properties": {"answer": {"enum": ["예", "아니오"]}},
          "required": ["answer"]}


def _frame_matches(jpeg: bytes, phrase: str) -> bool:
    import ollama
    prompt = (f"강아지 영상의 한 장면이다. 이 장면이 ‘{phrase}’의 한 장면이라고 "
              "볼 수 있나? 확실하면 예, 아니면 아니오.")
    r = ollama.chat(model=MODEL,
                    messages=[{"role": "user", "content": prompt, "images": [jpeg]}],
                    options={"temperature": 0}, format=_YESNO, think=False)
    try:
        return json.loads(r.message.content).get("answer") == "예"
    except json.JSONDecodeError:
        return False


def infer_scene_match(ws: Workspace, names: list[str],
                      phrases: list[str]) -> dict[str, list[str]]:
    """장면 문구별 적합 영상 목록 {문구: [영상들]}. 프레임 다수결(2/3)."""
    phs = [p for p in dict.fromkeys(phrases) if p and p.strip()]
    if not phs:
        return {}
    result: dict[str, list[str]] = {p: [] for p in phs}
    for name in names:
        jpegs = _sample_jpegs(str(ws.analysis(name)))
        if not jpegs:
            continue
        for p in phs:
            hits = sum(1 for j in jpegs if _frame_matches(j, p))
            if hits >= min(_MIN_AGREE, max(1, len(jpegs))):
                result[p].append(name)
    return result
