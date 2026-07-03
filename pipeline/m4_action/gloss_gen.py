"""라벨 gloss 일괄 생성(1회 도구) — 센서 출력 공간 전체를 한국어 관찰 표현으로.

배경(2026-07-03 사용자 지적): 손으로 고른 gloss 상수(~65개)는 라벨 공간(AudioSet
527 + Places365 365)의 일부만 덮는 고정 상수라, 표 밖 라벨은 영어 원문으로 남아
사실상 안 보이는 증거가 된다(실측: 영어 라벨은 한국어 텍스트 판정에서 증거로 안
읽힘 — 카페 0/2 → gloss 후 1/2). 유저가 무슨 footage 를 줄지 모르므로 라벨 공간
*전체*를 1회 번역해 데이터 자산(data/models/label_gloss_ko.json)으로 둔다.

gloss 는 순수 번역(함의·해석 아님). 생성은 Gemma 지만 결과는 정적 자산 — 검수
가능하고 런타임 비결정성이 없다. 이 자산이 gloss 의 **단일 출처**다(코드 상수
gloss 는 사용자 지시로 제거 — 이중 출처 금지). 재생성 시 골든 회귀 채점 필수.

실행: python -m pipeline.m4_action.gloss_gen
"""

from __future__ import annotations

import json

from .observe import AST_MODEL, MODELS_DIR

CHUNK = 20
OUT_PATH = MODELS_DIR / "label_gloss_ko.json"


def _audio_labels() -> list[str]:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(AST_MODEL)   # config 만 — 가중치 불필요
    return [cfg.id2label[i] for i in range(len(cfg.id2label))]


def _places_labels() -> list[str]:
    from .observe import _places_assets
    _places_assets()
    return [line.strip().split(" ")[0][3:]
            for line in (MODELS_DIR / "categories_places365.txt").read_text().splitlines()]


def _translate(labels: list[str], what: str) -> dict[str, str]:
    import ollama
    from .observe import MODEL
    out: dict[str, str] = {}
    for i in range(0, len(labels), CHUNK):
        chunk = labels[i:i + CHUNK]
        schema = {"type": "object",
                  "properties": {lb: {"type": "string"} for lb in chunk},
                  "required": chunk}
        # ⚠️ 기계적 '~소리' 접미 강제 금지 — 부자연 번역이 증거 강도를 바꾼 실측
        # (Music '음악 소리'→카페 도약 소실 / '음악'→회복, 2026-07-03 이분 탐색).
        # 자연스러운 한국어 우선, 재생성 후에는 골든 회귀 채점 필수.
        prompt = (f"다음은 {what} 분류 라벨(영어)이다. 각각을 짧고 *자연스러운* "
                  "한국어 관찰 표현으로 번역하라(2~6단어, 해석·추측 없이 라벨 뜻 "
                  "그대로. 어색한 접미어를 기계적으로 붙이지 마라).\n"
                  + "\n".join(f"- {lb}" for lb in chunk))
        r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0, "num_predict": 2048},
                        format=schema, think=False)
        try:
            got = json.loads(r.message.content)
        except json.JSONDecodeError:
            got = {}
        for lb in chunk:
            t = (got.get(lb) or "").strip()
            if t:
                out[lb] = t
        print(f"  {what}: {min(i + CHUNK, len(labels))}/{len(labels)}")
    return out


def main() -> int:
    # 기존 자산의 엔트리는 보존하고 빈 라벨만 채운다 — 자산에 가한 사람 검수
    # (계기 보정: 예. veterinarians_office 는 실측상 애견카페에도 0.9 로 발화하므로
    # '동물병원'이 아니라 '동물병원/동물시설')이 재생성으로 증발하면 안 된다.
    old = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else {}
    audio, places = dict(old.get("audio", {})), dict(old.get("places", {}))
    todo_a = [lb for lb in _audio_labels() if lb not in audio]
    todo_p = [lb for lb in _places_labels() if lb not in places]
    if todo_a:
        audio.update(_translate(todo_a, "오디오 이벤트(AudioSet)"))
    if todo_p:
        places.update(_translate(todo_p, "장소(Places365)"))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"audio": audio, "places": places},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {OUT_PATH}  (audio {len(audio)} / places {len(places)}, "
          f"신규 {len(todo_a) + len(todo_p)})")
    missing = [lb for lb in _audio_labels() if lb not in audio]
    missing += [lb for lb in _places_labels() if lb not in places]
    if missing:
        print(f"⚠️ 미번역 {len(missing)}개(원문 폴백): {missing[:10]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
