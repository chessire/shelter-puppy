"""M5 — TTS 내레이션 (Layer 2, 모드 A). 엔진: Qwen3-TTS 1.7B CustomVoice.

설계 재정의(2026-07-02 스파이크): "타임스탬프 주는 TTS 찾기"는 사라진 문제다.
대본은 Gemma 가 구절 단위로 분해해 주므로, **구절별 개별 합성 + 무음 결정론 조립**이면
구절 경계는 측정값이 아니라 조립 파라미터다(정확도 100%, forced alignment 불필요).
"LLM은 해석만, 실행은 결정론"의 오디오 버전 — TTS 는 파형 생성만, 타임라인은 조립기가.

엔진 선정(골든이어셋 2라운드, ~/tts-spike):
  - 1R 생텍스트에선 Qwen 이 CER 게이트 탈락(날짜 뭉갬)했으나, 숫자·날짜 축은
    정규화 전처리기가 제거하는 축 → 2R 정규화 입력에선 3엔진 전부 완벽 → 톤 청취로 Qwen 승.
  - 교훈: 게이트는 실전 조건(전처리 통과 후)으로 채점한다.
  - instruct 톤 제어("차분하고 따뜻한 목소리로")가 타 엔진에 없는 차별 능력.

보이스 정책: 잡(영상) 단위 통일 — 고객 선택, 기본 eric. 블록별 교체 금지(내레이터가
중간에 바뀌면 편집 사고처럼 들림). 블록별 분위기는 같은 보이스의 instruct 로.

격리: mlx-audio 는 프로젝트 venv 에 넣지 않고 전용 venv(venv-tts) + 서브프로세스 워커.
ollama 처럼 '외부 합성 프로세스' 패턴 — 서버 이전 시 워커만 transformers 백엔드로 교체.
"""

from __future__ import annotations

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
DEFAULT_VOICE = "eric"  # 2026-07-02 청취 선정. 후보: sohee/serena/vivian/ryan/aiden/eric/dylan
VOICES = ["sohee", "serena", "vivian", "ono_anna", "ryan", "aiden", "eric", "dylan", "uncle_fu"]

# ASR 왕복 게이트 — LLM-TTS 저빈도 오발음(1R "토리→프린") 방어. 표기 관행("킬로그램"↔"kg")
# 착시가 있어 원문/정규화문 중 min CER 로 비교하고, 임계는 '파국'만 잡게 느슨히.
ASR_CER_GATE = 0.20
DEFAULT_PAUSE = 0.35  # 구절 사이 기본 무음(초). 쉼표가 곧 편집점 — 컷은 무음 정중앙에.
