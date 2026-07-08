# shelter-puppy

강아지의 **폰 촬영 영상 + 자연어 요청 한 문단**을 받아, 자막·내레이션이 들어간
**세로 9:16 숏폼**을 자동으로 만들어내는 영상 파이프라인 (MVP).

```bash
python -m pipeline.job my_job \
  --inputs ~/videos/tori/ \
  --dog-photos ~/photos/tori/ \
  --request "우리 토리 소개 영상 만들어줘. 뛰어노는 모습 나오다가 산책하고, 하이파이브로 끝나게."
# → jobs/my_job/out/final.mp4
```

## 개발 과정
1. **[맥 로컬 LLM 구축](https://chessire.tistory.com/entry/local-ai-first-how)**, 무엇을 내 손 안에 두고, 무엇만 밖에 맡길 것인가. 노트북 한 대에 세운 환경.
2. **[로컬 LLM 아키텍처](https://chessire.tistory.com/entry/local-first-ai-why)**, 왜 로컬인지, 왜 이 구성인지. 선택의 이유.
3. **[YOLO11 해상도 함정](https://chessire.tistory.com/entry/measurement-first-ai-golden-set)**, 정답지부터 만들고 모델을 붙인다. 해상도를 올렸더니 오히려 나빠진 이야기.
4. **결정론적 AI 파이프라인 설계(발행예정)**, 같은 입력이면 같은 출력. 창작만 예외로 두되, 그 창작에도 가드를 건다.
5. **로컬 TTS로 타임라인 컨트롤(발행예정)**, 대본을 음성으로, 음성을 영상 타임라인에 맞추기.
6. **LLM은 지시만(발행예정)**, 프롬프트에 내용을 넣으면 출력으로 샌다. 같은 문제를 네 번 고치고 얻은 규칙.
7. **스레드보다 자원(발행예정)**, 8분을 4분으로. 가장 큰 배수는 멀티스레딩이 아니라, 놀고 있던 GPU를 깨운 한 줄에서 나왔습니다.
8. **전체 요약(발행예정)**

대표페이지 : [chessire.tistory.com](https://chessire.tistory.com)

## 핵심 설계 원칙

1. **LLM은 해석만, 실행은 결정론.** Gemma(로컬 멀티모달 LLM)는 의미 판단(동작 이름, 편집 요청
   분해, 장면 관찰)만 담당하고, 픽셀·프레임·타이밍은 전부 결정론 코드(Python/OpenCV/ffmpeg)가
   만진다. 같은 입력 = 같은 출력.
2. **측정-우선.** 가정으로 확정하지 않고 숫자부터 잰다. 손라벨 골든셋(GT) 5~9영상으로 모델·
   파라미터 변경을 매번 재채점 — M1 모델 체급, M2 임베딩 전략, TTS 엔진 선정까지 전부 골든
   채점으로 결정했다.
3. **처리 계층 분담.** 고전 CV(ByteTrack, 모션 보상)는 빠르고 일정하게, 전용 신경망(YOLO,
   DINOv2)은 검출·임베딩만, 생성형 LLM은 꼭 필요한 의미 해석 한 스텝에만. 특히
   **모션(동/정)은 M3 측정이 전담**하고 Gemma는 정지 크롭의 의미만 본다 — LLM은 구조적으로
   못 보는 축(단일 스틸의 모션)에서 자신 있게 틀리기 때문.
4. **사람 개입은 일관된 한 제스처.** 다견 중 임보견 지정, 애매한 매칭 확인 — 전부 "카드에서
   1탭"이라는 같은 결의 체크포인트로 수렴한다(상태머신이 해당 지점에서 멈춤).
5. **프롬프트에는 지시만, 내용은 금지.** 어휘·문구·프레이밍의 원천은 오직 유저 요청과 영상
   관찰이다. 코드/프롬프트에 박힌 상수 문자열은 출력으로 새는 사고를 반복해서 냈다.

## 파이프라인 구조

```
raw .MOV/.mp4 ─ P0 정규화 ─┐
                           │  Layer 1 — 영상 이해 (두 모드 공유)
                           ├─ M1 검출·추적 (YOLO11m + ByteTrack, MPS)
                           ├─ M2 re-ID    (DINOv2 임베딩 + 사진 앵커 → 임보견 확정)
                           ├─ M3 모션곡선  (카메라 보상 ROI 모션 → 동/정 구간)
                           └─ M4 동작판별  (Gemma logprob 군판정 + 관찰 프로필)
                                          = 동작 태그·관찰 자산 (재렌더 시 재사용)
자연어 요청 ─ 모드 판정 ─┬─ 모드 A: 내레이션 — M5 TTS (Qwen3-TTS, 구절=블록)
                        └─ 모드 B: 편집만 — 편집 인텐트 JSON
                                   └→ M6 편집 실행 (결정론 렌더: ffmpeg/OpenCV/PIL)
                                      → 세로 1080×1920 + AI 배지 + out/final.mp4
```

| 모듈 | 경로 | 역할 |
|---|---|---|
| M0 하네스 | `pipeline/harness/` | 데이터 계약(스키마)·메트릭·임계값·채점 CLI — 상세는 [harness/README.md](pipeline/harness/README.md) |
| P0 전처리 | `pipeline/preprocess/` | 회전 굽기(`-noautorotate`)·HDR→SDR 톤맵·VFR→CFR 30·768px 다운스케일 → 분석용 mp4 + 원본 좌표 매핑 |
| M1 검출·추적 | `pipeline/m1_track/` | 프레임별 강아지 박스 + 트랙 ID (MPS 가속, CPU 대비 3.5배) |
| M2 re-ID | `pipeline/m2_reid/` | 트랙→강아지 통합. 고객 사진 앵커(`photo_anchor.py`)로 다견 자동 확정, 애매하면 카드 |
| M3 모션 | `pipeline/m3_motion/` | 배경 특징점으로 카메라 모션 제거 → 임보견 ROI 잔차로 동/정 분리 |
| M4 동작·관찰 | `pipeline/m4_action/` | Gemma 5지선다 logprob 군판정 + 묘기 축, 관찰 프로필(`observe.py` — 캡션·오디오·장소·휘도·행동), 요청 주도 장면 태깅(`scene_auto.py`) |
| M5 TTS | `pipeline/m5_tts/` | 한국어 정규화 → 구절별 합성(전용 venv-tts 워커) → Whisper 왕복 CER 게이트 → 무음 결정론 조립 |
| M6 편집 | `pipeline/m6_edit/` | 요청→플랜 번역·저작(`author.py`), 텍스트 레벨 레이아웃(`layout.py`), 렌더(`run.py`), AI 배지(`badge.py`) |
| 잡 오케스트레이터 | `pipeline/job.py` | 상태머신 + 원샷 CLI |

## 저작 모드 — 요청 디테일의 그라디언트

요청이 얼마나 구조적이냐에 따라 세 경로로 갈린다 (`job._maybe_author`):

- **한 줄 목적** ("소개 영상 만들어줘") → **전체 저작**: Gemma가 영상별 관찰 프로필을 근거로
  구성(블록·자막·내레이션)을 창작. 파이프라인 유일의 의도된 비결정(온도 0.9) —
  `--rerender`가 "구성 복권".
- **스케치** ("뛰다가 산책하고 하이파이브로 끝") → **부분 저작**: 유저 뼈대는 그대로, 빈
  필드만 채움. 병합은 결정론(유저 명시 필드는 저작 출력에서 아예 안 읽음).
- **풀 스펙** → 저작 0호출, 번역만.

저작이 지어낸 자막은 **주장 분해 사실 검수**를 거친다: 자막에서 사실 주장을 추출해 소스별
관찰 기록과 logprob 대조, 근거 없으면 증거 제한 재작성. 유저가 직접 쓴 자막·대본은 검수하지
않는다(소유권).

화면 텍스트는 3레벨 계약: **L1 제목**(top, 오프닝 후 소멸) / **L2 설명 자막**(bottom, 블록과
함께, 1초당 6자 예산, TTS가 읽는 유일한 계층) / **L3 감탄**(피사체 옆 배치, 모션 피크에 스냅,
잠깐 떴다 사라짐). 위치·타이밍은 렌더러의 결정론 몫이고 저작은 "어느 계층에 무슨 말"만 정한다.

## 잡 시스템

잡은 fire-and-forget이 아니라 사람 개입 체크포인트가 있는 **2단 상태머신**:

```
uploaded → validated → prepared → (needs_foster_pick | needs_mode_pick | auto) → rendering → done
```

- `prepare`: P0+M1(병렬 오버랩) → 다견이면 사진 앵커로 자동 확정 시도, 애매하면 멈춤
- `render`: 모드 판정(키워드 핀 → Gemma logprob → uncertain) → M4 태깅(캐시 재사용) → 플랜 → 렌더
- 산출물: `out/final.mp4` + 플랜 사이드카 `out/final.plan.json`(사고 부검용) + AI 배지
  (편집만 = `AI 편집`, TTS 포함 = `AI 편집` + `내레이션: AI 음성`)

주요 플래그: `--rerender`(요청 재사용 재추첨) · `--prepare-only` · `--voice`(기본 eric) ·
`--size` · `--weights` · `--conf` · `--data-root`(기본 `./jobs`)

## 디렉토리

```
pipeline/    파이프라인 본체 (모듈 표 참고) + tests/
data/dev/    개발 골든셋 워크스페이스 — 영상·GT·pred·임베딩 (git 미포함)
data/models/ Places365 등 자동 다운로드 가중치
jobs/        잡 디렉토리 (잡마다 input/ refs/ analysis/ preds/ out/ meta.json 자족·격리)
demo_movies/ demo_pictures/  테스트 입력 소스
yolo11m.pt   YOLO 가중치 (측정으로 11n 대신 채택)
```

⚠️ 테스트·배터리는 사용자 잡 디렉토리에서 돌리지 않는다(덮어쓰기 사고 이력). 같은 잡 id
동시 실행 금지(meta 레이스) — 다른 잡 id끼리는 안전.

## 환경 셋업

Apple Silicon 전제(YOLO MPS · TTS MLX). 자세한 과정은 개발 기록의 [환경설정] 페이지 참고.

```bash
# 시스템 도구
brew install ffmpeg ollama
brew services start ollama
ollama pull gemma4:26b-a4b-it-q4_K_M    # 의미 해석 담당 (비전+logprob)

# 파이프라인 venv
python3 -m venv venv && source venv/bin/activate
pip install -r pipeline/requirements.txt
pip install ultralytics opencv-python transformers torchaudio  # M1·관찰 프로필

# TTS 전용 venv (mlx-audio 의존성 격리 — 상주 워커 서브프로세스로만 사용)
python3 -m venv venv-tts && venv-tts/bin/pip install mlx-audio soundfile
```

## 측정 하네스 · 테스트

```bash
# 골든셋 채점 (GT vs pred → 지표 + PASS/FAIL)
python -m pipeline.harness.cli eval --stage m1 --gt data/dev/gt/... --pred data/dev/preds/...
python -m pipeline.harness.cli report --stage m1

# 단위 테스트 (148개)
python -m pytest pipeline/tests
```

게이트 통과 이력: M2 re-ID 1.75탭/영상(목표 <2) · M4 군 정확도 1.00, uncertain 0.15 ·
e2e 2×2 매트릭스(자막×음성) 4/4 · 실사용 풀 라이프사이클 8분→4분(병렬화 1차).

## 개발 기록

의사결정 근거와 트러블슈팅 전체 기록은 Notion [MVP 개발 과정](https://app.notion.com/p/38e55c580506807493a1c66bb3d8869c):
환경설정 · 빌드 프로세스 · M0~M3 · M4/M6/통합 · M5 TTS/모드 라우팅 · 폴리싱(저작·텍스트 영역) ·
속도 최적화 · 폴리싱(텍스트 번인·TTS)
