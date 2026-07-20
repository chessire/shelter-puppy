# shelter-puppy

폰으로 찍은 강아지 영상과 자연어 요청 한 문단을 받아서, 자막과 내레이션이 들어간
세로 9:16 숏폼을 자동으로 만드는 영상 파이프라인 (MVP). 영상을 분석하며 얻은
관찰 기록과 동작 태그는 입양 응대봇의 근거 DB(`rag/`)로도 쓰인다.

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
4. **[결정론적 AI 파이프라인 설계](https://chessire.tistory.com/entry/deterministic-first-ai-interpretation)**, 같은 입력이면 같은 출력. 창작만 예외로 두되, 그 창작에도 가드를 건다.
5. **[로컬 TTS로 타임라인 컨트롤](https://chessire.tistory.com/entry/narration-first-ai)**, 대본을 음성으로, 음성을 영상 타임라인에 맞추기.
6. **LLM은 지시만(발행예정)**, 프롬프트에 내용을 넣으면 출력으로 샌다. 같은 문제를 네 번 고치고 얻은 규칙.
7. **스레드보다 자원(발행예정)**, 8분을 4분으로. 가장 큰 배수는 멀티스레딩이 아니라, 놀고 있던 GPU를 깨운 한 줄에서 나왔습니다.
8. **전체 요약(발행예정)**

대표페이지 : [chessire.tistory.com](https://chessire.tistory.com)

## 핵심 설계 원칙

1. **LLM은 해석만, 실행은 결정론.** 로컬 멀티모달 LLM인 Gemma는 동작 이름 짓기,
   편집 요청 분해, 장면 관찰 같은 의미 판단만 맡는다. 픽셀과 프레임, 타이밍은 전부
   Python, OpenCV, ffmpeg의 결정론 코드가 만진다. 같은 입력이면 같은 출력이 나온다.
2. **측정 우선.** 가정으로 확정하지 않고 숫자부터 잰다. 손으로 라벨한 골든셋
   5~9개 영상으로 모델과 파라미터 변경을 매번 다시 채점한다. M1 모델 체급도,
   M2 임베딩 전략도, TTS 엔진 선정도 전부 이 채점으로 결정했다.
3. **처리 계층 분담.** ByteTrack이나 모션 보상 같은 고전 CV는 빠르고 일정한 일을,
   YOLO와 DINOv2 같은 전용 신경망은 검출과 임베딩을, 생성형 LLM은 꼭 필요한
   의미 해석 한 스텝만 맡는다. 특히 동/정 판정은 M3 측정이 전담하고 Gemma는 정지
   크롭의 의미만 본다. 스틸 한 장에서 모션을 읽는 일은 LLM이 구조적으로 못 하는데,
   못 하면서 자신 있게 틀리기 때문이다.
4. **사람 개입은 한 가지 제스처로.** 여러 마리 중 주인공 강아지 지정도, 애매한
   매칭 확인도 전부 "카드에서 1탭"이라는 같은 모양의 체크포인트로 수렴한다.
   상태머신이 그 지점에서 멈추고 사람을 기다린다.
5. **프롬프트에는 지시만, 내용은 금지.** 어휘와 문구, 프레이밍의 원천은 유저 요청과
   영상 관찰뿐이다. 코드나 프롬프트에 박아둔 상수 문자열은 출력으로 새는 사고를
   반복해서 냈다.

## 파이프라인 구조

```
raw .MOV/.mp4 ─ P0 정규화 ─┐
                           │  Layer 1 — 영상 이해 (두 모드 공유)
                           ├─ M1 검출·추적 (YOLO11m + ByteTrack, MPS)
                           ├─ M2 re-ID    (DINOv2 임베딩 + 사진 앵커 → 주인공 강아지 확정)
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
| M3 모션 | `pipeline/m3_motion/` | 배경 특징점으로 카메라 모션 제거 → 주인공 강아지 ROI 잔차로 동/정 분리 |
| M4 동작·관찰 | `pipeline/m4_action/` | Gemma 5지선다 logprob 군판정 + 묘기 축, 관찰 프로필(`observe.py` — 캡션·오디오·장소·휘도·행동), 요청 주도 장면 태깅(`scene_auto.py`) |
| M5 TTS | `pipeline/m5_tts/` | 한국어 정규화 → 구절별 합성(전용 venv-tts 워커) → Whisper 왕복 CER 게이트 → 무음 결정론 조립 |
| M6 편집 | `pipeline/m6_edit/` | 요청→플랜 번역·저작(`author.py`), 텍스트 레벨 레이아웃(`layout.py`), 렌더(`run.py`), AI 배지(`badge.py`) |
| 잡 오케스트레이터 | `pipeline/job.py` | 상태머신 + 원샷 CLI |

## 저작 모드 — 요청 디테일의 그라디언트

요청이 얼마나 구조적이냐에 따라 세 경로로 갈린다 (`job._maybe_author`).

- **한 줄 목적.** "소개 영상 만들어줘"처럼 구조 정보가 없으면 전체 저작으로 간다.
  Gemma가 영상별 관찰 프로필을 근거로 블록 구성과 자막, 내레이션을 창작한다.
  파이프라인에서 유일하게 의도된 비결정 구간이라(온도 0.9), `--rerender`를 돌릴
  때마다 구성이 새로 나온다.
- **스케치.** "뛰다가 산책하고 하이파이브로 끝" 정도면 유저가 준 뼈대는 그대로
  두고 빈 필드만 채운다. 병합은 결정론이라, 유저가 명시한 필드는 저작 출력에서
  아예 읽지 않는다.
- **풀 스펙.** 저작을 부르지 않고 번역만 한다.

저작이 지어낸 자막은 주장 분해 사실 검수를 거친다. 자막에서 사실 주장을 추출해
소스별 관찰 기록과 logprob으로 대조하고, 근거가 없으면 증거 범위 안에서 다시 쓴다.
유저가 직접 쓴 자막과 대본은 검수하지 않는다. 자막의 소유권은 유저에게 있다.

화면에 얹는 텍스트는 세 레벨로 나뉜다. L1 제목은 상단에 떠 있다가 오프닝이 지나면
사라진다. L2 설명 자막은 하단에서 블록과 함께 가고, 1초당 6자 예산을 지키며, TTS가
읽는 유일한 계층이다. L3 감탄은 피사체 옆에 붙어 모션 피크에 맞춰 잠깐 떴다
사라진다. 위치와 타이밍은 렌더러의 결정론 몫이고, 저작은 어느 계층에 무슨 말을
얹을지만 정한다.

## 잡 시스템

잡은 던져두면 알아서 끝나는 구조가 아니라, 사람 개입 체크포인트가 있는 2단
상태머신이다.

```
uploaded → validated → prepared → (needs_foster_pick | needs_mode_pick | auto) → rendering → done
```

- `prepare`는 P0와 M1을 병렬로 겹쳐 돌린다. 강아지가 여러 마리 나오면 사진 앵커로
  주인공 자동 확정을 시도하고, 애매하면 여기서 멈춘다.
- `render`는 모드 판정을 거쳐 M4 태깅(캐시 재사용), 플랜, 렌더로 이어진다. 모드
  판정은 키워드 핀을 먼저 보고, 다음 Gemma logprob, 그래도 애매하면 uncertain으로
  사람에게 넘긴다.
- 산출물은 `out/final.mp4`와 사고 부검용 플랜 사이드카 `out/final.plan.json`, 그리고
  AI 배지다. 편집만 했으면 `AI 편집`, TTS가 들어가면 `내레이션: AI 음성`이 함께
  붙는다.

주요 플래그: `--rerender`(요청 재사용 재추첨) · `--prepare-only` · `--voice`(기본 eric) ·
`--size` · `--weights` · `--conf` · `--data-root`(기본 `./jobs`)

## RAG 저장·검색 — 입양 응대 근거

영상 분석 산출물을 입양 응대봇이 근거로 쓸 수 있게 DB로 옮긴다 (`rag/`). 스택은
Postgres에 pgvector를 얹은 단일 구성이고, 임베딩은 bge-m3(1024차원)를 쓰며, 모든
쿼리에 `dog_id` 필터가 붙는다. 여기의 L1/L2는 RAG 계층 이름으로, 화면 텍스트
3레벨과는 무관하다.

```
잡 산출물 ─ 인제스트 (foster 확정 잡만 — re-ID 게이트) ─┐
                 ├─ video_analysis  M4 세그먼트 + 관찰 프로필 원본 (JSONB)
                 ├─ L2 청크  영상당 관찰 서술 원문 + 인용 타임스탬프 (결정론 조립)
                 └─ L1 청크  trait 요약 (Gemma) — 주장 분해 사실 검수 통과분만
정형 카드 ─ 사람 입력, vis 게이팅(public/soft/staff) ─ 검색 없이 통째 주입
```

핵심 규칙은 저장 위치별로 텍스트 출처를 고정하는 것이다. 환각 위험은 검출
확신도가 아니라 그 문장을 누가 만들었느냐에 따라 달라지기 때문에, 청크마다 출처
컬럼을 두는 대신 테이블과 layer 단위로 출처를 통일했다. 정형 카드에는 사람이
입력한 문장만 들어간다. L2는 관찰 문장 원문만 담고 LLM이 손대지 못한다. L1은
LLM이 문장을 만드는 유일한 곳이라 사실 검수가 필수다. 영상에 들어간 저작 자막과
내레이션은 DB에 넣지 않는다. 같은 내용의 관찰 기록이 이미 있고, 자막 검수는 창작
문장을 사실로 보증해 주지 않기 때문이다.

```bash
python -m rag.cli init                            # 스키마 (Postgres + pgvector)
python -m rag.cli card tori --file card.json      # 정형 카드 (사람 입력)
python -m rag.cli ingest my_job --dog tori --dog-name 토리
python -m rag.cli search tori "다른 강아지랑 잘 지내나요?" --card
```

## 디렉토리

```
pipeline/    파이프라인 본체 (모듈 표 참고) + tests/
rag/         RAG 저장·검색 — 스키마·인제스트·검색 + tests/ (Postgres+pgvector)
data/dev/    개발 골든셋 워크스페이스 — 영상·GT·pred·임베딩 (git 미포함)
data/models/ Places365 등 자동 다운로드 가중치
jobs/        잡 디렉토리 (잡마다 input/ refs/ analysis/ preds/ out/ meta.json 자족·격리)
demo_movies/ demo_pictures/  테스트 입력 소스
yolo11m.pt   YOLO 가중치 (측정으로 11n 대신 채택)
```

⚠️ 테스트와 배터리는 사용자 잡 디렉토리에서 돌리지 않는다. 덮어쓰기 사고를 낸
이력이 있다. 같은 잡 id 동시 실행도 금지다(meta 레이스). 다른 잡 id끼리는 안전하다.

## 환경 셋업

Apple Silicon을 전제로 한다(YOLO는 MPS, TTS는 MLX). 자세한 내용은 개발 과정의
[맥 로컬 LLM 구축](https://chessire.tistory.com/entry/local-ai-first-how) 페이지 참고.

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

# RAG 저장소 (rag/ — 입양 응대 근거 DB)
brew install postgresql@18 pgvector
brew services start postgresql@18
createdb shelter_puppy
ollama pull bge-m3                      # 임베딩 (1024차원)
pip install psycopg2-binary
```

## 측정 하네스 · 테스트

```bash
# 골든셋 채점 (GT vs pred → 지표 + PASS/FAIL)
python -m pipeline.harness.cli eval --stage m1 --gt data/dev/gt/... --pred data/dev/preds/...
python -m pipeline.harness.cli report --stage m1

# 단위 테스트 (159개 — 파이프라인 + RAG)
python -m pytest pipeline/tests rag/tests
```
