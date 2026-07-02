# M0 — 측정 하네스

파이프라인 전 단계(M1~M5)의 **정답(GT) vs 자동출력(pred)** 을 채점해 지표를 숫자로
내는 측정 인프라. 설계서 원칙: *"가정으로 확정 말고 숫자부터 측정."* 이게 없으면
M1~M5의 "테스트(임계 넘는지)"가 성립하지 않는다.

## 구성

```
pipeline/harness/
  schemas.py        # 데이터 계약 — 모든 모듈이 쓰고 모든 메트릭이 읽는 백본
  io.py             # JSONL(프레임열)/JSON(단건) 입출력
  metrics/
    common.py       # IoU·구간겹침·헝가리안 매칭
    m1_tracking.py  # CLEAR-MOT(MOTA·IDSW·miss·fragmentation) + IDF1
    m2_reid.py      # 군집 pair P/R(false merge) + 자동재연결률 + 개입수  ★게이트
    m3_motion.py    # 동/정 분리 정확도 + 손떨림 오검출율
    m4_action.py    # 군 정확도 + uncertain율 + 저모션 표집률            ★게이트(합침C)
    m5_match.py     # 구절↔클립 매칭 정확도 + 매칭 uncertain율
  thresholds.yaml   # 블록별 pass 임계값(현재 전부 null=미정)
  thresholds.py     # 임계값 로더 + PASS/FAIL 판정
  cli.py            # `eval --stage ...` 진입점
  convert/          # (Phase 2) CVAT export → JSONL 변환기 자리
```

## 데이터 계약 (단계별 GT/pred 포맷)

| 단계 | 포맷 | 파일 |
|---|---|---|
| M1 | `FrameDetections` JSONL (한 줄=한 프레임) | `*.jsonl` |
| M2 | `{"mapping": {track_id: global_id}}` (GT) / `ReIDResult`(pred) | `*.json` |
| M3 | `{"segments": [Segment(label=moving\|static)]}` | `*.json` |
| M4 | `{"segments": [ActionSegment(group=dynamic\|static, uncertain)]}` | `*.json` |
| M5 | `{"matches": [MatchEntry(phrase_id→source 구간)]}` | `*.json` |

GT와 pred는 **같은 포맷**을 공유한다 (같은 로더 → 같은 메트릭).

## 사용

```bash
# 단일 단계 채점
python -m pipeline.harness.cli eval --stage m1 \
    --gt data/gt/<video>_m1.jsonl --pred data/preds/<video>_m1.jsonl

# JSON 출력
python -m pipeline.harness.cli eval --stage m3 --gt GT --pred PRED --json
```

임계값(`thresholds.yaml`)이 설정돼 있으면 PASS/FAIL, 미설정(null)이면 숫자만 보고.

## 검증

```bash
python -m pytest pipeline/tests/ -q     # 메트릭 자체를 손계산 답으로 채점(25 cases)
```

## 다음 (Phase 2 — 영상 확보 시)

1. `convert/cvat_to_jsonl.py` — CVAT MOT export → M1 GT JSONL 변환기
2. 영상 3개(개+고양이 ×2, 개2마리 ×1) 스파스 키프레임 라벨링 → `data/gt/`
3. 첫 실측 → 분포 보고 `thresholds.yaml` 의 null 을 실제 숫자로 확정
