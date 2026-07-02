# jobs/ — 요청당 1개 잡(job) 런타임 (제품 경로)

개발 골든셋(`data/dev/`)과 분리된 런타임 작업공간. 각 잡은 자족적이며 GT 불필요(pred 경로).
저장 위치는 향후 `DATA_ROOT` 환경변수로 설정 가능(기본 ./jobs).

## 잡 디렉토리 레이아웃 (계획)
```
<job_id>/
  input/        # 검증 통과 업로드 (≤10개, ≤N MB)
  analysis/     # P0 정규화 mp4 + map.json
  preds/        # M1 검출, M4 동작태그
  cards/        # 임보견 후보 크롭(다견일 때 고객 제시)
  meta.json     # foster_track · scene_tags · 편집요청 · 검증결과 · state
  out/          # 렌더 결과
```

## 잡 라이프사이클 (Phase 2 구현됨 — `pipeline/job.py`)
1. `init_job(id, inputs)`: 디렉토리 생성 + input/ 저장 + meta 초기화
2. `prepare(ws)`: P0 정규화 + M1 검출 → 임보견 후보(단독견이면 자동확정)
3. [다견이면 고객이 트랙 선택 → `meta.foster_track` 저장]  ← Phase 3(카드)
4. `render(ws, request)`: M4 + M6 → out/

상태: uploaded → validated → prepared → (needs_foster_pick | auto) → rendering → done → cleaned

### CLI (잡 1개 끝까지)
```bash
# 단독견: init → prepare → render 한 번에
python -m pipeline.job test1 \
  --inputs data/dev/videos/IMG_0004.MOV \
  --request '차분하게 앉아있는 모습 8초로 담고 "우리 토리" 자막'

python -m pipeline.job test1 --inputs <영상> --prepare-only   # prepare 까지만
```
저장 루트는 `DATA_ROOT` 환경변수(기본 ./jobs) 또는 `--data-root` 로 변경.

단독견 자동확정: M1 track id 는 영상마다 독립이라 단독견은 `meta.foster_track=null`
로 두고 foster_boxes_pred 가 '최대 dog 박스'로 자족 처리. meta.foster_track 은
단일영상 다견(고객이 트랙 1개 선택)일 때만 채운다.

상세 전략: Notion "백엔드·폴더 구조 — 잡(job) 격리 전략"
