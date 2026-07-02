# M1 GT 라벨링 가이드 (CVAT)

분석 mp4(`data/analysis/*.mp4`)에 **임보견 박스 + 트랙ID**를 스파스 키프레임으로
라벨해 M1 정답(GT)을 만든다. 결과는 변환기로 JSONL 스키마로 떨군다.

## 원칙 (왜 이렇게 라벨하나)
- **분석 mp4에만 라벨한다** — 원본(.MOV)이 아니라 `data/analysis/<name>.mp4`. 이게
  모델이 추론할 좌표계(768 다운스케일·CFR)와 같아야 채점이 성립.
- **강아지만 라벨한다(label = `dog`)** — 고양이는 라벨하지 않는다. 그래야 모델이
  고양이를 개로 잡으면 **FP(타종 오검출)** 로 측정된다. (IMG_0004/0008 = 개+고양이)
- **개체별 트랙ID 분리** — 두 마리 영상(IMG_0069)은 강아지마다 **다른 track**으로.
  M1 은 둘 다 `dog`(track_id 다름), 누가 임보견인지는 M2(re-ID)가 정한다.
- **스파스 키프레임** — 전 프레임이 아니라 **약 0.5초 간격(=15프레임 @30fps)** 만
  키프레임으로 찍는다. 사이는 CVAT 보간에 맡기되, GT 는 키프레임만 쓴다(사람 판단).

## 1. CVAT 띄우기 (로컬 도커 권장 — 영상 비공개)
```bash
git clone https://github.com/cvat-ai/cvat && cd cvat
docker compose up -d
# 브라우저 http://localhost:8080  (최초 1회 superuser 생성:
#   docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser')
```
> 외부 유출이 괜찮으면 app.cvat.ai 무료 계정도 가능. 임보견 영상이면 로컬 권장.

## 2. 태스크 생성 (영상당 1개)
- **Tasks → +Create new task**
- Name: `IMG_0004` 등
- **Labels**: `dog` 하나 추가 (Rectangle). (원하면 `cat` 도 추가해 나중에 측정 가능)
- **Select files**: `data/analysis/IMG_0004.mp4` 업로드 → Submit

## 3. 라벨링 (Track 모드)
- 작업 열기 → 왼쪽 도구에서 **Track** 모드 + label `dog` 선택
- 강아지에 박스 드래그 → 한 마리 = 하나의 track (자동으로 track_id 부여)
- **0.5초(15프레임) 단위로** 이동하며 박스를 개 위치에 맞춰 조정(= 키프레임 생성)
- 개가 화면 밖으로 나가면 그 프레임에서 **Outside** 토글(빈 구간 처리)
- 두 마리면 각각 별도 track 으로 같은 방식 반복

## 4. Export
- **Menu → Export task dataset → Format: `CVAT for video 1.1`** → zip 다운로드
- zip 안의 `annotations.xml` 를 꺼낸다

## 5. JSONL 로 변환
```bash
python -m pipeline.harness.convert.cvat_to_jsonl \
    annotations.xml data/gt/IMG_0004_m1.jsonl --fps 30 --labels dog
```

## 6. 채점 (pred 가 준비되면)
```bash
python -m pipeline.harness.cli eval --stage m1 \
    --gt data/gt/IMG_0004_m1.jsonl --pred data/preds/IMG_0004_m1.jsonl
```
임계값(`thresholds.yaml`)이 아직 null 이라 숫자만 출력된다 → 3영상 분포를 보고
임계값을 확정한다(설계서가 지목한 "첫 할 일").

## M2~M5 라벨 (가벼움, 도구 불필요)
- **M2**: `data/gt/<name>_m2.json` = `{"mapping": {track_id: 임보견ID}}` 손작성
- **M3/M4**: `{"segments": [{start_t,end_t,label/group,...}]}`
- **M5**: `{"matches": [{phrase_id,source,start_t,end_t}]}`
