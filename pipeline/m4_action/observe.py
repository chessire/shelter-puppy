"""상비 관찰(관찰 프로필) — 요청과 무관한 영상별 사실을 센서+캡션으로 1회 수집.

설계(2026-07-03, Notion '영상 구현 — 장면 추론'): "속성은 단어가 아니라 픽셀(과
오디오)에서". 죽은 건 고정 *어휘*(카페/산책…)지 고정 *센서*(닫힌 물리 축)가 아니다 —
센서는 상황이 아니라 축 전체를 커버하므로 미리 정해도 사각이 없다(M3 전례).

축별 스파이크 채점(2026-07-03, 골든 9영상):
  오디오(AST/AudioSet)  — 카페 특효: Speech 0.66~0.74 + Music 0.40~0.68 이 카페 2영상
                          에서만(비카페 Music ≤0.03). 스틸 4형식 전패 축이 오디오로
                          뚫림. 비도 통과(Rain·Ocean/Waves 물소리 0.6+, 타영상 ≤0.08).
                          실내/실외 클래스는 전부 <0.12 → 이 축은 못 가름(Places 몫).
                          지뢰: 강아지 발소리→Horse/Clip-clop 오분류 — 관찰은 그대로 적고
                          해석은 매칭 Gemma 에 맡긴다(관찰≠해석).
  Places365(ResNet18)   — 실내 확정만 신뢰(실내 6/6 avg 0.94+, 실외 3영상은 0.45~0.64
                          전부 애매 — 지면 위주 핸드헬드라 하늘 단서가 프레임에 없음).
                          한쪽 방향 센서: avg≥0.9 일 때만 '실내' 관찰, 실외 판정은
                          제공하지 않는다(캡션 몫).
  휘도                  — 밤 9/9 완벽 분리(밤 luma 41·암부 0.58 vs 나머지 126+·≤0.06).

수집은 결정론 파이썬이 무조건 실행(Gemma tool-calling 기각 — 재현성·초소형 출력·
비용), 프로필은 meta.scene_profile 에 영상별 캐시(M4 태그와 같은 요청 무관 재사용
자산). 관찰은 소프트 증거 — 하드 필터 승격은 축별 정밀도 채점 후. 센서 하나가
죽어도 prepare 가 죽으면 안 되므로 축별로 실패를 삼키고 빈 관찰로 둔다(안전한 실패).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..workspace import Workspace

MODEL = "gemma4:26b-a4b-it-q4_K_M"          # 캡션·매칭(해석) — M4 와 동일
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODELS_DIR = Path("data/models")             # Places365 가중치(재다운로드 가능 자산)

_SAMPLE_FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)   # 센서용 5장
_CAPTION_FRACS = (0.2, 0.5, 0.8)            # 캡션용 3장(scene_auto 와 동일)

# 오디오: 10초 윈도(5초 홉) 클래스별 max 집계, 임계 이상 top-N 만 관찰로 기록
_AUDIO_SR = 16000
_AUDIO_WIN = 10 * _AUDIO_SR
_AUDIO_HOP = 5 * _AUDIO_SR
AUDIO_MIN_PROB = 0.15
AUDIO_TOP_N = 8

# Places: 실내 확정 임계(골든 마진 0.94 vs 0.64) / 카테고리 보고 임계
PLACES_INDOOR_TAU = 0.9
PLACES_MIN_PROB = 0.2

# 휘도: 밤 판정(골든 마진 luma 41 vs 126+, 암부 0.58 vs ≤0.06)
LUMA_DARK = 60.0
DARK_FRAC = 0.5

# 라벨 한국어 gloss — *순수 번역*(함의·해석 아님). 영어 기계 라벨(Music 0.68,
# veterinarians_office …)은 한국어 텍스트 판정에서 증거로 안 읽힌다(실측: gloss 전
# 카페 0/2 → gloss 후 1/2). 출처는 data/models/label_gloss_ko.json **단일 자산**
# (gloss_gen.py 가 전체 라벨 공간 AudioSet 527 + Places365 365 를 1회 번역) —
# 코드 상수 gloss 는 2026-07-03 사용자 지시("상수 문자열 빼기")로 제거했다. 부분
# 손검수 표는 표 밖 라벨을 안 보이는 증거로 만드는 고정 상수 함정이었고, 이중
# 출처(코드+자산) 유지비만 남긴다. 자산이 없으면 라벨 원문 폴백(안전한 저하).


# --------------------------------------------------------------------------- #
# 프레임 샘플링 (공용)
# --------------------------------------------------------------------------- #
def _sample_frames(mp4: str, fracs: tuple) -> list[np.ndarray]:
    cap = cv2.VideoCapture(mp4)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for frac in fracs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n * frac)))
        ok, frame = cap.read()
        if ok:
            out.append(frame)
    cap.release()
    return out


# --------------------------------------------------------------------------- #
# 센서 ① 오디오 — AST(AudioSet 527) 이벤트 태깅. 원본에서 추출(분석 mp4 는 -an).
# --------------------------------------------------------------------------- #
_ast_cache: list = []
# 센서 팬아웃(ensure_profiles) 시 지연 초기화가 스레드에서 경합하지 않게 — 이중 로드
# (모델 2회 다운로드·메모리 2배) 방지. 추론 자체는 eval 모듈이라 동시 호출 안전.
_init_lock = threading.Lock()


def _ast():
    if not _ast_cache:
        with _init_lock:
            if not _ast_cache:
                from transformers import ASTFeatureExtractor, ASTForAudioClassification
                fe = ASTFeatureExtractor.from_pretrained(AST_MODEL)
                model = ASTForAudioClassification.from_pretrained(AST_MODEL)
                model.eval()
                _ast_cache.append((fe, model))
    return _ast_cache[0]


def _audio_probs(src_video: str | Path):
    """원본 영상 오디오 → AudioSet 527 확률 벡터(윈도별 max 집계). 실패 시 None."""
    import soundfile as sf
    import torch
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src_video),
             "-vn", "-ac", "1", "-ar", str(_AUDIO_SR), tmp.name],
            capture_output=True, text=True)
        if r.returncode != 0:
            return None                      # 오디오 트랙 없음 등 — 관찰 없음
        wav, sr = sf.read(tmp.name, dtype="float32")
    if len(wav) < _AUDIO_SR // 2:            # 0.5초 미만이면 신호가 못 됨
        return None
    fe, model = _ast()
    starts = [0] if len(wav) <= _AUDIO_WIN else \
        list(range(0, len(wav) - _AUDIO_WIN + 1, _AUDIO_HOP)) + \
        ([len(wav) - _AUDIO_WIN] if (len(wav) - _AUDIO_WIN) % _AUDIO_HOP else [])
    probs = None
    for s in starts:
        inputs = fe(wav[s:s + _AUDIO_WIN], sampling_rate=_AUDIO_SR, return_tensors="pt")
        with torch.no_grad():
            p = torch.sigmoid(model(**inputs).logits[0]).numpy()
        probs = p if probs is None else np.maximum(probs, p)
    return probs


def _audio_top(probs) -> list[list]:
    """확률 벡터 → [[클래스, prob], ...] (임계 이상 top-N, 프로필 표시용)."""
    if probs is None:
        return []
    _, model = _ast()
    id2label = model.config.id2label
    top = np.argsort(-probs)[:AUDIO_TOP_N]
    return [[id2label[int(i)], round(float(probs[i]), 3)]
            for i in top if probs[i] >= AUDIO_MIN_PROB]


def audio_tags(src_video: str | Path) -> list[list]:
    """원본 영상 오디오 → [[클래스, prob], ...] (max 집계, 임계 이상 top-N)."""
    return _audio_top(_audio_probs(src_video))


# --------------------------------------------------------------------------- #
# 센서 ② Places365 — 실내 확정(한쪽 방향) + 확신 카테고리
# --------------------------------------------------------------------------- #
_places_cache: list = []


_PLACES_URLS = {
    "resnet18_places365.pth.tar":
        "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
    "categories_places365.txt":
        "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt",
    "IO_places365.txt":
        "https://raw.githubusercontent.com/CSAILVision/places365/master/IO_places365.txt",
}


def _places_assets() -> None:
    """가중치·메타 없으면 1회 다운로드(재다운로드 가능 자산이라 data/는 gitignore)."""
    import urllib.request
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in _PLACES_URLS.items():
        p = MODELS_DIR / fname
        if not p.exists():
            print(f"     [관찰] Places365 자산 다운로드: {fname}…")
            urllib.request.urlretrieve(url, p)


def _places():
    if not _places_cache:
        with _init_lock:
            if not _places_cache:
                import torch
                import torchvision.models as tvm
                _places_assets()
                ckpt_path = MODELS_DIR / "resnet18_places365.pth.tar"
                model = tvm.resnet18(num_classes=365)
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                model.load_state_dict({k.replace("module.", ""): v
                                       for k, v in ckpt["state_dict"].items()})
                model.eval()
                cats = [line.strip().split(" ")[0][3:]
                        for line in (MODELS_DIR / "categories_places365.txt").read_text().splitlines()]
                io_map = {}
                for line in (MODELS_DIR / "IO_places365.txt").read_text().splitlines():
                    name, flag = line.strip().rsplit(" ", 1)
                    io_map[name[3:]] = int(flag)     # 1=indoor, 2=outdoor
                _places_cache.append((model, cats, io_map))
    return _places_cache[0]


def places_scene(analysis_mp4: str | Path) -> dict:
    """{indoor: bool|None, categories: [[이름, prob], ...]} — indoor 는 확정일 때만 True."""
    import torch
    import torchvision.transforms as T
    model, cats, io_map = _places()
    tf = T.Compose([T.ToPILImage(), T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    probs = []
    for fr in _sample_frames(str(analysis_mp4), _SAMPLE_FRACS):
        with torch.no_grad():
            logits = model(tf(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).unsqueeze(0))
        probs.append(torch.softmax(logits, dim=1)[0].numpy())
    if not probs:
        return {"indoor": None, "categories": []}
    p = np.mean(probs, axis=0)
    indoor_mass = float(sum(p[i] for i in range(365) if io_map[cats[i]] == 1))
    top = np.argsort(-p)[:5]
    return {"indoor": True if indoor_mass >= PLACES_INDOOR_TAU else None,
            "categories": [[cats[i], round(float(p[i]), 3)]
                           for i in top if p[i] >= PLACES_MIN_PROB]}


# --------------------------------------------------------------------------- #
# 센서 ③ 휘도 — 밤/어두움
# --------------------------------------------------------------------------- #
def luma_stats(analysis_mp4: str | Path) -> dict:
    lumas, darks = [], []
    for fr in _sample_frames(str(analysis_mp4), _SAMPLE_FRACS):
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        lumas.append(float(gray.mean()))
        darks.append(float((gray < 40).mean()))
    if not lumas:
        return {"luma": None, "dark_frac": None, "dark": False}
    luma, dark = float(np.mean(lumas)), float(np.mean(darks))
    return {"luma": round(luma, 1), "dark_frac": round(dark, 2),
            "dark": is_dark(luma, dark)}


def is_dark(luma: float, dark_frac: float) -> bool:
    return luma < LUMA_DARK or dark_frac > DARK_FRAC


# --------------------------------------------------------------------------- #
# 배경 벡터 — 피사체(검출 박스) 마스킹한 배경 부자 프레임의 DINOv2 임베딩.
# 장면 전파의 시각 채널(2026-07-03 스파이크): 강아지를 안 지우면 "나무 바닥 위 흰 강아지"
# 라는 피사체 유사도가 배경 유사도로 위장한다(다른 실내 0.78 → 마스킹 후 0.52,
# 같은 카페는 0.71 유지 = 격차 0.19). 상단 밴드만 보기는 목표까지 망가져 기각.
# --------------------------------------------------------------------------- #
_dino_cache: list = []


def _dino():
    if not _dino_cache:
        with _init_lock:
            if not _dino_cache:
                from ..m2_reid.embed import DinoEmbedder
                _dino_cache.append(DinoEmbedder())
    return _dino_cache[0]


def _bg_frames_masked(ws: Workspace, name: str, k: int = 5) -> list[np.ndarray]:
    """시간 구간(bin)별 검출 커버리지 최소 프레임 → 검출 박스를 평균색으로 마스킹."""
    from ..harness import io as hio
    p = ws.preds_m1(name)
    if not p.exists():
        return []
    frames = hio.read_frames(str(p))
    by_idx = {f.frame_idx: f for f in frames}
    cap = cv2.VideoCapture(str(ws.analysis(name)))
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    area = max(W * H, 1.0)
    cov = {f.frame_idx: min(sum(d.bbox.w * d.bbox.h for d in f.detections) / area, 1.0)
           for f in frames}
    out = []
    idxs = sorted(cov)
    if idxs:
        for b in np.array_split(idxs, k):
            if not len(b):
                continue
            i = min(b, key=lambda j: cov[j])
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                continue
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            mean = fr.reshape(-1, 3).mean(axis=0)
            h, w = fr.shape[:2]
            for d in by_idx[i].detections:      # 강아지·고양이 등 검출 전부(피사체 제거)
                pad_w, pad_h = d.bbox.w * 0.15, d.bbox.h * 0.15
                x0 = max(0, int(d.bbox.x - pad_w)); y0 = max(0, int(d.bbox.y - pad_h))
                x1 = min(w, int(d.bbox.x + d.bbox.w + pad_w))
                y1 = min(h, int(d.bbox.y + d.bbox.h + pad_h))
                fr[y0:y1, x0:x1] = mean
            out.append(fr)
    cap.release()
    return out


def scene_vec(ws: Workspace, name: str) -> list[float]:
    """마스킹 배경 프레임들의 DINOv2 평균 단위벡터(384). 계산 불가면 []."""
    frs = _bg_frames_masked(ws, name)
    if not frs:
        return []
    e = _dino().embed(frs)
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    v = e.mean(axis=0)
    v = v / np.linalg.norm(v)
    return [round(float(x), 4) for x in v]


# --------------------------------------------------------------------------- #
# 캡션 — Gemma vision 자유 묘사(열린 어휘). 프레임 3장 한 호출.
# --------------------------------------------------------------------------- #
_CAPTION_SCHEMA = {"type": "object", "properties": {"caption": {"type": "string"}},
                   "required": ["caption"]}


def caption(analysis_mp4: str | Path) -> str:
    import ollama
    jpegs = []
    for fr in _sample_frames(str(analysis_mp4), _CAPTION_FRACS):
        ok, buf = cv2.imencode(".jpg", fr)
        if ok:
            jpegs.append(buf.tobytes())
    if not jpegs:
        return ""
    prompt = ("한 강아지 영상에서 시간순으로 뽑은 프레임들이다. 영상의 장소·환경·"
              "상황을 한국어 1~2문장으로 객관적으로 묘사하라. 보이는 것만 적고 "
              "추측(장소 이름 단정, 강아지의 감정)은 하지 마라.")
    r = ollama.chat(model=MODEL,
                    messages=[{"role": "user", "content": prompt, "images": jpegs}],
                    options={"temperature": 0, "num_predict": 200,
                             "repeat_penalty": 1.3},
                    format=_CAPTION_SCHEMA, think=False)
    try:
        return json.loads(r.message.content).get("caption", "").strip()
    except json.JSONDecodeError:
        return ""


# 행동 관찰은 캡션(장소·환경)보다 시간 문맥이 필요해 프레임을 더 촘촘히 뽑는다.
_BEHAVIOR_FRACS = (0.1, 0.3, 0.5, 0.7, 0.9)
_BEHAVIOR_SCHEMA = {"type": "object",
                    "properties": {"behavior": {"type": "string"}},
                    "required": ["behavior"]}


def behavior(analysis_mp4: str | Path) -> str:
    """대상의 행동·상호작용 관찰(사용자 제안 2026-07-07) — 자막 환각의 근치.

    캡션은 장소·환경 축이라 '식기 주변에서 움직임'까지만 말한다 → 저작·검수가
    '밥 먹기'를 중재해야 했다(simple-demo3 3연속 사고). 행동 축이 '뒤엉켜 장난치며
    놀고 있음'이라고 말해주면 저작은 진짜 소재를 얻고 검수는 반증을 얻는다
    (스파이크 4/4: 놀이·앞발 내밀기·고양이 앞 멈춤·목줄 질주 정확, 지어냄 0).
    ⚠️ profile_text(장면 매칭 입력)에는 넣지 않는다 — 골든 5/6 회귀 없이 저작·검수
    기록(author._record_lines)에만 합류. 매칭 합류는 골든 재채점과 함께 별도 실험.
    """
    import ollama
    jpegs = []
    for fr in _sample_frames(str(analysis_mp4), _BEHAVIOR_FRACS):
        ok, buf = cv2.imencode(".jpg", fr)
        if ok:
            jpegs.append(buf.tobytes())
    if not jpegs:
        return ""
    prompt = ("한 강아지 영상에서 시간순으로 뽑은 프레임들이다. 강아지(들)의 "
              "행동과 상호작용(사람·다른 동물·물건과 무엇을 하는지)을 보이는 "
              "것만 한국어 1~2문장으로 서술하라. 흔히 연상되는 행동이라도 "
              "프레임에 보이지 않으면 적지 마라.")
    r = ollama.chat(model=MODEL,
                    messages=[{"role": "user", "content": prompt, "images": jpegs}],
                    options={"temperature": 0, "num_predict": 200,
                             "repeat_penalty": 1.3},
                    format=_BEHAVIOR_SCHEMA, think=False)
    try:
        return json.loads(r.message.content).get("behavior", "").strip()
    except json.JSONDecodeError:
        return ""


# --------------------------------------------------------------------------- #
# 프로필 빌드 + meta 캐시
# --------------------------------------------------------------------------- #
def _sensor_profile(ws: Workspace, name: str) -> dict:
    """센서 4축(오디오·배경벡터·장면분류·휘도) — 캡션 제외한 CPU 축.

    축별 실패는 삼키고 빈 관찰(안전한 실패). ensure_profiles 가 영상 간 스레드로
    팬아웃하는 단위라 meta 를 만지지 않는다(같은 잡 동시 쓰기 = 레이스 수칙).
    """
    prof: dict = {}
    src = ws.source_video(name)
    try:
        probs = _audio_probs(src) if src else None
        prof["audio"] = _audio_top(probs)
        # 전파용 전체 벡터(527) — 상위 태그만 저장하면 코사인이 측정과 어긋난다
        prof["audio_vec"] = [round(float(p), 3) for p in probs] if probs is not None else []
    except Exception as e:                    # noqa: BLE001 — 센서가 prepare 를 못 죽인다
        print(f"     [관찰] {name} 오디오 실패: {e}")
        prof["audio"], prof["audio_vec"] = [], []
    try:
        prof["scene_vec"] = scene_vec(ws, name)
    except Exception as e:                    # noqa: BLE001
        print(f"     [관찰] {name} 배경벡터 실패: {e}")
        prof["scene_vec"] = []
    try:
        prof["places"] = places_scene(ws.analysis(name))
    except Exception as e:                    # noqa: BLE001
        print(f"     [관찰] {name} 장면분류 실패: {e}")
        prof["places"] = {"indoor": None, "categories": []}
    try:
        prof["luma"] = luma_stats(ws.analysis(name))
    except Exception as e:                    # noqa: BLE001
        print(f"     [관찰] {name} 휘도 실패: {e}")
        prof["luma"] = {"luma": None, "dark_frac": None, "dark": False}
    return prof


def build_profile(ws: Workspace, name: str) -> dict:
    """영상 1개의 관찰 프로필(센서 4축 + 캡션 + 행동). 축별 실패는 삼키고 빈 관찰."""
    prof = _sensor_profile(ws, name)
    try:
        prof["caption"] = caption(ws.analysis(name))
    except Exception as e:                    # noqa: BLE001
        print(f"     [관찰] {name} 캡션 실패: {e}")
        prof["caption"] = ""
    try:
        prof["behavior"] = behavior(ws.analysis(name))
    except Exception as e:                    # noqa: BLE001
        print(f"     [관찰] {name} 행동 실패: {e}")
        prof["behavior"] = ""
    return prof


def ensure_profiles(ws: Workspace, names: list[str]) -> dict[str, dict]:
    """meta.scene_profile 캐시 — 없는 영상만 빌드(요청 무관 재사용 자산).

    구버전 캐시(전파 벡터 없는 프로필)는 빠진 벡터만 채워 넣는다(마이그레이션).
    속도: 센서 4축(CPU)은 영상 간 팬아웃, 캡션(Gemma)은 직렬 — ollama 는 서버가
    요청을 직렬 처리하는 공유 자원이라 스레드를 늘려도 줄만 선다. 총시간 ≈
    max(캡션 직렬 합, 센서 병렬 합). meta 쓰기는 루프 밖 1회.
    """
    meta = ws.read_meta()
    profiles = dict(meta.get("scene_profile") or {})
    todo = [n for n in names if n not in profiles]
    dirty = bool(todo)
    if todo:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=min(4, len(todo))) as pool:
            futs = {n: pool.submit(_sensor_profile, ws, n) for n in todo}
            for name in todo:
                print(f"[관찰] {name} 프로필(오디오·장면·휘도·캡션·행동)…")
                try:
                    cap_txt = caption(ws.analysis(name))
                except Exception as e:        # noqa: BLE001
                    print(f"     [관찰] {name} 캡션 실패: {e}")
                    cap_txt = ""
                try:
                    beh_txt = behavior(ws.analysis(name))
                except Exception as e:        # noqa: BLE001
                    print(f"     [관찰] {name} 행동 실패: {e}")
                    beh_txt = ""
                prof = futs[name].result()
                prof["caption"] = cap_txt
                prof["behavior"] = beh_txt
                profiles[name] = prof
                print(f"     {profile_text(prof)}")
        print(f"[시간] 관찰 프로필 {len(todo)}영상 {time.time() - t0:.1f}s")
    for name in names:
        prof = profiles.get(name)
        if prof is None or name in todo:
            continue
        if "audio_vec" not in prof:
            src = ws.source_video(name)
            probs = _audio_probs(src) if src else None
            prof["audio_vec"] = [round(float(p), 3) for p in probs] if probs is not None else []
            dirty = True
        if "scene_vec" not in prof:
            prof["scene_vec"] = scene_vec(ws, name)
            dirty = True
        if "behavior" not in prof:            # 구캐시 마이그레이션(잡당 1회, ~3s/영상)
            try:
                prof["behavior"] = behavior(ws.analysis(name))
            except Exception as e:            # noqa: BLE001
                print(f"     [관찰] {name} 행동 실패: {e}")
                prof["behavior"] = ""
            dirty = True
    if dirty:
        ws.update_meta(scene_profile=profiles)
    return {n: profiles[n] for n in names if n in profiles}


_gloss_tables: dict = {}


def _gloss(kind: str) -> dict[str, str]:
    """'audio'|'places' → 전체 gloss 표(생성 자산 단일 출처, 없으면 빈 표=원문 폴백)."""
    if not _gloss_tables:
        p = MODELS_DIR / "label_gloss_ko.json"
        gen = {}
        if p.exists():
            try:
                gen = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                gen = {}
        _gloss_tables["audio"] = gen.get("audio", {})
        _gloss_tables["places"] = gen.get("places", {})
    return _gloss_tables[kind]


def profile_text(prof: dict, extra: str | None = None) -> str:
    """프로필 → 매칭 프롬프트용 한 줄 관찰 기록. 라벨은 한국어 gloss(순수 번역)."""
    audio_ko, places_ko = _gloss("audio"), _gloss("places")
    parts = []
    audio = prof.get("audio") or []
    if audio:
        parts.append("소리: " + ", ".join(f"{audio_ko.get(c, c)}({p})" for c, p in audio))
    else:
        parts.append("소리: (신호 없음)")
    places = prof.get("places") or {}
    cats = places.get("categories") or []
    seg = []
    if places.get("indoor"):
        seg.append("실내 확실")
    if cats:
        seg.append(", ".join(f"{places_ko.get(c, c)}({p})" for c, p in cats))
    if seg:
        parts.append("장면분류: " + " · ".join(seg))
    luma = prof.get("luma") or {}
    if luma.get("dark"):
        parts.append("밝기: 어두움(밤/저조도)")
    if extra:
        parts.append(extra)
    cap = (prof.get("caption") or "").strip()
    if cap:
        parts.append("묘사: " + cap)
    # ⚠️ 행동 축(behavior)은 여기 넣지 않는다 — 매칭 합류 골든 재채점(2026-07-07):
    # 행동 없이 6/7 안정 → 행동 포함 5/7(카페-0069 소실 3/3, 증거 빈약 축을 행동
    # 텍스트가 흔듦) = 기각. 행동은 저작·검수 기록(author._record_lines) 전용,
    # 거기선 검수 배터리 21/21. 재도전 시 골든 재채점 필수.
    return " | ".join(parts)


def motion_summary(ws: Workspace, name: str) -> str | None:
    """M4 태그 → 정성 모션 한 줄(결정론 요약). 골든 실측: '움직임 0초' 같은 숫자만으론
    텍스트 판정이 모순을 못 쓰고(놀이 가짜), '거의 정지' 정성 표현이 있어야 쓴다.
    동작 *라벨*은 넣지 않는다 — 라벨이 거칠어(놀이가 '걷기') 산책류 오배정을 유발.
    """
    p = ws.preds_m4(name)
    if not p.exists():
        return None
    from ..harness import io as hio
    segs = hio.read_action_segments(str(p))
    if not segs:
        return None
    dyn = sum(s.end_t - s.start_t for s in segs if s.group == "dynamic")
    sta = sum(s.end_t - s.start_t for s in segs if s.group == "static")
    q = ("활발히 움직임" if dyn > sta * 2
         else "거의 정지" if sta > dyn * 2 else "움직임 보통")
    return f"동작측정: {q}(움직임 {dyn:.0f}초/정지 {sta:.0f}초)"


# --------------------------------------------------------------------------- #
# 매칭 — 키워드 ↔ 프로필, Gemma *텍스트* 판정 (vision 불필요, 키워드당 1호출)
# --------------------------------------------------------------------------- #
def match_keywords(profiles: dict[str, dict], keywords: list[str],
                   extras: dict[str, str] | None = None) -> dict[str, list[str]]:
    """{키워드: [부합 영상들]} — Gemma 텍스트 판정, 키워드당 1호출.

    레시피(골든 9영상 × 3회 안정 채점으로 확정, 2026-07-03): **완화 매칭 → 프루닝**.
      1) 완화 매칭 '뒷받침하면 포함, 모순·무관이면 제외' — 재현 확보.
         strict('확실한 것만')는 빈 배열 어트랙터로 카페·놀이·하이파이브 전멸(2/6).
      2) 프루닝 패스(검증 비대칭): 후보 중 '명백히 반대'인 것만 골라 *빼라*.
         출력⊆입력이라 가짜 추가가 구조적으로 불가능하고, 빈 배열 어트랙터가
         안전 방향(안 뺌)으로 작동. 완화 단독의 놀이-9980 가짜(정지 영상이 놀이로,
         핀 오염 방향)를 정확히 제거하며 재현 무손실 = 안정 4/6.
         프루닝 지시의 요체: "빈약·애매 ≠ 모순"(v1은 발소리 없다고 산책 영상을,
         동물병원 라벨 보고 카페 후보를 과잉 제거 — v2 문구가 해소).
      기각 계보: 1단에 모순 규칙 통합(카페·하이파이브 재현 소실 + 2단 vision 이
      하이파이브를 못 받음 실측), think=True(추론 배회로 가짜·비결정), 예/아니오
      분할(가짜 유출 — 융합 vision 포함 재확인). extras = 영상별 추가 관찰(모션 등).
      잔여 한계: 카페-0066(증거 빈약 → 부분 매칭, 안전 방향), 산책-0199(밤 질주도
      목줄 산책이라 라벨 판단 경계). 처방은 사람 태그.
    초소형 출력: 응답은 영상 이름 enum 배열뿐. 결과는 결정론 소독(enum 재검증).
    """
    import ollama
    kws = [k for k in dict.fromkeys(keywords) if k and k.strip()]
    names = sorted(profiles)
    if not (kws and names):
        return {k: [] for k in kws}
    extras = extras or {}
    records = "\n".join(f"[{n}] {profile_text(profiles[n], extras.get(n))}"
                        for n in names)
    schema = {"type": "object",
              "properties": {"videos": {"type": "array", "items": {"enum": names}}},
              "required": ["videos"]}
    rec = {n: profile_text(profiles[n], extras.get(n)) for n in names}
    out: dict[str, list[str]] = {}
    for kw in kws:
        prompt = ("강아지 영상들의 관찰 기록이다(소리·장면분류·밝기·동작은 기계 "
                  "측정이라 오류 가능, 종합해서 판단).\n" + records +
                  f"\n\n질문: ‘{kw}’ 장면/상황에 부합하는 영상을 골라라. 관찰 기록이 "
                  "그 장면을 뒷받침하면 포함하고, 모순되거나 무관하면 빼라.")
        r = ollama.chat(model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0}, format=schema, think=False)
        try:
            picked = json.loads(r.message.content).get("videos", [])
        except json.JSONDecodeError:
            picked = []
        cand = [n for n in dict.fromkeys(picked) if n in profiles]
        out[kw] = _prune(kw, cand, rec)
    return out


def _prune(kw: str, cand: list[str], rec: dict[str, str]) -> list[str]:
    """프루닝 패스 — 후보 중 관찰과 '명백히 반대'인 것만 제거(출력⊆입력).

    가짜 추가가 구조적으로 불가능한 검증 비대칭. '빈약·애매 ≠ 모순' 문구가 핵심
    (없으면 과잉 프루닝 — 산책·카페 후보 소실 실측).
    """
    import ollama
    if not cand:
        return cand
    schema = {"type": "object",
              "properties": {"remove": {"type": "array", "items": {"enum": cand}}},
              "required": ["remove"]}
    recs = "\n".join(f"[{n}] {rec[n]}" for n in cand)
    prompt = (f"다음 영상들은 ‘{kw}’ 장면 후보로 뽑혔다. 관찰 기록이다.\n" + recs +
              f"\n\n이 중 관찰이 ‘{kw}’ 와 *명백히 반대*인 영상만 골라 빼라"
              "(예: 움직임이 필수인 상황인데 동작측정이 거의 정지). 관찰은 기계 "
              "측정이라 틀릴 수 있고, 관찰이 부족하거나 애매한 것은 모순이 아니다 — "
              "확실치 않으면 빼지 마라(빈 배열).")
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0}, format=schema, think=False)
    try:
        rm = set(json.loads(r.message.content).get("remove", []))
    except json.JSONDecodeError:
        rm = set()
    return [n for n in cand if n not in rm]


# --------------------------------------------------------------------------- #
# 장면 전파 — 확정 영상과 "같은 장소"인 미매칭 영상에 태그 확장 (이중 AND 게이트)
# --------------------------------------------------------------------------- #
# 원리(2026-07-03 스파이크, 골든 9영상): 같은 장소 = 같은 배경 AND 같은 소리풍경.
# 단일 채널은 전부 함정이 목표와 같은 높이 —
#   시각(전역): 동종 실내·산책길 0.78~0.80 ≈ 목표 0.80 → 마스킹으로 실내 함정만 해소
#   기하(ORB+RANSAC): 마룻바닥 반복무늬 가짜 정합(다른 실내 66 > 같은 카페 16) 기각
#   오디오 단독: 말소리 편재(말소리 짝 0.88 > 목표 0.78)
# AND 게이트에선 두 채널의 실패 모드가 서로를 차단한다(M4 logprob×모션 교차와 동형):
#   목표 0.71/0.78 통과 · 실내 함정 0.52(시각 차단) · 산책길 0.80/0.35(오디오 차단)
#   · 말소리 짝 0.25(시각 차단). 기증자는 1단 매칭 확정분만(전파 체이닝 금지).
PROP_TAU_VIS = 0.65   # [잠정 n=9: 목표 0.71 vs 최고 함정 0.52]
PROP_TAU_AUD = 0.70   # [잠정 n=9: 목표 0.78 vs 통과시각쌍 최고 0.60]


def _cos(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def same_place(pa: dict, pb: dict,
               tau_vis: float = PROP_TAU_VIS, tau_aud: float = PROP_TAU_AUD) -> tuple:
    """두 프로필이 '같은 장소'인가 — (판정, 시각cos, 오디오cos). 둘 다 임계 이상."""
    vis = _cos(pa.get("scene_vec") or [], pb.get("scene_vec") or [])
    aud = _cos(pa.get("audio_vec") or [], pb.get("audio_vec") or [])
    return vis >= tau_vis and aud >= tau_aud, vis, aud


def propagate_tags(profiles: dict[str, dict],
                   matched: dict[str, list[str]]) -> dict[str, list[tuple]]:
    """{키워드: [(영상, 기증영상, 시각cos, 오디오cos)]} — 매칭 확장분만 반환.

    같은 방문에서 여러 클립을 찍는 임보 영상 분포에 맞춘 재현 보강. 기증자는
    이 호출의 matched(1단 매칭+프루닝 확정분)로 한정 — 전파분이 다시 기증하는
    체이닝은 금지(사진 앵커 2패스와 같은 보수 원칙).
    """
    adds: dict[str, list[tuple]] = {}
    for kw, donors in matched.items():
        if not donors:
            continue
        for u in profiles:
            if u in donors:
                continue
            for d in donors:
                ok, vis, aud = same_place(profiles[u], profiles[d])
                if ok:
                    adds.setdefault(kw, []).append((u, d, round(vis, 3), round(aud, 3)))
                    break
    return adds
