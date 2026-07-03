"""상비 관찰(관찰 프로필) — 요청과 무관한 영상별 사실을 센서+캡션으로 1회 수집.

설계(2026-07-03, Notion '영상 구현 — 장면 추론'): "속성은 단어가 아니라 픽셀(과
오디오)에서". 죽은 건 고정 *어휘*(카페/산책…)지 고정 *센서*(닫힌 물리 축)가 아니다 —
센서는 상황이 아니라 축 전체를 커버하므로 미리 정해도 사각이 없다(M3 전례).

축별 스파이크 채점(2026-07-03, 골든 9영상):
  오디오(AST/AudioSet)  — 카페 특효: Speech 0.66~0.74 + Music 0.40~0.68 이 카페 2영상
                          에서만(비카페 Music ≤0.03). 스틸 4형식 전패 축이 오디오로
                          뚫림. 비도 통과(Rain·Ocean/Waves 물소리 0.6+, 타영상 ≤0.08).
                          실내/실외 클래스는 전부 <0.12 → 이 축은 못 가름(Places 몫).
                          지뢰: 개 발소리→Horse/Clip-clop 오분류 — 관찰은 그대로 적고
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
# 카페 0/2 → gloss 후 1/2). 표에 없는 라벨은 원문 그대로(안전한 폴백).
AUDIO_KO = {
    "Speech": "사람 말소리", "Music": "음악", "Cat": "고양이 소리", "Meow": "야옹 소리",
    "Dog": "개 소리", "Bark": "짖는 소리", "Bow-wow": "멍멍 소리",
    "Rain": "빗소리", "Rain on surface": "표면에 떨어지는 빗소리",
    "Ocean": "물 흐르는/파도 소리", "Waves, surf": "물결 소리", "Stream": "물 흐르는 소리",
    "Wind": "바람 소리", "Wind noise (microphone)": "바람 소리(마이크)",
    "Run": "달리는 소리", "Walk, footsteps": "발걸음 소리",
    "Typing": "타자 소리", "Computer keyboard": "키보드 소리", "Television": "TV 소리",
    "Inside, small room": "실내(좁은 방) 울림",
    "Inside, large room or hall": "실내(홀) 울림",
    "Outside, rural or natural": "실외(자연) 소리",
    "Outside, urban or manmade": "실외(도심) 소리",
    "Vehicle": "차량 소리", "Car": "자동차 소리", "Animal": "동물 소리",
    "Domestic animals, pets": "반려동물 소리",
    "Livestock, farm animals, working animals": "가축류 소리",
    "Horse": "말발굽 같은 소리", "Clip-clop": "따각따각 발소리",
    "Tambourine": "탬버린 소리", "Jingle bell": "방울 소리",
    "Squawk": "새된 울음", "Caterwaul": "고양이 울음",
    "Child speech, kid speaking": "아이 말소리",
    "Male speech, man speaking": "남자 말소리",
    "Female speech, woman speaking": "여자 말소리",
    "Bird": "새 소리", "Silence": "정적", "Crowd": "군중 소리",
    "Chatter": "웅성거림", "Laughter": "웃음소리", "Water": "물소리",
}
PLACES_KO = {
    "veterinarians_office": "동물병원/동물시설", "pet_shop": "펫샵", "kennel/outdoor": "견사(실외)",
    "closet": "옷장/실내 수납", "corridor": "복도", "artists_loft": "작업실 로프트",
    "recreation_room": "오락실/거실", "martial_arts_gym": "체육관", "art_gallery": "갤러리",
    "playroom": "놀이방", "reception": "리셉션", "bathroom": "욕실",
    "living_room": "거실", "home_office": "홈오피스", "bedroom": "침실",
    "yard": "마당", "park": "공원", "field_road": "들길", "forest_path": "숲길",
    "swimming_pool/indoor": "실내 수영장", "swimming_pool/outdoor": "실외 수영장",
    "creek": "개울/계곡", "river": "강", "beach": "해변",
}


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


def _ast():
    if not _ast_cache:
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        fe = ASTFeatureExtractor.from_pretrained(AST_MODEL)
        model = ASTForAudioClassification.from_pretrained(AST_MODEL)
        model.eval()
        _ast_cache.append((fe, model))
    return _ast_cache[0]


def audio_tags(src_video: str | Path) -> list[list]:
    """원본 영상 오디오 → [[클래스, prob], ...] (max 집계, 임계 이상 top-N)."""
    import soundfile as sf
    import torch
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src_video),
             "-vn", "-ac", "1", "-ar", str(_AUDIO_SR), tmp.name],
            capture_output=True, text=True)
        if r.returncode != 0:
            return []                        # 오디오 트랙 없음 등 — 관찰 없음
        wav, sr = sf.read(tmp.name, dtype="float32")
    if len(wav) < _AUDIO_SR // 2:            # 0.5초 미만이면 신호가 못 됨
        return []
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
    id2label = model.config.id2label
    top = np.argsort(-probs)[:AUDIO_TOP_N]
    return [[id2label[int(i)], round(float(probs[i]), 3)]
            for i in top if probs[i] >= AUDIO_MIN_PROB]


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
              "추측(장소 이름 단정, 개의 감정)은 하지 마라.")
    r = ollama.chat(model=MODEL,
                    messages=[{"role": "user", "content": prompt, "images": jpegs}],
                    options={"temperature": 0, "num_predict": 200,
                             "repeat_penalty": 1.3},
                    format=_CAPTION_SCHEMA, think=False)
    try:
        return json.loads(r.message.content).get("caption", "").strip()
    except json.JSONDecodeError:
        return ""


# --------------------------------------------------------------------------- #
# 프로필 빌드 + meta 캐시
# --------------------------------------------------------------------------- #
def build_profile(ws: Workspace, name: str) -> dict:
    """영상 1개의 관찰 프로필. 축별 실패는 삼키고 빈 관찰(안전한 실패)."""
    prof: dict = {}
    src = ws.source_video(name)
    try:
        prof["audio"] = audio_tags(src) if src else []
    except Exception as e:                    # noqa: BLE001 — 센서가 prepare 를 못 죽인다
        print(f"     [관찰] {name} 오디오 실패: {e}")
        prof["audio"] = []
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
    try:
        prof["caption"] = caption(ws.analysis(name))
    except Exception as e:                    # noqa: BLE001
        print(f"     [관찰] {name} 캡션 실패: {e}")
        prof["caption"] = ""
    return prof


def ensure_profiles(ws: Workspace, names: list[str]) -> dict[str, dict]:
    """meta.scene_profile 캐시 — 없는 영상만 빌드(요청 무관 재사용 자산)."""
    meta = ws.read_meta()
    profiles = dict(meta.get("scene_profile") or {})
    todo = [n for n in names if n not in profiles]
    for name in todo:
        print(f"[관찰] {name} 프로필(오디오·장면·휘도·캡션)…")
        profiles[name] = build_profile(ws, name)
        print(f"     {profile_text(profiles[name])}")
    if todo:
        ws.update_meta(scene_profile=profiles)
    return {n: profiles[n] for n in names if n in profiles}


def profile_text(prof: dict, extra: str | None = None) -> str:
    """프로필 → 매칭 프롬프트용 한 줄 관찰 기록. 라벨은 한국어 gloss(순수 번역)."""
    parts = []
    audio = prof.get("audio") or []
    if audio:
        parts.append("소리: " + ", ".join(f"{AUDIO_KO.get(c, c)}({p})" for c, p in audio))
    else:
        parts.append("소리: (신호 없음)")
    places = prof.get("places") or {}
    cats = places.get("categories") or []
    seg = []
    if places.get("indoor"):
        seg.append("실내 확실")
    if cats:
        seg.append(", ".join(f"{PLACES_KO.get(c, c)}({p})" for c, p in cats))
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

    지시 레시피(골든 9영상 채점으로 확정, 2026-07-03):
      '뒷받침하면 포함, 모순·무관이면 제외' (완화) — '확실한 것만'(strict)은 빈 배열
      어트랙터로 카페·놀이·하이파이브 전멸(2/6), 완화+gloss+정성모션 = 4/6.
      think=True 는 기각(추론 배회로 가짜 생성·비결정성), 예/아니오 분할도 기각
      (가짜 유출 — 융합 vision 포함 재확인). extras = 영상별 추가 관찰(모션 등).
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
        out[kw] = [n for n in dict.fromkeys(picked) if n in profiles]
    return out
