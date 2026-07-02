"""M6 모드 B — 편집요청 → Gemma 인텐트 → M4 태그로 클립선택 → ffmpeg 렌더.

설계: LLM 은 인텐트 번역만. 컴파일러·렌더러는 전부 결정론 코드.
검증 성격: 정확도가 아니라 *엔지니어링*(컷·xfade·크롭·줌·한글번인이 안 깨지고 렌더되는지).

연산(베이스라인):
  - 컷/트림 + 트랜지션(cut|xfade)
  - subject=foster: 임보견 박스로 크롭(주인공 당기기)
  - zoom=gradual: 임보견 중심 점차 확대(Ken Burns)
  - speed: 슬로우/배속(최종 패스 전역 적용)
  - title: 한글 제목 번인(PIL 렌더 → ffmpeg overlay; 이 ffmpeg 빌드엔 drawtext 없음)

임보견 박스는 foster_map + GT 보간(측정 격리). 제품에선 M2 re-ID pred 로 교체 가능.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import EditBlock, EditPlan, PLAN_SCHEMA
from ..harness import io
from ..m4_action import foster_track, is_trick
from ..m4_action.gt_scaffold import foster_boxes as _foster_boxes_provider
from ..workspace import Workspace

MODEL = "gemma4:26b-a4b-it-q4_K_M"
# 묘기 블록은 gemma 가 이 신뢰도 이상으로 확신한 재주만 선택(저신뢰 묘기 오판 누수 방지).
TRICK_CONF = 0.6
_XFADE_DUR = 0.5
_KFONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
_FILL_BLUR = 28   # 블러 배경 채움 강도(방향 안 맞는 클립)

# 캐시는 (workspace root, name) 로 키링 — 한 프로세스가 dev/잡 여러 루트를 봐도 안 섞임.
_foster_cache: dict[tuple[str, str], dict] = {}


def _foster_boxes(name: str, ws: Workspace) -> dict:
    key = (str(ws.root), name)
    if key not in _foster_cache:
        try:
            _foster_cache[key] = _foster_boxes_provider(name, ws)  # GT 또는 pred
        except SystemExit:
            _foster_cache[key] = {}
    return _foster_cache[key]


# --------------------------------------------------------------------------- #
# 1) 자연어 → 인텐트
# --------------------------------------------------------------------------- #
def interpret_plan(request: str) -> EditPlan:
    """자연어 편집요청 → 구성형 EditPlan(전역 title + 순서 있는 블록들)."""
    import ollama
    prompt = (
        "너는 강아지 입양영상 자동편집기다. 편집요청을 *순서 있는 블록들*로 분해해라.\n"
        "각 블록 = 영상의 한 구간 연출. '먼저/그다음/마지막에' 같은 순서 표현은 "
        "블록을 나눠서 표현한다. 순서 언급이 없으면 블록 1개.\n"
        "블록 필드:\n"
        "- select: 활발/노는=dynamic, 잔잔/쉬는/자는=static, 다=all, "
        "재주/묘기/하이파이브/손/빵야/구르기/죽은척 같은 재주 장면=묘기\n"
        "- target_dur: 그 구간 목표 길이(초). 명시 안 했으면 0.\n"
        "- pace: 빠른컷=fast, 잔잔=calm\n"
        "- transition: 컷=cut, 부드러운전환/디졸브=xfade\n"
        "- subject: 임보견을 화면 주인공으로 당겨라=foster, 전체화면=full\n"
        "- zoom: 점점 확대/클로즈업=gradual(정적 구간에만), 아니면 none\n"
        "- speed: 슬로우모션=0.5, 보통=1, 빠르게=2\n"
        "- caption: 그 블록 동안 화면 아래 띄울 한글 자막. 없으면 빈 문자열.\n"
        "  ('~를 텍스트로 띄워줘/자막' 같은 말은 그 블록 caption 으로.)\n"
        "- keywords: 그 블록이 원하는 *장면/장소/상황* 키워드 배열. 예: 애견카페→[\"카페\"], "
        "산책→[\"산책\"], 비오는 날→[\"비\"], 밤→[\"밤\"], 하이파이브→[\"하이파이브\"]. "
        "장면 언급 없으면 빈 배열. (이걸로 어느 영상을 쓸지 거른다.)\n"
        "전역 title: 고객이 제목/전체 자막을 *명시적으로* 요청할 때만 그 문구를 넣어라. "
        "'~영상 만들자/만들어줘' 는 목적 문장이지 제목 요청이 아니다 — 그 경우 반드시 "
        "빈 문자열. (예: '우리 토리 소개 영상 만들자'→title=\"\", "
        "'제목으로 우리 토리 넣어줘'→title=\"우리 토리\")\n"
        "예) '얼굴 5초 클로즈업하며 \"안녕\" 자막, 그 다음 노는 거 10초 \"신나요\" 자막' →\n"
        "  blocks=[{select:static,zoom:gradual,target_dur:5,caption:\"안녕\"},"
        "{select:dynamic,pace:fast,target_dur:10,caption:\"신나요\"}]\n요청: " + request
    )
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0}, format=PLAN_SCHEMA, think=False)
    return EditPlan.from_dict(json.loads(r.message.content))


# --------------------------------------------------------------------------- #
# 2) 클립 선택 (결정론)
# --------------------------------------------------------------------------- #
def _time_overlap(a: tuple, b: tuple) -> bool:
    """두 클립 (mp4,t0,t1) 이 같은 소스에서 시간상 겹치나."""
    return a[0] == b[0] and min(a[2], b[2]) > max(a[1], b[1])


_scene_cache: dict[str, dict] = {}


def _scene_tags(ws: Workspace) -> dict:
    """영상별 장면 태그맵(사람 메타데이터). 개발=글로벌, 잡=meta.json. 없으면 빈 dict."""
    key = str(ws.root)
    if key not in _scene_cache:
        _scene_cache[key] = ws.scene_tags()
    return _scene_cache[key]


def _scene_filter(sources: list[tuple[str, str]], keywords: list,
                  ws: Workspace) -> list[tuple[str, str]]:
    """블록 keywords 와 태그가 맞는 소스만 남김. 매칭 없으면 전체 유지(빈 결과 방지).

    부분 일치 허용: '애견카페'⊃'카페' 처럼 한쪽이 다른쪽을 포함하면 매칭.
    """
    if not keywords:
        return sources
    tags = _scene_tags(ws)

    def hit(src_tags: set) -> bool:
        return any(t in kw or kw in t for kw in keywords for t in src_tags)

    matched = [(m, p) for (m, p) in sources if hit(tags.get(Path(m).stem, set()))]
    return matched or sources


def _free_intervals(mp4: str, s0: float, s1: float, exclude: set) -> list[tuple[float, float]]:
    """구간 [s0,s1] 에서 exclude(같은 mp4 가 이미 쓴 부분)를 뺀 자유 구간들."""
    used = sorted((e[1], e[2]) for e in exclude
                  if e[0] == mp4 and min(e[2], s1) > max(e[1], s0))
    free, cur = [], s0
    for u0, u1 in used:
        if u0 > cur:
            free.append((cur, min(u0, s1)))
        cur = max(cur, u1)
    if cur < s1:
        free.append((cur, s1))
    return [iv for iv in free if iv[1] - iv[0] >= 0.6]


def compile_editlist(intent: EditBlock, sources: list[tuple[str, str]],
                     ws: Workspace | None = None,
                     exclude: set | None = None) -> list[tuple[str, float, float]]:
    """소스(장면)당 *한 컷*만 — 같은 장면을 쪼개 흩뿌리거나 반복하지 않는다.

    사용자 선호: 들어간 장면 다시 안 넣고, 쓸 거면 그 장면을 길게 한 번. 그래서
    각 소스의 가장 긴 '자유(미사용) 매칭 구간'에서 클립 하나만 뽑는다.
    길이는 target_dur 를 소스 수로 나눠 분배(없으면 pace 기본). 줌 블록은 단일 클립.
    """
    ws = ws or Workspace.dev()
    min_clip = 0.6
    exclude = exclude or set()
    # 사용자가 키워드로 장면을 콕 집었으면(밤·카페 등) 그 소스는 M4 uncertain 이어도 쓴다.
    # (핀 우선 > M4 필터 — 검출 희소한 밤 footage 도 요청했으면 포함.)
    pinned = bool(intent.keywords)
    sources = _scene_filter(sources, intent.keywords, ws)   # 장면 키워드로 소스 거르기

    # 소스별 가장 긴 자유 매칭 구간 하나
    cand: list[tuple[str, float, float]] = []
    for mp4, preds in sources:
        free: list[tuple[float, float]] = []
        for s in io.read_action_segments(preds):
            if intent.select == "묘기":          # 묘기: 군 무관, gemma 가 확신한 재주만
                if not (is_trick(s.action) and s.conf >= TRICK_CONF):
                    continue                     # (군 모호 uncertain 이어도 묘기는 씀)
            else:
                if intent.select != "all" and s.group != intent.select:
                    continue
                if s.uncertain and not pinned:   # 동/정 블록은 군 모호 제외(단 키워드 핀은 통과)
                    continue
            free += _free_intervals(mp4, s.start_t, s.end_t, exclude)
        if free:
            t0, t1 = max(free, key=lambda iv: iv[1] - iv[0])
            cand.append((mp4, t0, t1))
    if not cand:
        return []

    if intent.zoom == "gradual":          # 단일 연속 줌: 가장 긴 후보 하나만
        mp4, s0, s1 = max(cand, key=lambda c: c[2] - c[1])
        length = min(intent.target_dur or (s1 - s0), s1 - s0)
        return [(mp4, round(s0, 2), round(s0 + length, 2))]

    # 비줌: 소스마다 한 컷(연속). 길이 = target_dur / 소스수 (없으면 pace 기본).
    per = ((intent.target_dur / len(cand)) if intent.target_dur
           else (2.0 if intent.pace == "fast" else 4.0))
    # 짧은 블록 + 매칭 소스 과다 → 컷당 길이가 min_clip 미달로 전부 탈락하는 엣지
    # (모드 A 구절 길이 블록에서 실측). 소스 수를 줄여 한 컷을 길게 유지한다.
    if intent.target_dur and per < min_clip:
        k = max(1, int(intent.target_dur / min_clip))
        cand = sorted(cand, key=lambda c: c[2] - c[1], reverse=True)[:k]
        per = intent.target_dur / len(cand)
    clips: list[tuple[str, float, float]] = []
    for mp4, s0, s1 in cand:
        length = min(per, s1 - s0)
        if length >= min_clip:
            clips.append((mp4, round(s0, 2), round(s0 + length, 2)))
    return clips


# --------------------------------------------------------------------------- #
# 3) 프레이밍(크롭/줌) 변환 빌드
# --------------------------------------------------------------------------- #
def _probe_dims(mp4: str) -> tuple[int, int]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", mp4],
                       capture_output=True, text=True)
    w, h = r.stdout.strip().split("x")[:2]
    return int(w), int(h)


def _foster_span(name: str, t0: float, t1: float, fps: float, ws: Workspace):
    """클립 구간의 임보견 박스 합(union) → (cx, cy, w, h) src px. 없으면 None."""
    boxes = _foster_boxes(name, ws)
    f0, f1 = int(t0 * fps), int(t1 * fps)
    bs = [boxes[i] for i in range(f0, f1 + 1) if i in boxes]
    if not bs:
        return None
    x1 = min(b.x for b in bs); y1 = min(b.y for b in bs)
    x2 = max(b.x2 for b in bs); y2 = max(b.y2 for b in bs)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def _src_map(name: str, ws: Workspace):
    """편집용 소스 + 분석→원본 배율 + 원본(표시)크기.

    설계서: '분석용 ≠ 편집용. 출력은 원본 그대로.' 원본 .MOV + map.json 있으면 원본
    고해상도로 렌더(분석 박스 × scale = 원본 좌표). 없으면 분석 mp4(scale=1) 폴백.
    회전은 normalize 가 픽셀에 구워뒀고 ffmpeg 추출도 autorotate 라 방향 일치.
    개발은 videos/, 잡은 input/ 에서 원본을 찾는다(ws.source_video).
    """
    mov = ws.source_video(name)
    if mov is not None:
        mp = ws.analysis_map(name)
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            ow, oh = m["orig_size"]
            return str(mov), float(m["scale_analysis_to_orig"]), int(ow), int(oh)
    a = str(ws.analysis(name))
    w, h = _probe_dims(a)
    return a, 1.0, w, h


def _normalize_fill(src: Path, out: Path, W: int, H: int, fps: float):
    """블러 배경 채움으로 WxH 정규화 — 잘림0·검은띠0(쇼츠 룩).

    fg=원클립을 비율유지로 맞춰 중앙, bg=같은 클립을 꽉 채워 크롭+블러로 빈공간 채움.
    """
    fc = (f"[0:v]split[bg][fg];"
          f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
          f"gblur=sigma={_FILL_BLUR}[bgb];"
          f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease[fgs];"
          f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps}[v]")
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-filter_complex", fc, "-map", "[v]",
                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    str(out)], check=True, capture_output=True)


def _extract_native(intent: EditBlock, name: str, t0: float, t1: float,
                    out_native: Path, W: int, H: int, fps: float, ws: Workspace):
    """원본에서 [t0,t1] 고해상도 추출(autorotate). *크롭하지 않음* — 클립을 통째로
    뽑고, 방향 차이는 stage2 블러채움이 처리(잘림0). 가로 클립이 잘리지 않게 하는 핵심.
    """
    src, _, _, _ = _src_map(name, ws)
    # -noautorotate: P0 와 동일하게 저장 픽셀 기준(소스 회전 태그가 잘못돼도 분석과 방향 일치).
    subprocess.run(["ffmpeg", "-y", "-noautorotate", "-ss", f"{t0}", "-to", f"{t1}", "-i", src,
                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    str(out_native)], check=True, capture_output=True)


def _opencv_zoom(intent: EditBlock, name: str, t0: float, t1: float,
                 out_native: Path, fps: float, ws: Workspace) -> bool:
    """임보견 얼굴로 *고정 영역* 매끄러운 줌인 — 원본 고해상도 위에서.

    얼굴 추적 안 함: 시작 시점 임보견 박스로 끝 줌영역 한 번 고정, 전체→그영역
    smoothstep 보간. float 어파인+Lanczos 서브픽셀로 떨림 제거. 출력은 네이티브 크기
    (이후 _normalize_fill 이 WxH 로). 박스 없으면 False.
    """
    boxes = _foster_boxes(name, ws)
    f0, f1 = int(t0 * fps), int(t1 * fps)
    near = [i for i in boxes if f0 <= i <= f1]
    if not near:
        return False
    src, scale, ow, oh = _src_map(name, ws)
    b = boxes[min(near, key=lambda i: abs(i - f0))]
    nat = out_native.with_suffix(".nat.mp4")   # 원본 추출(-noautorotate, cv2 회전이슈 회피)
    subprocess.run(["ffmpeg", "-y", "-noautorotate", "-ss", f"{t0}", "-to", f"{t1}", "-i", src,
                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    str(nat)], check=True, capture_output=True)
    cap = cv2.VideoCapture(str(nat))
    NW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); NH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fcx = ((b.x + b.x2) / 2) * scale
    fcy = (b.y + 0.35 * (b.y2 - b.y)) * scale        # 얼굴 ≈ 박스 상단 35%
    arN = NW / NH
    end_h = min(NH, (b.y2 - b.y) * scale * 1.4); end_w = end_h * arN
    if end_w > NW:
        end_w = NW; end_h = end_w / arN
    fcx = min(max(fcx, end_w / 2), NW - end_w / 2)
    fcy = min(max(fcy, end_h / 2), NH - end_h / 2)

    wr = cv2.VideoWriter(str(out_native), cv2.VideoWriter_fourcc(*"mp4v"), fps, (NW, NH))
    n = max(2, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (f1 - f0))
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        a = i / (n - 1); a = a * a * (3 - 2 * a)        # smoothstep
        w = NW + (end_w - NW) * a; h = NH + (end_h - NH) * a
        cx = NW / 2 + (fcx - NW / 2) * a; cy = NH / 2 + (fcy - NH / 2) * a
        S = NW / w
        M = np.float32([[S, 0, -(cx - w / 2) * S], [0, S, -(cy - h / 2) * S]])
        wr.write(cv2.warpAffine(fr, M, (NW, NH), flags=cv2.INTER_LANCZOS4))
    cap.release(); wr.release()
    nat.unlink(missing_ok=True)
    return True


def _extract_clip(intent: EditBlock, mp4: str, t0: float, t1: float,
                  out: Path, size: tuple[int, int], fps: float, ws: Workspace):
    """원본 고해상도에서 연산 적용(stage1 네이티브) → 블러채움 WxH 정규화(stage2)."""
    name = Path(mp4).stem
    W, H = size
    native = out.with_suffix(".native.mp4")
    if not (intent.zoom == "gradual" and _opencv_zoom(intent, name, t0, t1, native, fps, ws)):
        _extract_native(intent, name, t0, t1, native, W, H, fps, ws)
    _normalize_fill(native, out, W, H, fps)
    native.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# 4) 제목 번인 (PIL → overlay) / 배속
# --------------------------------------------------------------------------- #
def _title_png(text: str, W: int, H: int, out: Path):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(_KFONT, max(28, W // 18))
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x, y = (W - tw) // 2, H - th - max(30, H // 12)
    d.rectangle([x - 18, y - 14, x + tw + 18, y + th + 18], fill=(0, 0, 0, 150))
    d.text((x - bb[0], y - bb[1]), text, font=font, fill=(255, 255, 255, 255))
    img.save(out)


def _stitch(parts: list[Path], durs: list[float], transition: str, out: Path):
    """클립들을 transition(cut|xfade)으로 이어 out 으로."""
    if transition == "xfade" and len(parts) >= 2:
        inputs = []
        for p in parts:
            inputs += ["-i", str(p)]
        fc, prev, offset = [], "0:v", 0.0
        for i in range(1, len(parts)):
            offset += durs[i - 1] - _XFADE_DUR
            lbl = f"x{i}"
            fc.append(f"[{prev}][{i}:v]xfade=transition=fade:duration={_XFADE_DUR}:"
                      f"offset={offset:.3f}[{lbl}]")
            prev = lbl
        subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(fc),
                        "-map", f"[{prev}]", "-c:v", "libx264", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", str(out)], check=True, capture_output=True)
    else:
        lst = out.with_suffix(".txt")
        lst.write_text("".join(f"file '{p}'\n" for p in parts), encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                        str(out)], check=True, capture_output=True)


def _apply_speed(src: Path, speed: float, out: Path):
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vf", f"setpts=PTS/{speed}",
                    "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    str(out)], check=True, capture_output=True)


def _overlay_text(src: Path, text: str, out: Path, size: tuple[int, int]):
    """src 영상 아래에 한글 자막(PIL→overlay) 박아 out 으로. 빈 텍스트면 그대로 복사."""
    if not text:
        src.replace(out); return
    png = out.with_name(out.stem + "_txt.png")
    _title_png(text, size[0], size[1], png)
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-i", str(png),
                    "-filter_complex", "[0:v][1:v]overlay=0:0", "-c:v", "libx264",
                    "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)],
                   check=True, capture_output=True)
    png.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# 5) 렌더 (블록 → plan)
# --------------------------------------------------------------------------- #
def render_block(block: EditBlock, sources, tmp: Path, idx: int,
                 size=(768, 432), fps=30.0, exclude: set | None = None,
                 ws: Workspace | None = None):
    """블록 1개 렌더(클립선택 → 연산 → 전환 → 배속 → 자막). 반환 (영상경로, 쓴 클립들).
    클립 없으면 (None, [])."""
    ws = ws or Workspace.dev()
    clips = compile_editlist(block, sources, ws, exclude)
    if not clips:
        return None, []
    parts = []
    for i, (mp4, t0, t1) in enumerate(clips):
        p = tmp / f"b{idx}_clip{i:03d}.mp4"
        _extract_clip(block, mp4, t0, t1, p, size, fps, ws)
        parts.append(p)
    vid = tmp / f"b{idx}_stitched.mp4"
    _stitch(parts, [c[2] - c[1] for c in clips], block.transition, vid)
    if block.speed != 1.0:
        sped = tmp / f"b{idx}_speed.mp4"
        _apply_speed(vid, block.speed, sped); vid = sped
    if block.caption:                       # 블록별 자막을 그 블록에만 박는다
        cap = tmp / f"b{idx}_cap.mp4"
        _overlay_text(vid, block.caption, cap, size); vid = cap
    return vid, clips


def render_plan(plan: EditPlan, sources, out_path: str, size=(768, 432), fps=30.0,
                ws: Workspace | None = None):
    """블록들을 순서대로 렌더(클립 중복 방지) → 이어붙이기 → 전역 title(있으면)."""
    ws = ws or Workspace.dev()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        block_vids, used = [], set()
        for idx, block in enumerate(plan.blocks):
            bv, clips = render_block(block, sources, tmp, idx, size, fps, exclude=used, ws=ws)
            if bv is not None:
                block_vids.append((bv, _probe_dur(str(bv))))
                used.update(clips)          # 다음 블록이 같은 클립 안 쓰게
        if not block_vids:
            raise SystemExit("선택된 클립이 없음 (어떤 블록도 조건에 맞는 구간 없음).")
        stitched_all = tmp / "all.mp4"
        _stitch([b for b, _ in block_vids], [d for _, d in block_vids], "cut", stitched_all)
        _overlay_text(stitched_all, plan.title, Path(out_path), size)


def _probe_dur(path: str) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nk=1:nw=1", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return -1.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="m6_run")
    p.add_argument("request", help="자연어 편집요청")
    p.add_argument("--sources", nargs="+", required=True,
                   help="소스 이름들 (data/analysis/{name}.mp4 + data/preds/{name}_m4.json)")
    p.add_argument("--out", default="data/dev/demo/m6_modeB.mp4")
    p.add_argument("--size", default="1080x1920")   # 세로 9:16 (원본 고해상도)
    args = p.parse_args(argv)

    w, h = (int(x) for x in args.size.split("x"))
    ws = Workspace.dev()
    plan = interpret_plan(args.request)
    print(f"[plan] title={plan.title!r}  블록 {len(plan.blocks)}개")
    sources = [(str(ws.analysis(n)), str(ws.preds_m4(n))) for n in args.sources]
    for i, b in enumerate(plan.blocks):
        print(f"  블록{i}: select={b.select} dur={b.target_dur} pace={b.pace} "
              f"trans={b.transition} subj={b.subject} zoom={b.zoom} speed={b.speed} "
              f"keywords={b.keywords} caption={b.caption!r}")
    render_plan(plan, sources, args.out, (w, h), ws=ws)
    print(f"[렌더] → {args.out}  (실측 길이 {_probe_dur(args.out):.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
