"""잡 엔트리포인트 — 요청당 1개 영상 파이프라인 (상태머신).

설계서 '백엔드 ↔ 파이프라인 계약'을 코드로 구현. 임보견 선택이 사람 개입
체크포인트라, 잡은 fire-and-forget 이 아니라 2단계 상태 머신이다:

  prepare(ws)            → P0 정규화 + M1 검출 → 임보견 후보(단독견이면 자동확정)
  [다견이면 고객이 트랙 선택 → meta 저장]   ← Phase 3(카드)
  render(ws, request)    → M4 동작판정 + M6 편집 → out/

개발 골든셋(data/dev)과 달리 잡은 GT 불필요(pred 경로)·자족·격리 — Workspace(root)가
경로를 잡 디렉토리 안으로 가둔다.

상태: uploaded → validated → prepared → (needs_foster_pick | auto) → rendering → done

단독견 자동확정 주의: M1 은 영상마다 독립 실행이라 track id 가 영상 간 공유되지
않는다. 그래서 단독견은 meta.foster_track 을 비워 두고(=None), foster_boxes_pred 가
영상별 '최대 dog 박스' 휴리스틱으로 자족 처리한다. meta.foster_track 은 단일영상
다견(고객이 트랙 1개 선택)일 때만 의미를 갖는다.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from .harness import io
from .m1_track.run import run as m1_run
from .m4_action.run import run as m4_run
from .m5_tts import DEFAULT_VOICE
from .m5_tts.interpret import decide_mode, interpret_narration
from .m5_tts.render import render_narrated
from .m6_edit.badge import apply_badge
from .m6_edit.run import _probe_dur, interpret_plan, render_plan
from .preprocess.normalize import normalize
from .workspace import Workspace

_VIDEO_EXTS = ("MOV", "mov", "mp4", "MP4", "m4v", "avi")


# --------------------------------------------------------------------------- #
# 잡 생성 (백엔드 역할의 최소 구현 — 디렉토리 + 업로드 저장 + meta 초기화)
# --------------------------------------------------------------------------- #
def init_job(job_id: str, inputs: list[str | Path], data_root: str | Path | None = None) -> Workspace:
    """잡 디렉토리 생성 + 입력 영상을 input/ 으로 복사 + meta 초기화(state=uploaded)."""
    ws = Workspace.job(job_id, data_root)
    ws.input_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for src in inputs:
        src = Path(src)
        if not src.exists():
            raise SystemExit(f"입력 영상 없음: {src}")
        dst = ws.input_dir / src.name
        shutil.copy2(src, dst)
        saved.append(dst.stem)
    ws.write_meta({"job_id": job_id, "state": "uploaded", "inputs": saved})
    print(f"[init] {ws.root}  입력 {len(saved)}개: {saved}")
    return ws


def _input_names(ws: Workspace) -> list[str]:
    """input/ 안의 영상 stem 목록(확장자 우선순위·순서 보존, 중복 제거)."""
    names: list[str] = []
    seen: set[str] = set()
    for ext in _VIDEO_EXTS:
        for p in sorted(ws.input_dir.glob(f"*.{ext}")):
            if p.stem not in seen:
                seen.add(p.stem)
                names.append(p.stem)
    return names


# --------------------------------------------------------------------------- #
# 1단계 — prepare: P0 + M1 + 임보견 후보
# --------------------------------------------------------------------------- #
def _dog_tracks(ws: Workspace, name: str, min_frac: float) -> dict[int, int]:
    """영상의 유의미한 dog 트랙 → 등장 프레임 수. (min_frac 미만 등장은 노이즈로 배제)"""
    frames = io.read_frames(ws.preds_m1(name))
    n = max(1, len(frames))
    counts: Counter[int] = Counter()
    for f in frames:
        for d in f.detections:
            if d.cls == "dog":
                counts[d.track_id] += 1
    return {tid: c for tid, c in counts.items() if c / n >= min_frac}


def prepare(ws: Workspace, weights: str = "yolo11m.pt", conf: float = 0.25,
            min_track_frac: float = 0.1) -> dict:
    """P0 정규화 + M1 검출 → 임보견 후보 판정. 단독견이면 자동확정.

    반환 meta(갱신본). state = prepared(단독견 자동) | needs_foster_pick(다견).
    """
    names = _input_names(ws)
    if not names:
        raise SystemExit(f"{ws.root}: input/ 에 영상이 없음. init_job 먼저.")
    ws.update_meta(state="validated", sources=names)

    for name in names:
        src = ws.source_video(name)
        print(f"[P0] {name} 정규화…")
        normalize(src, ws.analysis_dir)
        print(f"[M1] {name} 검출…")
        s = m1_run(str(ws.analysis(name)), str(ws.preds_m1(name)), weights=weights, conf=conf)
        print(f"     frames={s['frames']} boxes={s['boxes']} tracks={s['tracks']}")

    # 임보견 후보: 영상별 유의미 dog 트랙 수. 어느 영상이든 2마리+면 다견.
    per_video = {name: _dog_tracks(ws, name, min_track_frac) for name in names}
    max_dogs = max((len(t) for t in per_video.values()), default=0)
    candidates = {name: sorted(t, key=t.get, reverse=True) for name, t in per_video.items()}

    if max_dogs <= 1:
        meta = ws.update_meta(state="prepared", foster_auto=True, foster_track=None,
                              dog_candidates=candidates)
        print(f"[prepare] 단독견 자동확정 ✓  (영상별 트랙수 {{ {', '.join(f'{k}:{len(v)}' for k,v in per_video.items())} }})")
    else:
        meta = ws.update_meta(state="needs_foster_pick", foster_auto=False,
                              dog_candidates=candidates)
        print(f"[prepare] 다견 감지 — 임보견 선택 필요(Phase 3 카드). 후보: {candidates}")
    return meta


# --------------------------------------------------------------------------- #
# 2단계 — render: M4 + M6 → out/
# --------------------------------------------------------------------------- #
def render(ws: Workspace, request: str, size: tuple[int, int] = (1080, 1920),
           fps: float = 30.0, thr: float = 8.0, conf_thr: float = 0.6,
           max_crops: int = 10, out_name: str = "final.mp4",
           voice: str | None = None) -> Path:
    """M4 동작판정(소스별) + 모드 라우팅(A=내레이션/B=편집만) → out/.

    모드는 meta.mode(카드/수동 확정) 우선, 없으면 decide_mode 3단(핀→logprob→uncertain).
    uncertain 은 needs_mode_pick 상태로 멈춤 — "자막인지 음성인지" 고객 카드 1탭 대상.
    보이스는 voice 인자 > meta.voice(고객 선택) > 기본 eric — 잡 단위 통일 정책.
    M4 태그는 요청과 무관한 영상 분석이라 재렌더 시 재사용(설계서: '한 번 분석해두면
    내레이션 여러 버전에 재사용').
    """
    meta = ws.read_meta()
    names = meta.get("sources")
    if not names:
        raise SystemExit(f"{ws.root}: prepare 가 안 끝났음(sources 없음).")
    if meta.get("state") == "needs_foster_pick":
        raise SystemExit(f"{ws.root}: 임보견 선택 대기 중. meta.foster_track 설정 후 재시도.")

    mode = meta.get("mode")
    if mode in ("narration", "edit"):
        print(f"[모드] {mode} (meta 지정)")
    else:
        mode, conf = decide_mode(request)
        if mode == "uncertain":
            ws.update_meta(state="needs_mode_pick", request=request)
            raise SystemExit(
                f"{ws.root}: 자막/음성 모호(확신 {conf:.2f}) — 고객 카드 대상. "
                "meta.mode 를 'narration' 또는 'edit' 로 설정 후 재시도.")
        print(f"[모드] {mode} (확신 {conf:.2f})")

    voice = voice or meta.get("voice") or DEFAULT_VOICE
    ws.update_meta(state="rendering", request=request, mode=mode, voice=voice)
    for name in names:
        if ws.preds_m4(name).exists():
            print(f"[M4] {name} 태그 재사용")
            continue
        print(f"[M4] {name} 동작판정…")
        m4_run(name, thr, fps, conf_thr, max_crops, ws=ws)

    sources = [(str(ws.analysis(n)), str(ws.preds_m4(n))) for n in names]
    out_path = ws.out(out_name)
    if mode == "narration":
        print("[M5+M6] 대본 분해·합성·렌더…")
        plan = interpret_narration(request)
        n_narr = sum(1 for b in plan.blocks if b.narration)
        print(f"     블록 {len(plan.blocks)}개 (내레이션 {n_narr}구절, 보이스 {voice})")
        render_narrated(plan, sources, str(out_path), size, fps, ws=ws, voice=voice)
    else:
        print("[M6] 편집 인텐트 해석…")
        plan = interpret_plan(request)
        print(f"     title={plan.title!r}  블록 {len(plan.blocks)}개")
        raw = out_path.with_name("_raw_" + out_name)
        render_plan(plan, sources, str(raw), size, fps, ws=ws)
        apply_badge(raw, out_path, tts=False, size=size)   # 모드 B 도 "AI 편집" 배지
        raw.unlink(missing_ok=True)
    ws.update_meta(state="done", out=str(out_path))
    print(f"[done] → {out_path}  (실측 {_probe_dur(str(out_path)):.1f}s)")
    return out_path


# --------------------------------------------------------------------------- #
# CLI — 잡 1개를 끝까지 (init → prepare → render)
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="job", description="요청당 1개 잡 실행")
    p.add_argument("job_id", help="잡 식별자 (→ $DATA_ROOT/<job_id>)")
    p.add_argument("--inputs", nargs="+", help="입력 영상 경로들(주면 init_job 부터)")
    p.add_argument("--request", help="자연어 편집요청(주면 render 까지)")
    p.add_argument("--prepare-only", action="store_true", help="prepare 까지만")
    p.add_argument("--data-root", default=None, help="잡 저장 루트(기본 $DATA_ROOT 또는 ./jobs)")
    p.add_argument("--size", default="1080x1920")
    p.add_argument("--weights", default="yolo11m.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--voice", default=None,
                   help="TTS 보이스(모드 A, 잡 단위 통일). 기본 meta.voice 또는 eric")
    args = p.parse_args(argv)

    if args.inputs:
        ws = init_job(args.job_id, args.inputs, args.data_root)
    else:
        ws = Workspace.job(args.job_id, args.data_root)
        if not ws.meta_path.exists():
            raise SystemExit(f"{ws.root}: 기존 잡 없음. --inputs 로 생성하세요.")

    meta = prepare(ws, weights=args.weights, conf=args.conf)

    if args.prepare_only:
        return 0
    if meta.get("state") == "needs_foster_pick":
        print("→ 다견: meta.json 에 foster_track 설정 후 render 단계 재실행 필요.")
        return 0
    if not args.request:
        print("→ --request 미지정: prepare 까지 완료. render 는 --request 와 함께 재실행.")
        return 0

    w, h = (int(x) for x in args.size.split("x"))
    render(ws, args.request, (w, h), voice=args.voice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
