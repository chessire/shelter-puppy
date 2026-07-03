"""잡 엔트리포인트 — 요청당 1개 영상 파이프라인 (상태머신).

설계서 '백엔드 ↔ 파이프라인 계약'을 코드로 구현. 강아지 선택이 사람 개입
체크포인트라, 잡은 fire-and-forget 이 아니라 2단계 상태 머신이다:

  prepare(ws)            → P0 정규화 + M1 검출 → 강아지 후보(단독견이면 자동확정)
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
_VID_SET = {e.lower() for e in _VIDEO_EXTS}
_IMG_SET = {"jpg", "jpeg", "png", "heic", "webp"}


def _expand_paths(paths: list[str | Path], exts: set[str], what: str) -> list[Path]:
    """--inputs/--dog-photos 의 파일·폴더 혼용 확장 — 폴더면 안의 해당 확장자 전부(이름순).

    "하나하나 입력하기 힘들다"(2026-07-03) → 폴더째 지정 지원. 하위 폴더는 안 탐(1단계만).
    """
    out: list[Path] = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_dir():
            found = sorted(q for q in p.iterdir()
                           if q.is_file() and q.suffix.lower().lstrip(".") in exts)
            if not found:
                raise SystemExit(f"{p}: 폴더에 {what} 파일이 없음 (지원: {sorted(exts)})")
            out += found
        else:
            out.append(p)
    return out


# --------------------------------------------------------------------------- #
# 잡 생성 (백엔드 역할의 최소 구현 — 디렉토리 + 업로드 저장 + meta 초기화)
# --------------------------------------------------------------------------- #
def init_job(job_id: str, inputs: list[str | Path], data_root: str | Path | None = None,
             dog_photos: list[str | Path] | None = None) -> Workspace:
    """잡 디렉토리 생성 + 입력 영상 input/ 복사 + 강아지 사진 refs/ 복사 + meta 초기화.

    dog_photos = 강아지 레퍼런스 사진(선택). 다견 감지 시 사진 앵커가 트랙을 자동
    지정한다 — 사진은 잡마다 탭할 필요 없는 1회 자산(고객 프로필감).
    """
    ws = Workspace.job(job_id, data_root)
    ws.input_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for src in _expand_paths(inputs, _VID_SET, "영상"):
        if not src.exists():
            raise SystemExit(f"입력 영상 없음: {src}")
        dst = ws.input_dir / src.name
        shutil.copy2(src, dst)
        saved.append(dst.stem)
    refs = []
    if dog_photos:
        ref_dir = ws.root / "refs"
        ref_dir.mkdir(parents=True, exist_ok=True)
        for src in _expand_paths(dog_photos, _IMG_SET, "사진"):
            if not src.exists():
                raise SystemExit(f"강아지 사진 없음: {src}")
            shutil.copy2(src, ref_dir / src.name)
            refs.append(src.name)
    ws.write_meta({"job_id": job_id, "state": "uploaded", "inputs": saved,
                   "dog_photos": refs})
    print(f"[init] {ws.root}  입력 {len(saved)}개: {saved}"
          + (f"  강아지 사진 {len(refs)}장" if refs else ""))
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
# 1단계 — prepare: P0 + M1 + 강아지 후보
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
    """P0 정규화 + M1 검출 → 강아지 후보 판정. 단독견이면 자동확정.

    반환 meta(갱신본). state = prepared(단독견 자동) | needs_foster_pick(다견).
    """
    names = _input_names(ws)
    if not names:
        raise SystemExit(f"{ws.root}: input/ 에 영상이 없음. init_job 먼저.")
    ws.update_meta(state="validated", sources=names)

    for name in names:
        # 재실행(다견 확정 후 등) 시 P0/M1 재계산 방지 — 분석은 영상당 1회(M4 재사용과 동일 결).
        # 재사용 조건에 '검출 0이 아님'을 포함 — P0 산출물이 깨지면(실측: NAL 오류 mp4)
        # M1 이 빈 pred 를 남기고, 존재 여부만 보면 그 깨진 캐시가 영구 고착된다.
        if (ws.analysis(name).exists() and ws.preds_m1(name).exists()
                and ws.preds_m1(name).stat().st_size > 0):
            print(f"[P0+M1] {name} 재사용")
            continue
        if ws.preds_m1(name).exists():
            print(f"[P0+M1] {name} 빈 pred 감지 — 정규화·검출 재시도")
            ws.analysis(name).unlink(missing_ok=True)
        src = ws.source_video(name)
        print(f"[P0] {name} 정규화…")
        normalize(src, ws.analysis_dir)
        print(f"[M1] {name} 검출…")
        s = m1_run(str(ws.analysis(name)), str(ws.preds_m1(name)), weights=weights, conf=conf)
        print(f"     frames={s['frames']} boxes={s['boxes']} tracks={s['tracks']}")

    # 강아지 후보: 영상별 유의미 dog 트랙 수. 어느 영상이든 2마리+면 다견.
    per_video = {name: _dog_tracks(ws, name, min_track_frac) for name in names}
    max_dogs = max((len(t) for t in per_video.values()), default=0)
    candidates = {name: sorted(t, key=t.get, reverse=True) for name, t in per_video.items()}

    # 상비 관찰(2026-07-03 장면추론 설계) — 어휘 없는 닫힌 축 센서(오디오·장면분류·
    # 휘도) + 자유 캡션 = 영상별 관찰 프로필. 폐기된 '고정 어휘 선태깅'과 다르다:
    # 저건 상황의 열거(구조적 사각), 이건 센서의 열거(축 전체 커버, M3 전례).
    # 어휘가 필요한 매칭은 여전히 render(요청을 아는 시점)에서 요청 주도로 한다.
    from .m4_action.observe import ensure_profiles
    ensure_profiles(ws, names)

    # 사람이 이미 확정한 강아지(foster_auto/foster_track)은 재판정으로 뒤집지 않는다 —
    # 안 그러면 다견 확정 후 재실행할 때마다 needs_foster_pick 으로 되돌아가 무한 대기.
    meta0 = ws.read_meta()
    if meta0.get("foster_auto") or meta0.get("foster_track") is not None:
        meta = ws.update_meta(state="prepared", dog_candidates=candidates)
        print("[prepare] 강아지 기확정 — 재판정 생략 ✓")
    elif max_dogs <= 1:
        meta = ws.update_meta(state="prepared", foster_auto=True, foster_track=None,
                              dog_candidates=candidates)
        print(f"[prepare] 단독견 자동확정 ✓  (영상별 트랙수 {{ {', '.join(f'{k}:{len(v)}' for k,v in per_video.items())} }})")
    else:
        # 다견 — 2패스 앵커(2026-07-03 확정): ①사진 앵커로 확신 영상 자동 확정
        # ②그래도 애매하면 확정 영상 크롭 top-K(앵커 전파)를 레퍼런스에 더해 재시도.
        # 카드는 끝까지 애매한 영상만(사용자 선택은 확신 없는 것 중에서).
        from .m2_reid.photo_anchor import (DONOR_MIN_MARGIN, confident,
                                           donor_reference, load_ref_embedding,
                                           match_video, ref_photos)
        fmap = dict(meta0.get("foster_track_map") or {})
        multi = [n for n in names if len(per_video.get(n, {})) >= 2]
        singles = [n for n in names if len(per_video.get(n, {})) == 1]
        pending = [n for n in multi if fmap.get(n) is None]
        photos = ref_photos(ws)
        margins_now: dict[str, float] = {}

        def _anchor_pass(refs, tag):
            for n in list(pending):
                res = match_video(ws, n, list(per_video[n]), refs, embedder)
                if confident(res):
                    frag = res["tracks"]
                    fmap[n] = frag if len(frag) > 1 else res["track"]
                    margins_now[n] = res["margin"]
                    pending.remove(n)
                    label = f"track {frag}" if len(frag) > 1 else f"track {res['track']}"
                    print(f"     {n}: {label} 자동확정 "
                          f"(sim {res['sim']:.2f}, 격차 {res['margin']:.2f})")
                else:
                    sims = {t: round(s, 2) for t, s in res["sims"].items()}
                    print(f"     {n}: 애매{tag} (sims {sims})")

        if pending and (photos or singles or fmap):
            from .m2_reid.embed import DinoEmbedder
            embedder = DinoEmbedder()
            refs = []
            if photos:
                refs.append(load_ref_embedding(photos, embedder))
                print(f"[사진앵커] 레퍼런스 {len(photos)}장 ↔ 다견 {len(pending)}영상 매칭…")
                _anchor_pass(refs, "")
            if pending:
                # 기증자: 단독견(구조적 안전, margin 1.0 취급) + 이번에 고격차 확정된 다견
                donor_specs = [(n, None, 1.0) for n in singles]
                donor_specs += [(n, fmap[n], m) for n, m in margins_now.items()
                                if m >= DONOR_MIN_MARGIN]
                if donor_specs:
                    vref, picked = donor_reference(ws, donor_specs, embedder)
                    if vref is not None:
                        cuts = ", ".join(f"{c['video']}#{c['frame']}" for c in picked)
                        print(f"[앵커전파] 기증 컷 {len(picked)}개({cuts}) ↔ "
                              f"애매 {len(pending)}영상 재시도…")
                        _anchor_pass(refs + [vref], "(전파에도)")
        if pending:
            meta = ws.update_meta(state="needs_foster_pick", foster_auto=False,
                                  foster_track_map=fmap, foster_uncertain=pending,
                                  dog_candidates=candidates)
            print(f"[prepare] 다견 — 애매한 {len(pending)}영상만 강아지 선택 필요: "
                  f"{pending} (meta.foster_track_map 에 track 지정 후 재실행)")
        else:
            meta = ws.update_meta(state="prepared", foster_auto=False,
                                  foster_track_map=fmap, foster_uncertain=[],
                                  dog_candidates=candidates)
            print(f"[prepare] 다견 전원 확정 ✓ (사진앵커/기존맵): {fmap}")
    return meta


def _infer_scenes(ws: Workspace, names: list[str], plan) -> None:
    """요청 주도 장면 추론 — 에스컬레이션 사다리(2026-07-03 장면추론 설계).

    ① 관찰 프로필 텍스트 매칭: prepare 가 쌓은 프로필(오디오·장면분류·휘도·캡션)
       ↔ 요청 키워드를 Gemma *텍스트* 판정(vision 불필요, 싸다). 오디오가 카페·비
       같은 스틸 사각을 뚫는 자리(스파이크 실측).
    ② 프로필로 못 잡은 키워드만 기존 표적 vision 다지선다(scene_auto)로 승급.
    보기 = Gemma interpret 가 이 요청에서 뽑은 키워드들(고정 어휘 없음). 결과는
    meta.scene_tags_auto 에 누적(키워드별 캐시 — 재렌더 시 새 키워드만 질문).
    사람 태그(meta.scene_tags)가 있는 영상은 건드리지 않는다(사람 우선).
    """
    kws = [k for b in plan.blocks for k in (b.keywords or [])]
    kws = [k for k in dict.fromkeys(kws) if k]
    meta = ws.read_meta()
    human = meta.get("scene_tags") or {}
    targets = [n for n in names if not human.get(n)]
    asked = set(meta.get("scene_keywords_asked") or [])
    new_kws = [k for k in kws if k not in asked]
    if not (new_kws and targets):
        return
    auto = meta.get("scene_tags_auto") or {}

    def _merge(kw_to_names: dict[str, list[str]]) -> None:
        for kw, hit in kw_to_names.items():
            for n in hit:
                auto[n] = sorted(set(auto.get(n, [])) | {kw})

    from .m4_action.observe import (ensure_profiles, match_keywords,
                                    motion_summary, propagate_tags)
    profiles = ensure_profiles(ws, targets)   # 구잡(프로필 없는 prepare) 지연 빌드
    motions = {n: m for n in targets if (m := motion_summary(ws, n))}
    print(f"[장면추론] 프로필 매칭 — 키워드 {new_kws} ↔ 영상 {len(profiles)}개…")
    matched = match_keywords(profiles, new_kws, extras=motions)
    for kw, hit in matched.items():
        print(f"     '{kw}' → {hit or '(없음)'}")
    # 장면 전파: 확정 영상과 "같은 장소"(배경 AND 소리풍경)인 미매칭 영상에 확장
    for kw, exts in propagate_tags(profiles, matched).items():
        for u, d, vis, aud in exts:
            matched[kw].append(u)
            print(f"     '{kw}' 전파: {u} (시각 {vis:.2f}·오디오 {aud:.2f} ← {d})")
    _merge(matched)

    pending = [kw for kw in new_kws if not matched.get(kw)]
    if pending:
        from .m4_action.scene_auto import infer_scene_tags
        print(f"[장면추론] 표적 vision 승급 — 미해결 키워드 {pending}…")
        inferred = infer_scene_tags(ws, targets, pending)
        _merge({kw: [n for n, tags in inferred.items() if kw in tags]
                for kw in pending})
        for n, tags in inferred.items():
            if tags:
                print(f"     {n}: {tags}")
    ws.update_meta(scene_tags_auto=auto,
                   scene_keywords_asked=sorted(asked | set(new_kws)))


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
        raise SystemExit(f"{ws.root}: 강아지 선택 대기 중. meta.foster_track 설정 후 재시도.")

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

    def _maybe_author(plan, want_narration: bool):
        """저작 3분기 — 구조 소유권은 이진, 빈칸 채움이 그라디언트(2026-07-03 설계).

        ① 구조 없음(기본값 블록 1개/장님 작문) → 전체 저작(구성 작가)
        ② 구조는 유저 것, 필드 일부 빈칸(자막·초·소재) → 부분 저작 + 결정론 병합
           (유저 명시 필드는 저작 출력에서 읽지 않아 구조적으로 불변)
        ③ 풀 스펙 → 저작 0호출
        소재 인지 창작이라 실행마다 다를 수 있음(의도된 비결정 — --rerender = 복권).
        실패 시 번역 플랜 유지(안전한 저하)."""
        from .m6_edit.author import (author_plan, fill_plan, is_unstructured,
                                     script_invented)
        if is_unstructured(plan) or script_invented(plan, request):
            print("[저작] 요청에 구성 없음 → 관찰 프로필로 구성·대본 창작…")
            authored = author_plan(request, ws, names, narration=want_narration)
            if authored is None:
                print("     저작 실패 — 번역 플랜으로 폴백")
                return plan
            for i, b in enumerate(authored.blocks):
                print(f"     블록{i}: {b.sources} select={b.select} dur={b.target_dur} "
                      f"zoom={b.zoom} caption={b.caption!r}"
                      + (f" narration={b.narration!r}" if b.narration else ""))
            return authored
        return fill_plan(request, ws, names, plan, narration=want_narration)

    if mode == "narration":
        print("[M5+M6] 대본 분해·합성·렌더…")
        plan = _maybe_author(interpret_narration(request), True)
        n_narr = sum(1 for b in plan.blocks if b.narration)
        print(f"     블록 {len(plan.blocks)}개 (내레이션 {n_narr}구절, 보이스 {voice})")
        _infer_scenes(ws, names, plan)
        render_narrated(plan, sources, str(out_path), size, fps, ws=ws, voice=voice)
    else:
        print("[M6] 편집 인텐트 해석…")
        plan = _maybe_author(interpret_plan(request), False)
        print(f"     title={plan.title!r}  블록 {len(plan.blocks)}개")
        _infer_scenes(ws, names, plan)
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
    p.add_argument("--dog-photos", nargs="+",
                   help="강아지 레퍼런스 사진(선택) — 다견 감지 시 사진 앵커로 자동 지정")
    p.add_argument("--request", help="자연어 편집요청(주면 render 까지)")
    p.add_argument("--rerender", action="store_true",
                   help="저장된 요청(meta.request)으로 재렌더 — 분석·태그·장면·TTS 캐시를 "
                        "전부 재사용하는 빠른 재뽑기(예: TTS 복권 다시 긁기)")
    p.add_argument("--prepare-only", action="store_true", help="prepare 까지만")
    p.add_argument("--data-root", default=None, help="잡 저장 루트(기본 $DATA_ROOT 또는 ./jobs)")
    p.add_argument("--size", default="1080x1920")
    p.add_argument("--weights", default="yolo11m.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--voice", default=None,
                   help="TTS 보이스(모드 A, 잡 단위 통일). 기본 meta.voice 또는 eric")
    args = p.parse_args(argv)

    if args.inputs:
        ws = init_job(args.job_id, args.inputs, args.data_root, dog_photos=args.dog_photos)
    else:
        ws = Workspace.job(args.job_id, args.data_root)
        if not ws.meta_path.exists():
            raise SystemExit(f"{ws.root}: 기존 잡 없음. --inputs 로 생성하세요.")

    if args.rerender and not args.request:      # 명시 --request 가 있으면 그쪽 우선
        args.request = ws.read_meta().get("request")
        if not args.request:
            raise SystemExit(f"{ws.root}: 저장된 요청 없음 — 먼저 --request 로 렌더하세요.")
        print(f"[재렌더] 저장된 요청 재사용: {args.request[:60]}…")

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
