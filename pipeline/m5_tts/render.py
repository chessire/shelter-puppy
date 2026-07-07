"""모드 A 렌더 — 내레이션이 타임라인 주인, AV 싱크는 블록별 조립으로 결정론 보장.

구조: 요청 → interpret(모드 판정·대본 분해) → 구절 합성(M5) → 블록 렌더(M6 재사용)
     → 블록별 오디오 조립 → 먹스 → AI 표시 배지("AI 편집 · AI 음성").

싱크 원리(구절별 합성의 확장): 구절 i 는 정확히 블록 i 시작에서 발화한다 —
  - 블록 목표 길이 = max(사용자 명시 초, 구절 길이 + 기본 쉼). "길이는 음성이 이긴다":
    렌더된 영상이 구절보다 짧으면 setpts 로 늘린다(앵커는 시작점만, 음성은 안 자른다).
  - 오디오 트랙 = Σ(구절 wav + 무음(실측 블록길이 − 구절길이)). 경계가 실측 블록 길이로
    재계산되므로 오차가 누적되지 않는다(측정 아닌 산수). 블록 프레임 반올림(~1/30s)만 남음.
무음 블록(narration="")은 영상만 — 그 길이만큼 무음이 깔린다.

모델 캐시: TTS_HF_HOME 환경변수로 지정(예: ~/tts-spike/hf-cache 재사용).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import DEFAULT_PAUSE, DEFAULT_VOICE, VOICES
from .assemble import _read_wav, _write_wav, synth_phrases
from .engine import QwenSubprocessEngine
from .interpret import decide_mode, interpret_narration
from ..m6_edit import EditPlan, layout
from ..m6_edit.badge import apply_badge
from ..m6_edit.run import (_apply_speed, _burn_plan_texts, _overlay_text,
                           _probe_dur, _stitch, allowed_sources_per_block,
                           render_block)
from ..workspace import Workspace

_MIN_TAIL = 0.15   # 구절 끝과 컷 사이 최소 숨(초) — 이보다 짧으면 영상을 늘린다
_TTS_SR = 24000


def _mux(video: Path, wav: Path, out: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)],
                   check=True, capture_output=True)


def render_narrated(plan: EditPlan, sources, out_path: str, size=(1080, 1920),
                    fps=30.0, ws: Workspace | None = None,
                    voice: str = DEFAULT_VOICE) -> dict:
    """모드 A 렌더 본체. 반환: 타임라인/게이트 리포트(사이드카로도 저장)."""
    ws = ws or Workspace.dev()
    if voice not in VOICES:      # 워커까지 안 내려가고 입구에서 친절하게
        raise SystemExit(f"지원하지 않는 보이스: {voice!r} — 후보: {', '.join(VOICES)}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1) 구절 합성 (블록 순서 = 구절 순서). 캐시는 출력 폴더 곁 tts/ — 재렌더 시 재사용.
    narr_idx = [i for i, b in enumerate(plan.blocks) if b.narration.strip()]
    if not narr_idx:
        raise SystemExit("내레이션 구절이 없음 — 모드 B(m6_edit.run)를 쓰세요.")
    phrases = [{"id": f"b{i}", "text": plan.blocks[i].narration} for i in narr_idx]
    with QwenSubprocessEngine() as engine:
        rows = synth_phrases(phrases, out.parent / "tts", engine, voice=voice)
    row_by_block = dict(zip(narr_idx, rows))

    def _phrase_sec(row: dict) -> float:
        data, sr = _read_wav(Path(row["wav"]))
        return len(data) / sr

    allowed = allowed_sources_per_block(plan, sources, ws)   # 핀 소스 예약(선점 방지)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        used: set = set()
        # (원 블록 idx, block, 영상, 실측길이, clips, 확정 자막 pos, 구절row)
        seq: list[tuple] = []
        for idx, block in enumerate(plan.blocks):
            row = row_by_block.get(idx)
            if row is not None:
                need = _phrase_sec(row) + DEFAULT_PAUSE
                block.target_dur = max(block.target_dur or 0.0, need)
            vid, clips, cap_pos = render_block(block, allowed[idx], tmp, idx, size,
                                               fps, exclude=used, ws=ws)
            if vid is None:                    # 조건 맞는 클립 없음 → 블록·구절 함께 드롭
                print(f"  [경고] 블록{idx} 클립 없음 → 구절과 함께 제외: "
                      f"{block.narration[:30]!r}")
                continue
            used.update(clips)
            vd = _probe_dur(str(vid))
            if row is not None:
                psec = _phrase_sec(row)
                if vd < psec + _MIN_TAIL:      # 길이는 음성이 이긴다 — 영상을 늘린다
                    target = psec + DEFAULT_PAUSE
                    stretched = tmp / f"b{idx}_stretch.mp4"
                    _apply_speed(vid, vd / target, stretched)
                    vid, vd = stretched, _probe_dur(str(stretched))
            seq.append((idx, block, vid, vd, clips, cap_pos, row))
        if not seq:
            raise SystemExit("선택된 클립이 없음 (어떤 블록도 조건에 맞는 구간 없음).")

        # 2) 영상 concat + 전역 title + 블록 걸침 카피(모드 B 와 같은 결정론 경로)
        stitched = tmp / "all.mp4"
        _stitch([v for _, _, v, _, _, _, _ in seq],
                [d for _, _, _, d, _, _, _ in seq], "cut", stitched)
        titled = tmp / "titled.mp4"
        total = sum(d for _, _, _, d, _, _, _ in seq)
        _overlay_text(stitched, plan.title, titled, size, pos="top",
                      window=(layout.title_window(plan.title, total)
                              if plan.title else None), style="title")
        if plan.texts:
            texted = tmp / "texted.mp4"
            _burn_plan_texts(titled, plan,
                             [(oi, b, c, d, cp) for oi, b, _v, d, c, cp, _r in seq],
                             texted, size, ws)
            titled = texted

        # 3) 오디오 트랙 — 실측 블록 길이 기준 산수 (경계 누적오차 없음)
        chunks, timeline, t = [], [], 0.0
        for _oi, _b, vid, vd, _c, _cp, row in seq:
            if row is None:
                chunks.append(np.zeros(int(vd * _TTS_SR), dtype=np.int16))
            else:
                data, sr = _read_wav(Path(row["wav"]))
                assert sr == _TTS_SR, f"TTS 샘플레이트 예상 밖: {sr}"
                chunks.append(data)
                tail = max(0.0, vd - len(data) / sr)
                chunks.append(np.zeros(int(tail * sr), dtype=np.int16))
                timeline.append({"text": row["text"], "start": round(t, 3),
                                 "end": round(t + len(data) / sr, 3),
                                 "asr_flag": row["asr_flag"]})
            t += vd
        narr_wav = tmp / "narration.wav"
        _write_wav(narr_wav, np.concatenate(chunks), _TTS_SR)

        # 4) 먹스 + AI 표시 배지(TTS 변형) + 메타데이터
        muxed = tmp / "muxed.mp4"
        _mux(titled, narr_wav, muxed)
        apply_badge(muxed, out, tts=True, size=size)

    report = {"mode": "narration", "voice": voice, "total_sec": round(t, 3),
              "timeline": timeline,
              "asr_flags": [r["id"] for r in rows if r["asr_flag"]]}
    out.with_suffix(".narration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1))
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="m5_render", description="모드 라우터 + 모드 A 렌더")
    p.add_argument("request", help="자연어 요청 (대본 포함 가능)")
    p.add_argument("--sources", nargs="+", required=True, help="소스 이름들")
    p.add_argument("--out", default="data/dev/demo/m5_modeA.mp4")
    p.add_argument("--size", default="1080x1920")
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument("--job", help="잡 워크스페이스 id (미지정 시 dev)")
    args = p.parse_args(argv)

    w, h = (int(x) for x in args.size.split("x"))
    ws = Workspace.job(args.job) if args.job else Workspace.dev()
    sources = [(str(ws.analysis(n)), str(ws.preds_m4(n))) for n in args.sources]

    mode, conf = decide_mode(args.request)
    print(f"[모드] {mode} (확신 {conf:.2f})")
    if mode == "uncertain":
        print("  → 자막인지 음성인지 애매 — 고객 카드 1탭 대상. (핀 키워드를 넣으면 확정)")
        return 2
    if mode == "edit":                       # 모드 B 위임 + 배지
        from ..m6_edit.run import interpret_plan, render_plan
        plan = interpret_plan(args.request)
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw.mp4"
            render_plan(plan, sources, str(raw), (w, h), ws=ws)
            apply_badge(raw, Path(args.out), tts=False, size=(w, h))
        print(f"[렌더·모드B] → {args.out}")
        return 0

    plan = interpret_narration(args.request)
    print(f"[plan] title={plan.title!r}  블록 {len(plan.blocks)}개")
    for i, b in enumerate(plan.blocks):
        print(f"  블록{i}: select={b.select} dur={b.target_dur} zoom={b.zoom} "
              f"subj={b.subject} keywords={b.keywords} caption={b.caption!r}\n"
              f"         narration={b.narration!r}")
    rep = render_narrated(plan, sources, args.out, (w, h), ws=ws, voice=args.voice)
    print(f"[렌더·모드A] → {args.out}  ({rep['total_sec']:.1f}s, "
          f"구절 {len(rep['timeline'])}개, asr_flag {len(rep['asr_flags'])}건)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
