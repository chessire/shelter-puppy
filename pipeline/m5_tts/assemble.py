"""구절별 합성 → 무음 결정론 조립 — 내레이션 트랙과 타임라인을 만든다.

핵심: 구절 경계는 측정하지 않는다. 구절 wav 길이 + 우리가 정한 무음이 곧 타임라인이라
forced alignment 없이 경계가 정확하다. 구절 사이 무음은 컷이 들어올 자리(쉼표=편집점).

ASR 왕복 게이트: 합성 직후 Whisper 로 되읽혀 CER 체크 — LLM-TTS 저빈도 오발음
(스파이크 1R "토리→프린") 방어. 표기 관행("세 살"↔"3살") 착시가 있어 원문/정규화문 중
min CER 로 재고, 실패 시 1회 재합성(샘플링이라 재시도로 달라짐), 그래도 실패면
asr_flag 로 표시만(uncertain 카드 후보) — 억지로 확정하지 않는다.

캐시: hash(모델·보이스·instruct·정규화문) → wav. 재렌더(자막 수정·블록 순서 변경) 시
변한 구절만 재합성.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import wave
from pathlib import Path

import numpy as np

from . import ASR_CER_GATE, DEFAULT_PAUSE, DEFAULT_VOICE, MODEL_ID
from .engine import QwenSubprocessEngine, TTSEngine
from .normalize_ko import normalize


def _norm_txt(t: str) -> str:
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", t)


def cer(ref: str, hyp: str) -> float:
    """문자 오류율(Levenshtein/len). 한글·영숫자만 비교(문장부호·공백 무시)."""
    r, h = _norm_txt(ref), _norm_txt(hyp)
    if not r:
        return 0.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2, "워커는 PCM_16 로 쓴다"
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return data, w.getframerate()


def _write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.astype(np.int16).tobytes())


def _cache_key(text_norm: str, voice: str, instruct: str) -> str:
    return hashlib.sha1(f"{MODEL_ID}|{voice}|{instruct}|{text_norm}".encode()).hexdigest()[:16]


def synth_phrases(phrases: list[dict], out_dir: Path, engine: TTSEngine,
                  voice: str = DEFAULT_VOICE, asr_gate: bool = True) -> list[dict]:
    """구절 리스트를 개별 합성(캐시·게이트). phrase: {id?, text, instruct?, pause_after?}"""
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ph in enumerate(phrases):
        pid = ph.get("id", f"p{i:02d}")
        raw = ph["text"]
        norm = normalize(raw)
        instruct = ph.get("instruct", "")
        wav = cache / f"{_cache_key(norm, voice, instruct)}.wav"
        row = {"id": pid, "text": raw, "text_norm": norm,
               "pause_after": float(ph.get("pause_after", DEFAULT_PAUSE)),
               "wav": str(wav), "cached": wav.exists(), "asr_flag": False}
        if not wav.exists():
            engine.synth(norm, str(wav), voice=voice, instruct=instruct)
            if asr_gate:
                hyp = engine.transcribe(str(wav))
                c = min(cer(raw, hyp), cer(norm, hyp))
                if c > ASR_CER_GATE:  # 파국 오발음 → 1회 재합성
                    engine.synth(norm, str(wav), voice=voice, instruct=instruct)
                    hyp = engine.transcribe(str(wav))
                    c = min(cer(raw, hyp), cer(norm, hyp))
                    row["asr_flag"] = c > ASR_CER_GATE  # 그래도 실패 → 카드 후보
                row["asr_cer"] = round(c, 4)
                row["asr_hyp"] = hyp
        rows.append(row)
    return rows


def assemble(rows: list[dict], out_wav: Path, lead_in: float = 0.2) -> dict:
    """구절 wav + 무음을 이어붙여 단일 트랙 + 타임라인. 경계는 조립이 정의한다."""
    chunks, timeline = [], []
    sr = None
    t = lead_in
    for row in rows:
        data, r = _read_wav(Path(row["wav"]))
        if sr is None:
            sr = r
            chunks.append(np.zeros(int(lead_in * sr), dtype=np.int16))
        assert r == sr, "구절 간 샘플레이트 불일치"
        dur = len(data) / sr
        timeline.append({"id": row["id"], "text": row["text"], "start": round(t, 3),
                         "end": round(t + dur, 3), "asr_flag": row["asr_flag"]})
        chunks.append(data)
        chunks.append(np.zeros(int(row["pause_after"] * sr), dtype=np.int16))
        t += dur + row["pause_after"]
    out = np.concatenate(chunks)
    _write_wav(out_wav, out, sr)
    return {"wav": str(out_wav), "sr": sr, "total_sec": round(len(out) / sr, 3),
            "timeline": timeline}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="구절 JSON → 내레이션 wav + 타임라인")
    ap.add_argument("--phrases", required=True, help='JSON 파일 [{"text":…, "pause_after":…}]')
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--no-asr-gate", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phrases = json.loads(Path(args.phrases).read_text())

    with QwenSubprocessEngine() as engine:
        rows = synth_phrases(phrases, out_dir, engine, voice=args.voice,
                             asr_gate=not args.no_asr_gate)
    result = assemble(rows, out_dir / "narration.wav")
    (out_dir / "narration.json").write_text(
        json.dumps({"voice": args.voice, "phrases": rows, **result},
                   ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
