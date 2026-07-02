"""Qwen3-TTS 워커 — 전용 venv(venv-tts) 안에서 도는 장기 실행 서브프로세스.

프로젝트 venv 를 mlx-audio 로 오염시키지 않기 위한 격리 경계(ollama 패턴).
stdin 으로 JSON 한 줄씩 받고 stdout 으로 JSON 한 줄씩 답한다. 모델은 첫 요청에서
1회 로드 후 상주(구절마다 재로드하면 RTF 가 죽는다).

프로토콜:
  {"op":"synth", "text":…, "voice":…, "instruct":…, "out":wav경로} → {"ok":true, "sec":…, "sr":…}
  {"op":"transcribe", "path":wav경로}                              → {"ok":true, "text":…}
  {"op":"ping"}                                                    → {"ok":true}

주의: stdout 은 JSONL 전용 — 라이브러리가 찍는 진행바·경고는 전부 stderr 로 우회.
이 파일은 프로젝트 venv 가 아니라 venv-tts 인터프리터로 실행된다(의존성 import 금지 경계).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys

MODEL_ID = os.environ.get("TTS_MODEL_ID", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
WHISPER_ID = os.environ.get("TTS_WHISPER_ID", "mlx-community/whisper-large-v3-turbo")

_tts_model = None


def _synth(req: dict) -> dict:
    global _tts_model
    import numpy as np
    import soundfile as sf
    from mlx_audio.tts.utils import load_model

    if _tts_model is None:
        with contextlib.redirect_stdout(sys.stderr):
            _tts_model = load_model(MODEL_ID)

    kw = {}
    if req.get("instruct"):
        kw["instruct"] = req["instruct"]
    import time
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        segs = list(_tts_model.generate(
            text=req["text"], voice=req["voice"], lang_code="ko", **kw))
    audio = np.concatenate([np.array(s.audio) for s in segs])
    sr = getattr(segs[0], "sample_rate", 24000)
    # PCM_16 고정 — 조립기(프로젝트 venv)가 stdlib wave 로 읽는다(soundfile 의존 회피)
    sf.write(req["out"], audio, sr, subtype="PCM_16")
    return {"ok": True, "sec": round(len(audio) / sr, 3), "sr": sr,
            "synth_sec": round(time.perf_counter() - t0, 3)}


def _transcribe(req: dict) -> dict:
    import mlx_whisper
    with contextlib.redirect_stdout(sys.stderr):
        r = mlx_whisper.transcribe(req["path"], path_or_hf_repo=WHISPER_ID, language="ko")
    return {"ok": True, "text": r["text"].strip()}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "synth":
                resp = _synth(req)
            elif op == "transcribe":
                resp = _transcribe(req)
            elif op == "ping":
                resp = {"ok": True}
            else:
                resp = {"ok": False, "error": f"unknown op: {op}"}
        except Exception as e:  # 워커는 죽지 않고 에러를 응답으로 — 엔진이 재시도 판단
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
