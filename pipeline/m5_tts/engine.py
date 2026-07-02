"""TTSEngine — 엔진 교체 슬롯(얇은 어댑터) + Qwen 서브프로세스 구현.

M0 골든셋으로 re-ID 백본을 갈아끼우던 패턴의 TTS 판: 파이프라인은 이 인터페이스만
알고, 엔진 교체(예: 서버 이전 시 MLX→transformers, 또는 Supertonic 폴백)는 구현
클래스 하나로 끝난다. 골든이어셋(~/tts-spike/golden)으로 재채점하면 회귀 비교.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Protocol

from . import DEFAULT_VOICE


class TTSEngine(Protocol):
    def synth(self, text: str, out_path: str, voice: str = DEFAULT_VOICE,
              instruct: str = "") -> dict: ...
    def transcribe(self, wav_path: str) -> str: ...
    def close(self) -> None: ...


def _default_python() -> str:
    """워커 인터프리터 탐색: $TTS_PYTHON > 레포 venv-tts. 없으면 안내와 함께 실패."""
    env = os.environ.get("TTS_PYTHON")
    if env:
        return env
    repo = Path(__file__).resolve().parents[2] / "venv-tts" / "bin" / "python"
    if repo.exists():
        return str(repo)
    raise RuntimeError(
        "venv-tts 없음. 프로비저닝: uv venv venv-tts --python 3.12 && "
        "uv pip install --python venv-tts/bin/python mlx-audio soundfile mlx-whisper "
        "(또는 TTS_PYTHON 환경변수로 지정)")


class QwenSubprocessEngine:
    """venv-tts 의 qwen_worker 를 상주 프로세스로 띄우고 JSONL 로 대화한다."""

    def __init__(self, python: Optional[str] = None):
        worker = Path(__file__).resolve().parent / "qwen_worker.py"
        env = os.environ.copy()
        # 모델 캐시 위치는 호출부가 TTS_HF_HOME 으로 제어 (미지정 시 HF 기본 캐시)
        if env.get("TTS_HF_HOME"):
            env["HF_HOME"] = env["TTS_HF_HOME"]
        self._proc = subprocess.Popen(
            [python or _default_python(), "-u", str(worker)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, env=env)

    def _rpc(self, req: dict) -> dict:
        assert self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("TTS 워커가 응답 없이 종료됨 (stderr 확인)")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"TTS 워커 오류: {resp.get('error')}")
        return resp

    def synth(self, text: str, out_path: str, voice: str = DEFAULT_VOICE,
              instruct: str = "") -> dict:
        return self._rpc({"op": "synth", "text": text, "voice": voice,
                          "instruct": instruct, "out": out_path})

    def transcribe(self, wav_path: str) -> str:
        return self._rpc({"op": "transcribe", "path": wav_path})["text"]

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
