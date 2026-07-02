"""스키마 객체의 디스크 입출력. JSONL(프레임열) / JSON(단건) 표준화.

GT(정답)와 pred(자동출력)는 동일 포맷을 공유한다 — 그래야 같은 로더로 읽어
같은 메트릭에 넣을 수 있다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import (
    ActionSegment,
    FrameDetections,
    MatchEntry,
    ReIDResult,
    Segment,
)


# --------------------------------------------------------------------------- #
# M1 — 프레임열 (JSONL)
# --------------------------------------------------------------------------- #
def write_frames(path: str | Path, frames: Iterable[FrameDetections]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr.to_dict(), ensure_ascii=False) + "\n")


def read_frames(path: str | Path) -> list[FrameDetections]:
    frames: list[FrameDetections] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(FrameDetections.from_dict(json.loads(line)))
    return frames


# --------------------------------------------------------------------------- #
# 단건 산출물 (JSON)
# --------------------------------------------------------------------------- #
def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_reid(path: str | Path) -> ReIDResult:
    return ReIDResult.from_dict(read_json(path))


def read_segments(path: str | Path) -> list[Segment]:
    raw = read_json(path)
    return [Segment.from_dict(s) for s in raw["segments"]]


def write_segments(path: str | Path, segments: list[Segment]) -> None:
    write_json(path, {"segments": [s.to_dict() for s in segments]})


def read_action_segments(path: str | Path) -> list[ActionSegment]:
    raw = read_json(path)
    return [ActionSegment.from_dict(s) for s in raw["segments"]]


def write_action_segments(path: str | Path, segments: list[ActionSegment]) -> None:
    write_json(path, {"segments": [s.to_dict() for s in segments]})


def read_matches(path: str | Path) -> list[MatchEntry]:
    raw = read_json(path)
    return [MatchEntry.from_dict(m) for m in raw["matches"]]


def write_matches(path: str | Path, matches: list[MatchEntry]) -> None:
    write_json(path, {"matches": [m.to_dict() for m in matches]})
