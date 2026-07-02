"""M4 — 동작 판별.

설계서 매핑: 임보견 크롭 → LLM 5지선다 + logprob → 동작군/정지군 판정 → 모션 대조.
이 패키지는 (1) GT 라벨링 스캐폴드, (2) 추론 러너, (3) 군 판정·폴백을 담는다.

5지선다 라벨셋과 군(group) 매핑은 여기서 동결한다 — GT·러너·메트릭이 같은
정의를 공유해야 채점이 성립한다.
"""

from __future__ import annotations

import json
from pathlib import Path

_FOSTER_MAP = Path("data/dev/gt/foster_map.json")


def foster_track(name: str) -> int:
    """영상의 임보견 GT track_id 를 명시적 config 에서 읽는다.

    track 0 자동가정 금지 — 임보견은 사람이 지정한 한 마리이고, GT 라벨과 M4 러너
    크롭이 *같은* 개를 써야 측정이 성립한다. foster_map.json 이 그 단일 소스.
    """
    m = json.loads(_FOSTER_MAP.read_text(encoding="utf-8"))
    if name not in m:
        raise SystemExit(f"{name}: foster_map.json 에 임보견 지정 없음. 먼저 등록 필요.")
    return int(m[name])

# 사용자 확정 라벨셋 (2026-06-30, 7동작으로 확장). 로코모션 5 + 묘기(재주) 2.
# 묘기 = 앉거나 엎드린 자세에서도 부리는 '제스처'(하이파이브·손·빵야·죽은척·구르기).
# 모션 크기로 동/정 가르는 5동작과 직교한 개념이라, 별도 라벨로 두고 TRICK 플래그로 표시.
ACTIONS: tuple[str, ...] = ("점프", "달리기", "걷기", "앉기", "엎드림", "동적묘기", "정적묘기")

# 동작군(dynamic) vs 정지군(static). logprob 을 개별 선택지가 아니라 군 합으로 읽는다.
# 묘기도 편집연출(빠른컷 vs 잔잔)을 위해 모션 크기 기준으로 두 군에 접는다.
DYNAMIC: frozenset[str] = frozenset({"점프", "달리기", "걷기", "동적묘기"})
STATIC: frozenset[str] = frozenset({"앉기", "엎드림", "정적묘기"})

# 묘기(재주) = 모션군과 직교한 '제스처' 플래그. M6 가 select="묘기" 로 모션군 무관하게 고른다.
TRICK: frozenset[str] = frozenset({"동적묘기", "정적묘기"})

# 저모션 동작 = 동적이지만 모션이 약해 표집/검출에서 놓치기 쉬운 것.
# 이 라벨셋에선 걷기가 해당(점프·달리기는 고모션). low_motion_recall 측정 기준.
LOW_MOTION: frozenset[str] = frozenset({"걷기"})


def group_of(action: str) -> str:
    """동작 라벨 → 군. 알 수 없는 라벨은 ValueError."""
    if action in DYNAMIC:
        return "dynamic"
    if action in STATIC:
        return "static"
    raise ValueError(f"알 수 없는 동작 라벨: {action!r} (허용: {ACTIONS})")


def is_trick(action: str | None) -> bool:
    """묘기(재주) 동작이면 True. None/일반동작은 False."""
    return action in TRICK
