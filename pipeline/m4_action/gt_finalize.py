"""M4 GT 확정 — 사람이 action 을 채운 draft → 최종 GT(ActionSegment).

draft(`data/gt/{name}_m4.draft.json`)의 각 구간 action 을 검증(허용 라벨셋)하고,
group 을 action 에서 파생(group_of)해 `data/gt/{name}_m4.json` 으로 동결한다.
GT 는 정답이므로 conf=1.0, uncertain=False.

action 이 아직 "TODO" 인 구간이 있으면 거부한다(미완성 GT 방지).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import ACTIONS, group_of
from ..harness import io
from ..harness.schemas import ActionSegment


def finalize(name: str) -> int:
    draft_path = Path(f"data/dev/gt/{name}_m4.draft.json")
    if not draft_path.exists():
        print(f"  ! {name}: draft 없음 ({draft_path}) — 먼저 gt_scaffold 실행")
        return 1
    draft = io.read_json(draft_path)
    segs: list[ActionSegment] = []
    errors: list[str] = []
    for i, s in enumerate(draft.get("segments", [])):
        action = str(s.get("action", "")).strip()
        if action in ("", "TODO"):
            errors.append(f"#{i} ({s['start_t']}-{s['end_t']}s): action 미입력")
            continue
        if action not in ACTIONS:
            errors.append(f"#{i}: 허용 안 된 action {action!r} (허용: {ACTIONS})")
            continue
        segs.append(ActionSegment(
            start_t=float(s["start_t"]), end_t=float(s["end_t"]),
            group=group_of(action), action=action, conf=1.0, uncertain=False,
        ))
    if errors:
        print(f"  ! {name}: 확정 불가 ({len(errors)}건)")
        for e in errors:
            print(f"      {e}")
        return 1
    out = Path(f"data/dev/gt/{name}_m4.json")
    io.write_action_segments(out, segs)
    dyn = sum(s.dur for s in segs if s.group == "dynamic")
    sta = sum(s.dur for s in segs if s.group == "static")
    print(f"[OK] {name}: {len(segs)}구간 확정 → {out.name}  (dynamic {dyn:.1f}s / static {sta:.1f}s)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="m4_gt_finalize")
    p.add_argument("names", nargs="*",
                   help="영상 이름들 (기본: draft 있는 전부)")
    args = p.parse_args(argv)
    names = args.names or [
        p.name.replace("_m4.draft.json", "")
        for p in sorted(Path("data/dev/gt").glob("*_m4.draft.json"))
    ]
    rc = 0
    for n in names:
        rc |= finalize(n)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
