"""하네스 CLI — `eval --stage STAGE --gt GT --pred PRED`.

각 단계의 정답(GT)과 자동출력(pred)을 받아 메트릭을 계산하고, thresholds.yaml
임계값과 대조해 표로 출력한다. 임계 미설정이면 숫자만 보고하고 판정 보류.

사용 예:
  python -m pipeline.harness.cli eval --stage m1 --gt gt.jsonl --pred pred.jsonl
  python -m pipeline.harness.cli eval --stage m3 --gt gt.json  --pred pred.json --json
"""

from __future__ import annotations

import argparse
import json
import sys

from . import io
from .metrics.m1_tracking import evaluate_m1
from .metrics.m2_reid import evaluate_m2
from .metrics.m3_motion import evaluate_m3
from .metrics.m4_action import evaluate_m4
from .metrics.m5_match import evaluate_m5
from .thresholds import check_stage, load_thresholds, overall


def _eval_stage(stage: str, gt_path: str, pred_path: str, iou_thr: float) -> dict:
    if stage == "m1":
        m = evaluate_m1(io.read_frames(gt_path), io.read_frames(pred_path), iou_thr)
    elif stage == "m2":
        gt_map = {int(k): int(v) for k, v in io.read_json(gt_path)["mapping"].items()}
        m = evaluate_m2(gt_map, io.read_reid(pred_path))
    elif stage == "m3":
        m = evaluate_m3(io.read_segments(gt_path), io.read_segments(pred_path))
    elif stage == "m4":
        m = evaluate_m4(
            io.read_action_segments(gt_path), io.read_action_segments(pred_path)
        )
    elif stage == "m5":
        m = evaluate_m5(io.read_matches(gt_path), io.read_matches(pred_path), iou_thr)
    else:
        raise ValueError(f"알 수 없는 stage: {stage}")
    return m.as_dict()


def _render_table(stage: str, metrics: dict, thresholds: dict) -> str:
    checks = check_stage(stage, metrics, thresholds)
    checked = {c.metric for c in checks}
    lines = []
    lines.append(f"[{stage}] 평가 결과")
    lines.append(f"  {'metric':<20}{'value':>12}  {'rule':<10}{'status'}")
    lines.append("  " + "-" * 52)
    for c in checks:
        bound = "" if c.bound is None else f"{c.rule} {c.bound}"
        lines.append(f"  {c.metric:<20}{c.value:>12.4f}  {bound:<10}{c.status}")
    # 임계 대상이 아닌(숫자 아님) 메트릭도 참고로 출력.
    for k, v in metrics.items():
        if k not in checked:
            shown = "None" if v is None else v
            lines.append(f"  {k:<20}{str(shown):>12}  {'(참고)':<10}")
    lines.append("  " + "-" * 52)
    lines.append(f"  종합: {overall(checks)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="단일 단계 채점")
    pe.add_argument("--stage", required=True, choices=["m1", "m2", "m3", "m4", "m5"])
    pe.add_argument("--gt", required=True, help="정답 라벨 파일")
    pe.add_argument("--pred", required=True, help="자동출력 파일")
    pe.add_argument("--iou-thr", type=float, default=0.5)
    pe.add_argument("--thresholds", default=None, help="임계값 yaml(기본: 내장)")
    pe.add_argument("--json", action="store_true", help="JSON 으로 출력")

    pr = sub.add_parser("report", help="측정 세트 전체 집계 리포트")
    pr.add_argument("--stage", default="m1", choices=["m1"])
    pr.add_argument("--gt-dir", default="data/dev/gt")
    pr.add_argument("--pred-dir", default="data/dev/preds")

    args = parser.parse_args(argv)

    if args.cmd == "eval":
        metrics = _eval_stage(args.stage, args.gt, args.pred, args.iou_thr)
        thresholds = load_thresholds(args.thresholds)
        if args.json:
            checks = check_stage(args.stage, metrics, thresholds)
            print(json.dumps(
                {
                    "stage": args.stage,
                    "metrics": metrics,
                    "overall": overall(checks),
                },
                ensure_ascii=False,
                indent=2,
            ))
        else:
            print(_render_table(args.stage, metrics, thresholds))
        return 0

    if args.cmd == "report":
        from .report import render_m1_report
        print(render_m1_report(args.gt_dir, args.pred_dir))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
