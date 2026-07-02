"""다영상 집계 리포트 — 측정 세트 전체의 M1 분포를 한 표로.

per-video 지표 + micro 집계(전체 GT 합산 기준)를 낸다. IDF1 은 영상마다
정체성 공간이 달라 단순 합산이 안 되므로 macro 평균(영상별 평균)으로 본다.
"""

from __future__ import annotations

from pathlib import Path

from . import io
from .metrics.m1_tracking import evaluate_m1


def aggregate_m1(gt_dir: str | Path, pred_dir: str | Path, iou_thr: float = 0.5) -> dict:
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    rows = []
    for gt_path in sorted(gt_dir.glob("*_m1.jsonl")):
        name = gt_path.name.replace("_m1.jsonl", "")
        pred_path = pred_dir / f"{name}_m1.jsonl"
        if not pred_path.exists():
            continue
        m = evaluate_m1(io.read_frames(gt_path), io.read_frames(pred_path), iou_thr)
        rows.append((name, m))

    agg = {"num_gt": 0, "fp": 0, "fn": 0, "idsw": 0, "frag": 0}
    for _, m in rows:
        agg["num_gt"] += m.num_gt
        agg["fp"] += m.fp
        agg["fn"] += m.fn
        agg["idsw"] += m.idsw
        agg["frag"] += m.fragmentation
    micro_mota = (
        1 - (agg["fn"] + agg["fp"] + agg["idsw"]) / agg["num_gt"]
        if agg["num_gt"] else 1.0
    )
    micro_miss = agg["fn"] / agg["num_gt"] if agg["num_gt"] else 0.0
    macro_idf1 = sum(m.idf1 for _, m in rows) / len(rows) if rows else 1.0
    return {"rows": rows, "agg": agg, "micro_mota": micro_mota,
            "micro_miss": micro_miss, "macro_idf1": macro_idf1}


def render_m1_report(gt_dir: str | Path, pred_dir: str | Path) -> str:
    r = aggregate_m1(gt_dir, pred_dir)
    L = []
    L.append("M1 측정 세트 종합 리포트")
    L.append(f"  {'video':14}{'num_gt':>7}{'miss':>7}{'fp':>5}{'idsw':>6}"
             f"{'frag':>6}{'MOTA':>7}{'IDF1':>7}")
    L.append("  " + "-" * 60)
    for name, m in r["rows"]:
        L.append(f"  {name:14}{m.num_gt:>7}{m.miss_rate:>7.2f}{m.fp:>5}"
                 f"{m.idsw:>6}{m.fragmentation:>6}{m.mota:>7.2f}{m.idf1:>7.2f}")
    L.append("  " + "-" * 60)
    a = r["agg"]
    L.append(f"  {'MICRO(전체)':14}{a['num_gt']:>7}{r['micro_miss']:>7.2f}"
             f"{a['fp']:>5}{a['idsw']:>6}{a['frag']:>6}{r['micro_mota']:>7.2f}"
             f"{r['macro_idf1']:>7.2f}*")
    L.append("  * IDF1 은 macro 평균(영상별 정체성 공간이 달라 합산 불가)")
    return "\n".join(L)
