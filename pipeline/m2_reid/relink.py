"""재연결기 — 트랙 임베딩을 코사인 유사도 임계값으로 온라인 군집화.

설계서 흐름 시뮬레이션:
- 트랙을 등장 순서로 처리. 기존 앵커(군집 대표)와 최대 코사인 유사도 비교.
- 유사도 ≥ τ → 자동 재연결(그 군집에 합침, 대표 갱신 = 다각도 레퍼런스 성장).
- 미만 → 새 정체성 = 사람 1탭 앵커 생성(개입 1회).
  (경계는 LLM 타이브레이커가 들어갈 자리지만 v1 은 새 앵커로 처리)

τ 가 높을수록: 보수적 → 개입↑·false merge↓(precision↑)
τ 가 낮을수록: 공격적 자동병합 → 개입↓·false merge↑(precision↓)
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from ..harness import io
from ..harness.schemas import InterventionRecord, ReIDResult


def relink(
    track_ids: list[int], embeddings: np.ndarray, tau: float,
    track_frames: dict[int, set] | None = None,
) -> ReIDResult:
    """track_frames 가 주어지면 시간적 상호배제 적용: 같은 프레임에 동시 등장한
    두 트랙은 다른 개로 보고 절대 병합하지 않는다(다견 false merge 차단)."""
    clusters: list[dict] = []  # {rep, count, frames(set)}
    mapping: dict[int, int] = {}
    interventions: list[InterventionRecord] = []
    evidence: dict[int, dict] = {}

    order = sorted(range(len(track_ids)), key=lambda i: track_ids[i])  # 등장순 근사
    for i in order:
        tid = int(track_ids[i])
        e = embeddings[i]
        tf = track_frames.get(tid, set()) if track_frames else set()
        best_c, best_sim = -1, -1.0
        for ci, c in enumerate(clusters):
            if track_frames and (tf & c["frames"]):
                continue  # 시간 겹침 → 다른 개, 병합 금지
            sim = float(np.dot(c["rep"], e))  # 둘 다 정규화 → 코사인
            if sim > best_sim:
                best_sim, best_c = sim, ci
        if best_c >= 0 and best_sim >= tau:
            c = clusters[best_c]
            rep = c["rep"] * c["count"] + e
            c["rep"] = rep / (np.linalg.norm(rep) + 1e-8)
            c["count"] += 1
            c["frames"] |= tf
            mapping[tid] = best_c
            evidence[tid] = {"sim": round(best_sim, 3), "anchor": best_c, "auto": True}
        else:
            clusters.append({"rep": e.copy(), "count": 1, "frames": set(tf)})
            mapping[tid] = len(clusters) - 1
            interventions.append(InterventionRecord(tid, 0.0, "new_anchor"))
            evidence[tid] = {"sim": round(best_sim, 3), "anchor": -1, "auto": False}
    return ReIDResult(mapping=mapping, interventions=interventions, relink_evidence=evidence)


def _pair_counts(gt_map: dict[int, int], pred_map: dict[int, int]):
    tracks = sorted(set(gt_map) & set(pred_map))
    tp = fp = fn = 0
    for a, b in combinations(tracks, 2):
        sg = gt_map[a] == gt_map[b]
        sp = pred_map[a] == pred_map[b]
        if sg and sp: tp += 1
        elif sp and not sg: fp += 1
        elif sg and not sp: fn += 1
    return tp, fp, fn


def foster_taps(gt_map, track_ids, embeddings, track_frames, tau):
    """단독 임보견 시나리오 — 임보견 1탭 지정 후 그 조각이 자동으로 붙나.

    각 진짜 개를 '임보견'으로 두고: 앵커 1탭 → 이후 그 개의 다른 트랙은 코사인
    유사도≥τ & 시간 비충돌이면 자동, 아니면 카드(탭). 다른 개가 임보견으로 자동
    오인되면 false_attach(안전 위반).
    반환: [(taps, false_attach), ...]  (개별 임보견 시나리오마다)
    """
    import numpy as np
    emb = {int(t): embeddings[i] for i, t in enumerate(track_ids)}
    from collections import defaultdict
    dogs = defaultdict(list)
    for t, d in gt_map.items():
        if t in emb:
            dogs[d].append(t)

    out = []
    for d, tracks in dogs.items():
        tracks = sorted(tracks, key=lambda t: min(track_frames.get(t, {10**9})))
        anchor = tracks[0]
        rep = emb[anchor].astype(float).copy(); cnt = 1
        frames = set(track_frames.get(anchor, set()))
        taps = 1
        for t in tracks[1:]:
            tf = track_frames.get(t, set())
            sim = float(np.dot(rep / (np.linalg.norm(rep) + 1e-8), emb[t]))
            auto = (not (tf & frames)) and sim >= tau
            if not auto:
                taps += 1  # 카드 → 사람 탭
            rep = rep + emb[t]; cnt += 1; frames |= tf  # 어느 쪽이든 임보견에 흡수
        # 안전: 다른 개 트랙이 임보견으로 자동 오인되나
        repn = rep / (np.linalg.norm(rep) + 1e-8)
        fa = 0
        for t, dd in gt_map.items():
            if dd == d or t not in emb:
                continue
            tf = track_frames.get(t, set())
            if (not (tf & frames)) and float(np.dot(repn, emb[t])) >= tau:
                fa += 1
        out.append((taps, fa))
    return out


def foster_taps_multi(gt_map, track_embs, track_frames, tau):
    """멀티 레퍼런스 버전 — 트랙당 여러 샘플 임베딩, max 유사도로 매칭.

    sim(트랙, 임보견) = max over (트랙 샘플 × 임보견 레퍼런스 샘플). 평균이 각도를
    뭉개는 문제를 피해 같은 개 재인식↑ / 다른 개 구분↑ 동시에 노린다.
    """
    from collections import defaultdict
    dogs = defaultdict(list)
    for t, d in gt_map.items():
        if t in track_embs:
            dogs[d].append(t)

    def maxsim(A, B):  # 견고 유사도 — 상위 2개 쌍 평균(outlier 크롭 1개에 안 흔들림)
        if not len(A) or not len(B):
            return -1.0
        s = np.sort((A @ B.T).ravel())
        return float(s[-2:].mean()) if len(s) >= 2 else float(s[-1])

    out = []
    for d, tracks in dogs.items():
        tracks = sorted(tracks, key=lambda t: min(track_frames.get(t, {10**9})))
        anchor = tracks[0]
        refs = track_embs[anchor].copy()       # 임보견 레퍼런스 세트(성장)
        frames = set(track_frames.get(anchor, set()))
        taps = 1
        for t in tracks[1:]:
            tf = track_frames.get(t, set())
            auto = (not (tf & frames)) and maxsim(track_embs[t], refs) >= tau
            if not auto:
                taps += 1
            refs = np.vstack([refs, track_embs[t]]); frames |= tf  # 임보견에 흡수
        fa = 0
        for t, dd in gt_map.items():
            if dd == d or t not in track_embs:
                continue
            tf = track_frames.get(t, set())
            if (not (tf & frames)) and maxsim(track_embs[t], refs) >= tau:
                fa += 1
        out.append((taps, fa))
    return out


def _track_frames(pred_frames) -> dict[int, set]:
    tf: dict[int, set] = {}
    for pf in pred_frames:
        for d in pf.detections:
            tf.setdefault(d.track_id, set()).add(pf.frame_idx)
    return tf


def _load(name: str, gt_dir: str, emb_dir: str, pred_dir: str):
    gt_raw = io.read_json(Path(gt_dir) / f"{name}_m2.json")["mapping"]
    gt_map = {int(k): int(v) for k, v in gt_raw.items() if int(v) != -1}  # 진짜 개만
    z = np.load(Path(emb_dir) / f"{name}_m2.npz")
    tf = _track_frames(io.read_frames(Path(pred_dir) / f"{name}_m1.jsonl"))
    return gt_map, list(z["track_ids"]), z["embeddings"], tf


def _load_multi(name: str, gt_dir: str, emb_dir: str, pred_dir: str):
    """gt_map, track_embs(dict tid->[k,384]), track_frames."""
    gt_raw = io.read_json(Path(gt_dir) / f"{name}_m2.json")["mapping"]
    gt_map = {int(k): int(v) for k, v in gt_raw.items() if int(v) != -1}
    z = np.load(Path(emb_dir) / f"{name}_m2.npz")
    rt, re = z["row_tracks"], z["row_embeddings"]
    track_embs: dict[int, np.ndarray] = {}
    for tid in np.unique(rt):
        track_embs[int(tid)] = re[rt == tid]
    tf = _track_frames(io.read_frames(Path(pred_dir) / f"{name}_m1.jsonl"))
    return gt_map, track_embs, tf


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="m2_relink")
    p.add_argument("--gt-dir", default="data/dev/gt")
    p.add_argument("--emb-dir", default="data/dev/embeds")
    p.add_argument("--pred-dir", default="data/dev/preds")
    p.add_argument("--taus", default="0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--temporal", action="store_true",
                   help="시간적 상호배제(동시 등장 트랙 병합 금지) 적용")
    p.add_argument("--foster", action="store_true",
                   help="단독 임보견 시나리오(임보견 1마리만 추적, <2탭 목표)")
    p.add_argument("--multi", action="store_true",
                   help="멀티 레퍼런스(트랙당 여러 샘플, max 유사도) 매칭")
    args = p.parse_args(argv)

    names = sorted(z.name.replace("_m2.npz", "")
                   for z in Path(args.emb_dir).glob("*_m2.npz"))
    taus = [float(x) for x in args.taus.split(",")]

    if args.foster:
        mref = "멀티레퍼런스(max)" if args.multi else "평균"
        print(f"M2 — 단독 임보견 시나리오 [{mref}] (임보견 1탭 후 자동, 목표 <2탭/영상)")
        print(f"  {'τ':>5}{'평균탭/임보견':>14}{'<2탭 비율':>11}{'falseAttach':>13}")
        print("  " + "-" * 46)
        for tau in taus:
            taps_list = []; fa_total = 0
            for name in names:
                if args.multi:
                    gt_map, te, tf = _load_multi(name, args.gt_dir, args.emb_dir, args.pred_dir)
                    rows = foster_taps_multi(gt_map, te, tf, tau)
                else:
                    gt_map, tids, embs, tf = _load(name, args.gt_dir, args.emb_dir, args.pred_dir)
                    rows = foster_taps(gt_map, tids, embs, tf, tau)
                for taps, fa in rows:
                    taps_list.append(taps); fa_total += fa
            avg = sum(taps_list) / len(taps_list) if taps_list else 0
            under2 = sum(1 for t in taps_list if t < 2) / len(taps_list) if taps_list else 0
            print(f"  {tau:>5.2f}{avg:>14.2f}{under2:>11.0%}{fa_total:>13}")
        print(f"  * 임보견 시나리오 {len(taps_list)}건(영상별 각 개를 임보견으로). "
              f"falseAttach=다른 개를 임보견으로 자동 오인(0이어야 안전)")
        return 0

    n_vid = len(names)
    mode = "시간제약 ON" if args.temporal else "임베딩만"
    print(f"M2 — 자동병합 안전성 + 사람 탭  [{mode}]  (안 묶인 건 사람 카드로 = 설계대로)")
    print(f"  {'τ':>5}{'자동안전(P)':>11}{'falseMerge':>11}{'탭/영상':>9}{'자동절감':>9}")
    print("  " + "-" * 50)
    for tau in taus:
        TP = FP = 0
        interv = ideal = total_tracks = auto = 0
        for name in names:
            gt_map, tids, embs, tf = _load(name, args.gt_dir, args.emb_dir, args.pred_dir)
            res = relink(tids, embs, tau, tf if args.temporal else None)
            tp, fp, _ = _pair_counts(gt_map, res.mapping)
            TP += tp; FP += fp
            real = set(gt_map)
            # 사람 탭 = 자동 병합 안 된 트랙(각자 사람이 확인) = 앵커 생성 수
            interv += sum(1 for iv in res.interventions if iv.track_id in real)
            ideal += len(set(gt_map.values()))
            total_tracks += len(real)
            auto += sum(1 for t in real if res.relink_evidence[t]["auto"])
        prec = TP / (TP + FP) if TP + FP else 1.0
        taps_per_vid = interv / n_vid
        auto_rate = auto / total_tracks if total_tracks else 1.0
        print(f"  {tau:>5.2f}{prec:>11.2f}{FP:>11}{taps_per_vid:>9.1f}{auto_rate:>9.2f}")
    print(f"  * falseMerge=0 이 안전. 탭/영상 = 사용성(설계: 2~3 우아, 10+ 노가다).")
    print(f"    이상적 탭 = 개 마리수 ≈ {ideal/n_vid:.1f}/영상. 자동절감=사람 안 거치고 묶인 비율")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
