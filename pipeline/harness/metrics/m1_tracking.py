"""M1 검출·추적 메트릭 — CLEAR-MOT(MOTA·ID-switch·박스누락) + IDF1.

설계서 측정 지표:
  - ID switch 횟수          → idsw
  - 박스 누락률(miss rate)   → miss_rate = FN / num_gt
  - 트랙 fragmentation       → fragmentation
  - MOTA / IDF1 차용         → mota, idf1

매칭은 IoU>=iou_thr. CLEAR-MOT 관례대로 직전 프레임의 대응을 우선 유지해
허위 ID switch를 줄인 뒤 나머지를 헝가리안으로 채운다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..schemas import FrameDetections
from .common import iou


@dataclass
class M1Metrics:
    num_gt: int          # 전체 GT 박스 수
    num_pred: int        # 전체 예측 박스 수
    fp: int              # 오검출 (예측인데 정답 없음)
    fn: int              # 누락 (정답인데 예측 없음)
    idsw: int            # ID switch 횟수
    fragmentation: int   # 트랙 끊김 횟수 (GT 트랙이 추적 끊겼다 재개)
    mota: float          # 1 - (FN+FP+IDSW)/num_gt
    miss_rate: float     # FN / num_gt
    idf1: float          # 2·IDTP / (2·IDTP + IDFP + IDFN)

    def as_dict(self) -> dict:
        return {
            "num_gt": self.num_gt,
            "num_pred": self.num_pred,
            "fp": self.fp,
            "fn": self.fn,
            "idsw": self.idsw,
            "fragmentation": self.fragmentation,
            "mota": round(self.mota, 4),
            "miss_rate": round(self.miss_rate, 4),
            "idf1": round(self.idf1, 4),
        }


def _align_frames(
    gt: list[FrameDetections], pred: list[FrameDetections]
) -> list[tuple[FrameDetections, FrameDetections]]:
    """**GT가 라벨한 프레임에서만** GT/pred 를 짝짓는다 (스파스 키프레임 GT 지원).

    사람이 0.5~1초 간격 키프레임만 라벨하므로, 라벨 안 한 프레임의 pred 박스를
    FP로 오인하면 안 된다. GT 프레임열 = 채점 대상. pred 에 해당 frame_idx 가
    없으면 빈 프레임(누락)으로 본다. GT 한 프레임이 detections=[] 면 '라벨했으나
    강아지 없음'(true negative) 으로 정상 채점된다.
    """
    pmap = {f.frame_idx: f for f in pred}
    out = []
    for g in sorted(gt, key=lambda f: f.frame_idx):
        p = pmap.get(g.frame_idx, FrameDetections(g.frame_idx, g.t, []))
        out.append((g, p))
    return out


def compute_clearmot(
    gt: list[FrameDetections], pred: list[FrameDetections], iou_thr: float = 0.5
) -> tuple[int, int, int, int, int]:
    """CLEAR-MOT 누적치 반환: (num_gt, fp, fn, idsw, fragmentation)."""
    num_gt = fp = fn = idsw = frag = 0

    # gt_id 별로 마지막에 매칭됐던 pred_id (ID switch 판정 기준).
    last_pred_for_gt: dict[int, int] = {}
    # gt_id 가 직전 (시간상 가장 최근) 프레임에서 매칭됐는지 (fragmentation 판정).
    was_tracked: dict[int, bool] = {}

    for g_frame, p_frame in _align_frames(gt, pred):
        g_dets = g_frame.detections
        p_dets = p_frame.detections
        num_gt += len(g_dets)

        matched_g: set[int] = set()
        matched_p: set[int] = set()
        cur_match: dict[int, int] = {}  # gt_id -> pred_id (이 프레임)

        # 1) 직전 대응 우선 유지: 같은 (gt_id,pred_id)가 여전히 IoU>=thr면 잠금.
        for gi, gd in enumerate(g_dets):
            prev_pid = last_pred_for_gt.get(gd.track_id)
            if prev_pid is None:
                continue
            for pj, pd in enumerate(p_dets):
                if pj in matched_p or pd.track_id != prev_pid:
                    continue
                if iou(gd.bbox, pd.bbox) >= iou_thr:
                    matched_g.add(gi)
                    matched_p.add(pj)
                    cur_match[gd.track_id] = pd.track_id
                    break

        # 2) 남은 것들 헝가리안.
        rem_g = [gi for gi in range(len(g_dets)) if gi not in matched_g]
        rem_p = [pj for pj in range(len(p_dets)) if pj not in matched_p]
        if rem_g and rem_p:
            m = np.zeros((len(rem_g), len(rem_p)))
            for a, gi in enumerate(rem_g):
                for b, pj in enumerate(rem_p):
                    m[a, b] = iou(g_dets[gi].bbox, p_dets[pj].bbox)
            rows, cols = linear_sum_assignment(-m)
            for a, b in zip(rows, cols):
                if m[a, b] >= iou_thr:
                    gi, pj = rem_g[a], rem_p[b]
                    matched_g.add(gi)
                    matched_p.add(pj)
                    cur_match[g_dets[gi].track_id] = p_dets[pj].track_id

        # 3) 집계: FP/FN.
        fp += len(p_dets) - len(matched_p)
        fn += len(g_dets) - len(matched_g)

        # 4) ID switch & fragmentation.
        for gid, pid in cur_match.items():
            prev = last_pred_for_gt.get(gid)
            if prev is not None and prev != pid:
                idsw += 1
            # 직전 관측에서 끊겼다가(was_tracked=False) 다시 잡히면 fragmentation.
            if gid in was_tracked and not was_tracked[gid]:
                frag += 1
            last_pred_for_gt[gid] = pid

        # 5) was_tracked 갱신: 이 프레임에 등장한 gt 중 매칭 여부 기록.
        present = {g_dets[gi].track_id for gi in range(len(g_dets))}
        matched_ids = set(cur_match.keys())
        for gid in present:
            was_tracked[gid] = gid in matched_ids

    return num_gt, fp, fn, idsw, frag


def compute_idf1(
    gt: list[FrameDetections], pred: list[FrameDetections], iou_thr: float = 0.5
) -> tuple[float, int, int, int]:
    """전역 ID 매칭으로 IDF1 계산. 반환 (idf1, idtp, idfp, idfn).

    각 (gt_id, pred_id) 쌍이 같은 프레임에서 IoU>=thr 로 겹친 프레임 수를 세고,
    전역 이분매칭으로 IDTP(매칭된 겹침 프레임 합)를 최대화한다.
    """
    # gt_id/pred_id 별 등장 프레임 수, 그리고 쌍별 겹침 프레임 수.
    gt_len: dict[int, int] = {}
    pred_len: dict[int, int] = {}
    pair: dict[tuple[int, int], int] = {}

    for g_frame, p_frame in _align_frames(gt, pred):
        for gd in g_frame.detections:
            gt_len[gd.track_id] = gt_len.get(gd.track_id, 0) + 1
        for pd in p_frame.detections:
            pred_len[pd.track_id] = pred_len.get(pd.track_id, 0) + 1
        # 이 프레임 내 IoU>=thr 인 (gt,pred) 쌍을 겹침으로 카운트(헝가리안 1:1).
        g_dets = g_frame.detections
        p_dets = p_frame.detections
        if g_dets and p_dets:
            m = np.zeros((len(g_dets), len(p_dets)))
            for a, gd in enumerate(g_dets):
                for b, pd in enumerate(p_dets):
                    m[a, b] = iou(gd.bbox, pd.bbox)
            rows, cols = linear_sum_assignment(-m)
            for a, b in zip(rows, cols):
                if m[a, b] >= iou_thr:
                    key = (g_dets[a].track_id, p_dets[b].track_id)
                    pair[key] = pair.get(key, 0) + 1

    total_gt = sum(gt_len.values())
    total_pred = sum(pred_len.values())
    if total_gt == 0 and total_pred == 0:
        return 1.0, 0, 0, 0

    gids = list(gt_len.keys())
    pids = list(pred_len.keys())
    gidx = {g: i for i, g in enumerate(gids)}
    pidx = {p: i for i, p in enumerate(pids)}
    G, P = len(gids), len(pids)

    # 비용행렬: 크기 (G+P)x(P+G).
    #  - 실-실(i<G, j<P): (len_i - m_ij) + (len_j - m_ij)  [IDFN+IDFP 기여]
    #  - gt i 미매칭(더미 col P+i): len_i
    #  - pred j 미매칭(더미 row G+j): len_j
    INF = 1e9
    N = G + P
    cost = np.full((N, N), INF)
    for (gid, pid), c in pair.items():
        i, j = gidx[gid], pidx[pid]
        li, lj = gt_len[gid], pred_len[pid]
        cost[i, j] = (li - c) + (lj - c)
    # 실-실 미겹침 쌍도 매칭 가능(겹침 0): 비용 li+lj.
    for gi, gid in enumerate(gids):
        for pj, pid in enumerate(pids):
            if cost[gi, pj] >= INF:
                cost[gi, pj] = gt_len[gid] + pred_len[pid]
    # 더미: gt 미매칭.
    for gi, gid in enumerate(gids):
        cost[gi, P + gi] = gt_len[gid]
    # 더미: pred 미매칭.
    for pj, pid in enumerate(pids):
        cost[G + pj, pj] = pred_len[pid]
    # 더미-더미 = 0.
    for gi in range(G):
        for pj in range(P):
            cost[G + pj, P + gi] = 0.0

    rows, cols = linear_sum_assignment(cost)
    total_cost = cost[rows, cols].sum()
    # IDFN+IDFP = total_cost → IDTP = (total_gt + total_pred - total_cost)/2
    idtp = int(round((total_gt + total_pred - total_cost) / 2))
    idfn = total_gt - idtp
    idfp = total_pred - idtp
    denom = 2 * idtp + idfp + idfn
    idf1 = (2 * idtp / denom) if denom > 0 else 1.0
    return idf1, idtp, idfp, idfn


def evaluate_m1(
    gt: list[FrameDetections], pred: list[FrameDetections], iou_thr: float = 0.5
) -> M1Metrics:
    num_gt, fp, fn, idsw, frag = compute_clearmot(gt, pred, iou_thr)
    idf1, idtp, idfp, idfn = compute_idf1(gt, pred, iou_thr)
    num_pred = sum(len(f.detections) for f in pred)
    mota = 1.0 - (fn + fp + idsw) / num_gt if num_gt > 0 else 1.0
    miss_rate = fn / num_gt if num_gt > 0 else 0.0
    return M1Metrics(
        num_gt=num_gt,
        num_pred=num_pred,
        fp=fp,
        fn=fn,
        idsw=idsw,
        fragmentation=frag,
        mota=mota,
        miss_rate=miss_rate,
        idf1=idf1,
    )
