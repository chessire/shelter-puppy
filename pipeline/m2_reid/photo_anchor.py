"""임보견 사진 앵커 — 고객 사진 몇 장으로 다견 영상의 임보견 트랙 자동 지정 (M2 재사용).

설계서 re-ID 의 '다각도 레퍼런스 임베딩'에서 레퍼런스를 시스템 자동수집 대신 *고객
제공 사진*으로 만든다 — 카드 탭은 잡마다지만 사진은 한 번 올리면 재사용되는 자산.

결정 규칙(2026-07-03 사용자 확정): 확신(절대 유사도 + 1·2위 격차) 넘으면 자동 확정,
아니면 그 영상만 uncertain — **사용자 선택은 애매한 것 중에서만**. M2 실측 falseAttach
1/12(유사견종 한계)가 격차 게이트의 근거. 실패 모드 안전: 억지 확정 없이 카드로.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ..harness import io
from ..workspace import Workspace
from .embed import DinoEmbedder, embed_tracks

# 골든 캘리브레이션(2026-07-03, 토리 사진 vs 다견 3영상 + 시각검증): 절대 유사도는
# 조명·화질에 취약(토리 본인도 0.27(밤)~0.63 출렁) → 약한 바닥으로만. 진짜 신호는
# *같은 영상 안* 트랙 간 격차. 단 격차는 **동시등장 트랙(=진짜 다른 개)**와만 비교 —
# 추적 조각(IDF1 0.17짜리 카페 영상에서 토리가 트랙 1·35로 갈라짐, 동시등장 0프레임,
# 시각검증 동일견)끼리 비교하면 본견 대 본견이라 가짜 애매가 난다(실측 격차 0.004).
# 조각 합침 후 진짜 경쟁자와의 격차: 9980 0.37 / 0066 0.08 / 0069 0.07 — 전부 정답
# 분리되는 값으로 임계 설정. [표본 3 잠정]
TAU_SIM = 0.40      # 절대 유사도 바닥 — 미만이면 "이 영상엔 그 개가 없을 수도"
TAU_MARGIN = 0.05   # 진짜 경쟁자와의 격차 하한 — 유사견종 오귀속(falseAttach) 방어
OVERLAP_EPS = 2     # 동시등장 ≤N프레임은 ID 스위치 순간 노이즈로 간주(같은 개 조각 허용)
FRAG_DELTA = 0.15   # top1 과 유사도 차 이내 + 비동시면 같은 개 조각으로 합침

# ── 앵커 전파(2026-07-03 사용자 설계) — 확정 영상 크롭 소수정예를 레퍼런스에 추가 ──
# 사진은 정적·도메인 갭(밤 0.27 실측)이 약점 → 확정 영상의 *동적 포즈* 크롭이 보강.
# 오염·희석 방어: ①사진 씨앗 불변(별도 벡터) ②top-K 소수 ③기증 자격 = 단독견(구조적
# 안전) 또는 확정 임계보다 엄격한 격차 ④유사컷 배제(같은 영상·프레임 거리 가중 중복도).
DONOR_K = 3            # 기증 컷 수 — 소수정예
DONOR_MIN_MARGIN = 0.2  # 다견 확정이 기증자가 되기 위한 격차 하한(확정 0.05보다 엄격)
DUP_W_VIDEO = 0.5      # 유사컷 가중치 ①: 같은 영상
DUP_W_DIST = 0.5       # 유사컷 가중치 ②: 프레임 거리(가까울수록 1)
DUP_DIST_NORM = 90     # 이 프레임 수(30fps 기준 3초) 이상 떨어지면 거리 페널티 소멸
DUP_MAX = 0.75         # 중복도 이 값 이상이면 유사컷 — 랭킹에서 제외

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".webp")


def ref_photos(ws: Workspace) -> list[Path]:
    """잡의 refs/ 폴더에서 임보견 레퍼런스 사진 목록."""
    d = ws.root / "refs"
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)


def _read_image(p: Path):
    """cv2 로 읽되, 아이폰 기본 HEIC 는 macOS sips 로 jpg 변환 폴백."""
    img = cv2.imread(str(p))
    if img is None and p.suffix.lower() == ".heic" and shutil.which("sips"):
        tmp = Path(tempfile.mkdtemp()) / (p.stem + ".jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", str(p), "--out", str(tmp)],
                       capture_output=True)
        img = cv2.imread(str(tmp))
    return img


def load_ref_embedding(photo_paths: list[Path], embedder: DinoEmbedder) -> np.ndarray:
    """사진들 → 평균 레퍼런스 임베딩(L2 정규화). M2 결론대로 멀티레퍼런스 아닌 평균."""
    crops = []
    for p in photo_paths:
        img = _read_image(p)
        if img is not None:
            crops.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not crops:
        raise SystemExit(f"레퍼런스 사진을 읽을 수 없음: {[str(p) for p in photo_paths]}")
    raw = embedder.embed(crops)
    raw = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
    mean = raw.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-8)


def fragments_and_margin(sims: dict[int, float], frame_sets: dict[int, set],
                         overlap_eps: int = OVERLAP_EPS,
                         frag_delta: float = FRAG_DELTA) -> tuple[list[int], float]:
    """조각 합침 + 진짜 경쟁자 격차 (순수 함수 — 테스트용 분리).

    top1 부터 유사도 내림차순으로, 기존 조각들과 동시등장이 없고(≤eps) 유사도가
    top1-delta 이내인 트랙을 같은 개 조각으로 합친다(시간상호배제의 역 — M2 신호 재사용).
    margin = top1 유사도 − 조각 밖(=동시등장한 진짜 다른 개) 최고 유사도.
    """
    order = sorted(sims, key=sims.get, reverse=True)
    if not order:
        return [], 0.0
    top = order[0]
    frag = [top]
    for t in order[1:]:
        if sims[t] < sims[top] - frag_delta:
            break
        if all(len(frame_sets.get(t, set()) & frame_sets.get(m, set())) <= overlap_eps
               for m in frag):
            frag.append(t)
    rivals = [t for t in order if t not in frag]
    margin = sims[top] - sims[rivals[0]] if rivals else 1.0
    return frag, round(margin, 4)


def match_video(ws: Workspace, name: str, candidate_tids, refs, embedder: DinoEmbedder) -> dict:
    """영상의 후보 dog 트랙별 대표 임베딩 ↔ 레퍼런스 코사인. 유의미 트랙만 임베딩.

    refs: 단일 벡터 또는 리스트(사진 + 기증). 트랙 점수 = max(레퍼런스별 코사인) —
    사진은 정적·기증 컷은 동적 도메인이라 어느 한쪽에라도 잘 붙으면 그 개다.
    """
    if isinstance(refs, np.ndarray):
        refs = [refs]
    frames = io.read_frames(ws.preds_m1(name))
    keep = {int(t) for t in candidate_tids}
    frame_sets: dict[int, set] = {t: set() for t in keep}
    for f in frames:
        f.detections = [d for d in f.detections
                        if d.cls == "dog" and d.track_id in keep]
        for d in f.detections:
            frame_sets[d.track_id].add(f.frame_idx)
    tids, means, _, _ = embed_tracks(ws.analysis(name), frames, embedder)
    sims = {int(t): max(float(means[i] @ r) for r in refs) for i, t in enumerate(tids)}
    frag, margin = fragments_and_margin(sims, frame_sets)
    top = frag[0] if frag else None
    return {"track": top, "tracks": frag, "sim": sims.get(top, 0.0),
            "margin": margin, "sims": sims}


def confident(res: dict, tau_sim: float = TAU_SIM, tau_margin: float = TAU_MARGIN) -> bool:
    """자동 확정 여부 — 절대 유사도와 격차 둘 다 넘어야(둘 중 하나 미달 = 카드)."""
    return (res["track"] is not None
            and res["sim"] >= tau_sim and res["margin"] >= tau_margin)


# --------------------------------------------------------------------------- #
# 앵커 전파 — 확정 영상 크롭 top-K 기증
# --------------------------------------------------------------------------- #
def dup_score(cand: dict, picked: list[dict],
              w_video: float = DUP_W_VIDEO, w_dist: float = DUP_W_DIST,
              dist_norm: int = DUP_DIST_NORM) -> float:
    """유사컷 중복도 0~1 (사용자 규칙: 같은 영상 가중치 + 프레임 거리 가중치).

    다른 영상이면 0. 같은 영상이면 기본 w_video, 프레임이 가까울수록 +w_dist·근접도
    — 같은 영상 멀리 떨어진 컷 0.5, 인접 컷 1.0.
    """
    worst = 0.0
    for s in picked:
        if s["video"] != cand["video"]:
            continue
        prox = max(0.0, 1.0 - abs(s["frame"] - cand["frame"]) / dist_norm)
        worst = max(worst, w_video + w_dist * prox)
    return worst


def select_diverse(cands: list[dict], k: int = DONOR_K, dup_max: float = DUP_MAX) -> list[dict]:
    """랭킹순 그리디 선별 — 중복도 dup_max 이상(유사컷)은 건너뛴다.

    cands 는 (기증자 격차 내림차순, 영상 내 모션 내림차순) 정렬돼 있어야 한다.
    """
    picked: list[dict] = []
    for c in cands:
        if len(picked) >= k:
            break
        if dup_score(c, picked) < dup_max:
            picked.append(c)
    return picked


def donor_reference(ws: Workspace, donor_specs: list[tuple], embedder: DinoEmbedder,
                    k: int = DONOR_K) -> tuple[np.ndarray | None, list[dict]]:
    """확정 영상들에서 다양성 강제 top-K 크롭 → 기증 레퍼런스 임베딩.

    donor_specs: (name, tracks|None, margin) — tracks=None 이면 단독견(최대 dog 박스,
    margin 1.0 취급이 관례). 후보 컷은 0.5초 간격 샘플 중 *박스 이동량 큰 프레임 우선*
    (사진의 정적 편향을 동적 포즈로 보강). 유사컷은 dup_score 로 배제.
    """
    from ..m4_action.gt_scaffold import foster_boxes_pred

    cands: list[dict] = []
    step = 15                                   # 분석본은 P0 CFR 30fps — 0.5초 간격
    for name, tracks, margin in donor_specs:
        boxes = foster_boxes_pred(name, ws, tracks)
        idxs = sorted(boxes)[::step]
        prev = None
        vid_cands = []
        for fi in idxs:
            bb = boxes[fi]
            cx, cy = (bb.x + bb.x2) / 2, (bb.y + bb.y2) / 2
            motion = 0.0
            if prev is not None:
                pcx, pcy, size = prev
                motion = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5 / max(size, 1.0)
            prev = (cx, cy, max(bb.w, bb.h))
            vid_cands.append({"video": name, "frame": fi, "bbox": bb,
                              "margin": margin, "motion": round(motion, 4)})
        vid_cands.sort(key=lambda c: -c["motion"])
        cands += vid_cands[:6]                  # 영상당 후보 상한(동적 상위)
    cands.sort(key=lambda c: (-c["margin"], -c["motion"]))
    picked = select_diverse(cands, k)
    if not picked:
        return None, []

    crops = []
    for c in picked:                            # 선별된 컷만 디코드·임베딩
        cap = cv2.VideoCapture(str(ws.analysis(c["video"])))
        cap.set(cv2.CAP_PROP_POS_FRAMES, c["frame"])
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        bb = c["bbox"]
        crop = img[max(0, int(bb.y)):int(bb.y2), max(0, int(bb.x)):int(bb.x2)]
        if crop.size:
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if not crops:
        return None, []
    raw = embedder.embed(crops)
    raw = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
    mean = raw.mean(axis=0)
    return mean / (np.linalg.norm(mean) + 1e-8), picked
