"""M4 러너 — 임보견 크롭 → Gemma 군판정 → M3 모션 대조 → uncertain 폴백.

설계서 레이어1 4·5단계:
  4. 동작 판별: 768px 임보견 크롭, 0.25초 샘플 → 7지선다 + logprob
  5. 모션 대조: 전체화면이 아닌 *임보견 박스* 모션으로 4번 결과 검증
역할분담(2026-06-30 확정): **동/정 군은 M3 전담**(카메라 보상 ROI 잔차 = 피사체 모션의
직접 측정), **gemma 는 '의미'만**(동작 라벨 + 묘기=손/제스처). gemma 는 정지 크롭이라
모션 판정이 불안정하고 특히 어두우면 *확신하며 오답*(밤 질주를 정적묘기로 0.99 확신) —
신뢰도 가중을 줬더니 그 가짜 확신이 군을 뒤집어 폐기했고, 군에서 gemma 를 완전히 뺐다.
검출 희소(박스 튐)면 uncertain(단 사용자가 키워드로 콕 집은 장면은 M6 가 uncertain 을
무시하고 씀 — 핀 우선). 묘기는 군과 직교한 라벨로, gemma 가 conf 로 확신할 때만 인정.

측정 격리: 크롭 ROI 는 GT 보간 박스(M1/M2 오류 분리). M3·GT 스캐폴드와 동일.
검증된 추론 레시피: gemma4 `think=False` + top_logprobs, 답 토큰 첫 글자로 군 합.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import ollama

from . import DYNAMIC, STATIC, foster_track
from .gt_scaffold import _sample_indices, foster_boxes
from ..harness import io
from ..harness.schemas import ActionSegment, BBox
from ..m3_motion.curve import compute_motion_curve, segment_motion
from ..workspace import Workspace

MODEL = "gemma4:26b-a4b-it-q4_K_M"
# 구간 내 임보견 박스가 이 비율 미만으로만 검출되면(희소·튐) M3 모션·크롭을 불신 → uncertain.
# '가짜 큰 잔차'(박스가 프레임마다 튀어 ROI 내용이 바뀜) 방지 가드.
MIN_COVERAGE = 0.25
PROMPT_BASE = (
    "이 강아지의 동작을 보기에서 고르세요. 다른 말 없이 보기 단어 하나만 출력.\n"
    "보기: 점프 달리기 걷기 앉기 엎드림 동적묘기 정적묘기\n"
    "앉거나 엎드린 자세에서 하이파이브·손·빵야·죽은척·구르기 같은 재주를 부리면 묘기"
    "(움직임 크면 동적묘기, 작으면 정적묘기)."
)

# 7지선다 답의 첫 글자 → 군 / 동작. 묘기 2종 추가(동→동적묘기, 정→정적묘기).
_FIRST = {"점": ("dynamic", "점프"), "달": ("dynamic", "달리기"),
          "걷": ("dynamic", "걷기"), "앉": ("static", "앉기"), "엎": ("static", "엎드림"),
          "동": ("dynamic", "동적묘기"), "정": ("static", "정적묘기")}


def _build_prompt(motion_hint: str = "") -> str:
    """7지선다 프롬프트 + M3 모션 수치 힌트. gemma 가 크롭과 모션을 함께 보고 판정한다.

    설계 결정(2026-06-30): M3 를 사후 일치체크가 아니라 *입력 피처*로. 묘기처럼 자세는
    정적이어도 모션이 큰 제스처를, gemma 가 모션값을 알고 동/정묘기로 정합하게.
    """
    return PROMPT_BASE + (f"\n참고: {motion_hint}" if motion_hint else "")


def _crop_jpeg(frame: np.ndarray, b: BBox, long_side: int = 768) -> bytes | None:
    H, W = frame.shape[:2]
    x1, y1 = max(0, int(b.x)), max(0, int(b.y))
    x2, y2 = min(W, int(b.x2)), min(H, int(b.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    c = frame[y1:y2, x1:x2]
    s = long_side / max(c.shape[0], c.shape[1])
    if s < 1.0:
        c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))
    ok, buf = cv2.imencode(".jpg", c)
    return buf.tobytes() if ok else None


def judge_crop(jpeg: bytes, motion_hint: str = ""):
    """크롭 1장 → (dyn_p, sta_p, top_action). 군 신호 못 읽으면 None.

    motion_hint: M3 모션 수치(있으면 프롬프트에 실어 gemma 가 함께 고려)."""
    r = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": _build_prompt(motion_hint), "images": [jpeg]}],
        options={"temperature": 0, "num_predict": 6},
        think=False, logprobs=True, top_logprobs=10,
    )
    lp = r.logprobs
    if not lp:
        return None
    first = lp[0] if isinstance(lp[0], dict) else lp[0].model_dump()
    by_action: dict[str, float] = {}
    for cand in first.get("top_logprobs", []):
        tok = (cand["token"] or "").strip()
        if not tok or tok[0] not in _FIRST:
            continue
        _, action = _FIRST[tok[0]]
        by_action[action] = by_action.get(action, 0.0) + math.exp(cand["logprob"])
    if not by_action:
        return None
    dyn_p = sum(p for a, p in by_action.items() if a in DYNAMIC)
    sta_p = sum(p for a, p in by_action.items() if a in STATIC)
    top_action = max(by_action, key=by_action.get)
    return dyn_p, sta_p, top_action


def classify_segment(cap, boxes, f0, f1, fps, max_crops, motion_hint: str = ""):
    """구간 [f0,f1) 크롭들을 판정해 (llm_group, llm_conf, action) 집계.

    motion_hint: 이 구간의 M3 모션 수치(gemma 입력 피처)."""
    dyn_fracs, actions = [], []
    for idx in _sample_indices(f0, f1, fps, max_n=max_crops):
        if idx not in boxes:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        jpeg = _crop_jpeg(frame, boxes[idx])
        if jpeg is None:
            continue
        res = judge_crop(jpeg, motion_hint)
        if res is None:
            continue
        dyn_p, sta_p, action = res
        tot = dyn_p + sta_p
        if tot > 0:
            dyn_fracs.append(dyn_p / tot)
            actions.append(action)
    if not dyn_fracs:
        return None, 0.0, None
    mean_dyn = float(np.mean(dyn_fracs))
    llm_group = "dynamic" if mean_dyn >= 0.5 else "static"
    llm_conf = mean_dyn if llm_group == "dynamic" else (1.0 - mean_dyn)
    # 대표 동작: 결정된 군에 속하는 동작 중 최빈값
    grp_set = DYNAMIC if llm_group == "dynamic" else STATIC
    in_grp = [a for a in actions if a in grp_set]
    action = Counter(in_grp or actions).most_common(1)[0][0]
    return llm_group, llm_conf, action


def run(name: str, thr: float, fps: float, conf_thr: float, max_crops: int,
        ws: Workspace | None = None) -> int:
    ws = ws or Workspace.dev()
    boxes = foster_boxes(name, ws)   # GT 있으면 GT, 없으면 pred(프로덕션)
    if not boxes:
        print(f"[{name}] 임보견 박스 없음 — M1 pred 확인 필요. 건너뜀.")
        return 1
    mp4 = ws.analysis(name)
    motion = compute_motion_curve(mp4, boxes)
    segs, _ = segment_motion(motion, fps, thr)

    cap = cv2.VideoCapture(str(mp4))
    out: list[ActionSegment] = []
    print(f"[{name}]: {len(segs)}구간 판정 중…")
    for s in segs:
        f0, f1 = int(s.start_t * fps), int(s.end_t * fps)
        # 동/정 군 = M3 전담(카메라 보상 ROI 잔차 = 피사체 모션의 직접 측정).
        group = "dynamic" if s.label == "moving" else "static"
        seg_vals = [motion[i] for i in range(f0, f1) if i in motion]
        mval = float(np.mean(seg_vals)) if seg_vals else 0.0
        coverage = sum(1 for i in range(f0, f1) if i in boxes) / max(1, f1 - f0)
        motion_hint = (f"이 구간 움직임 세기(M3)={mval:.1f} "
                       f"({'강함' if mval >= thr else '약함'}, 기준 {thr:.0f})")
        # gemma 는 '의미'만: 동작 라벨 + 묘기(손/제스처). 군엔 관여 안 함(llm_group 은 로그용).
        llm_group, llm_conf, action = classify_segment(
            cap, boxes, f0, f1, fps, max_crops, motion_hint)

        # 검출 희소(박스 튐)면 크롭·모션 불신 → uncertain. 그 외엔 M3 군을 commit.
        # (사용자가 키워드로 콕 집은 장면은 M6 가 이 uncertain 을 무시하고 씀 — 핀 우선.)
        uncertain = (action is None) or (coverage < MIN_COVERAGE)
        out.append(ActionSegment(
            start_t=round(s.start_t, 2), end_t=round(s.end_t, 2),
            group=group, action=action, conf=round(llm_conf, 3), uncertain=uncertain,
        ))
        flag = "UNCERTAIN" if uncertain else "commit"
        print(f"   {s.start_t:5.1f}~{s.end_t:5.1f}s  M3={mval:4.1f}->{group:7s} "
              f"cov{coverage:3.0%}  gemma={str(action):6s}(c{llm_conf:.2f}) [{flag}]")
    cap.release()

    out_path = ws.preds_m4(name)
    io.write_action_segments(out_path, out)
    print(f"   → {out_path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="m4_run")
    p.add_argument("names", nargs="*", help="영상 이름들 (기본: 테스트 5영상)")
    p.add_argument("--thr", type=float, default=8.0, help="M3 모션 동/정 임계")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--conf-thr", type=float, default=0.6,
                   help="logprob 군확신 하한(미만이면 uncertain) [잠정]")
    p.add_argument("--max-crops", type=int, default=10, help="구간당 최대 크롭 수")
    args = p.parse_args(argv)
    names = args.names or ["IMG_0004", "IMG_0008", "IMG_9980", "IMG_0069", "IMG_0066"]
    for n in names:
        run(n, args.thr, args.fps, args.conf_thr, args.max_crops)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
