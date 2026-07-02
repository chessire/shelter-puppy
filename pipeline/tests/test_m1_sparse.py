"""스파스 키프레임 GT 채점 검증 — 라벨 안 한 프레임의 pred 는 무시돼야 한다."""

from pipeline.harness.schemas import BBox, Detection, FrameDetections
from pipeline.harness.metrics.m1_tracking import evaluate_m1


def _f(idx, dets):
    return FrameDetections(idx, idx * 0.25, dets)


def _dog(tid, x):
    return Detection(tid, BBox(x, 0, 10, 10), 1.0, "dog")


def test_unlabeled_frames_ignored():
    # GT 는 키프레임 0,15,30 만 라벨. pred 는 매 프레임(0..30) 출력.
    gt = [_f(0, [_dog(1, 0)]), _f(15, [_dog(1, 15)]), _f(30, [_dog(1, 30)])]
    pred = [_f(i, [_dog(1, i)]) for i in range(31)]  # 모든 프레임 출력
    m = evaluate_m1(gt, pred)
    # 라벨한 3프레임만 채점 → 완벽. 사이 프레임 pred 는 FP 가 아님.
    assert m.num_gt == 3
    assert (m.fp, m.fn, m.idsw) == (0, 0, 0)
    assert m.mota == 1.0 and m.idf1 == 1.0


def test_labeled_empty_frame_is_true_negative():
    # 키프레임 15 는 라벨했고 개가 없음(detections=[]). pred 가 거기 박스를 내면 FP.
    gt = [_f(0, [_dog(1, 0)]), _f(15, []), _f(30, [_dog(1, 30)])]
    pred = [_f(0, [_dog(1, 0)]), _f(15, [_dog(1, 15)]), _f(30, [_dog(1, 30)])]
    m = evaluate_m1(gt, pred)
    assert m.num_gt == 2          # 라벨된 박스는 2개(15는 빈 프레임)
    assert m.fp == 1 and m.fn == 0  # 15의 pred 박스 = 오검출
