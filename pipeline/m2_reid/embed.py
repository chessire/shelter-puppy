"""트랙 임베딩 — 각 pred 트랙의 강아지 크롭을 DINOv2 로 임베딩해 대표 벡터 산출.

설계서: 박스 크롭 → 임베딩 벡터, 트랙별 대표 임베딩(평균/메도이드). 다각도
레퍼런스(여러 프레임 샘플)로 각도 변화에 강인하게.

- 트랙별로 최대 K 프레임 균등 샘플 → 크롭 → DINOv2 CLS 임베딩
- 트랙 대표 = 샘플 임베딩 평균 후 L2 정규화 (코사인 유사도용)
- 출력: data/embeds/<video>_m2.npz  (track_ids, embeddings[N,384])
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..harness import io

MODEL_NAME = "facebook/dinov2-small"
SAMPLES_PER_TRACK = 8


class DinoEmbedder:
    def __init__(self, name: str = MODEL_NAME):
        import torch
        from transformers import AutoModel, AutoImageProcessor
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()

    def embed(self, crops_rgb: list[np.ndarray]) -> np.ndarray:
        """RGB ndarray 크롭 리스트 → [N,384] CLS 임베딩 (정규화 전)."""
        from PIL import Image
        imgs = [Image.fromarray(c) for c in crops_rgb]
        inp = self.proc(images=imgs, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(**inp)
        return out.last_hidden_state[:, 0].cpu().numpy()


def _sample_indices(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def embed_tracks(
    analysis_mp4: Path, pred_frames, embedder: DinoEmbedder,
    k: int = SAMPLES_PER_TRACK, margin: float = 0.05,
) -> tuple[list[int], np.ndarray]:
    # track_id -> [(frame_idx, bbox), ...]
    by_track: dict[int, list] = defaultdict(list)
    for pf in pred_frames:
        for d in pf.detections:
            by_track[d.track_id].append((pf.frame_idx, d.bbox))

    cap = cv2.VideoCapture(str(analysis_mp4))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    track_ids: list[int] = []
    means: list[np.ndarray] = []
    row_tracks: list[int] = []
    rows: list[np.ndarray] = []
    for tid, items in sorted(by_track.items()):
        items.sort()
        picks = [items[i] for i in _sample_indices(len(items), k)]
        crops = []
        for frame_idx, bb in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, img = cap.read()
            if not ok:
                continue
            mx, my = bb.w * margin, bb.h * margin
            x1 = max(0, int(bb.x - mx)); y1 = max(0, int(bb.y - my))
            x2 = min(W, int(bb.x2 + mx)); y2 = min(H, int(bb.y2 + my))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            crops.append(crop)
        if not crops:
            continue
        raw = embedder.embed(crops)
        raw = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)  # 샘플별 정규화
        mean = raw.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-8)
        track_ids.append(tid); means.append(mean)
        for r in raw:               # 멀티 레퍼런스: 샘플 임베딩 전부 보관
            row_tracks.append(tid); rows.append(r)
    cap.release()
    mean_arr = np.vstack(means) if means else np.zeros((0, 384))
    row_arr = np.vstack(rows) if rows else np.zeros((0, 384))
    return track_ids, mean_arr, np.array(row_tracks), row_arr


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="m2_embed")
    p.add_argument("--analysis-dir", default="data/dev/analysis")
    p.add_argument("--pred-dir", default="data/dev/preds")
    p.add_argument("--out-dir", default="data/dev/embeds")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    embedder = DinoEmbedder()
    for pred_path in sorted(Path(args.pred_dir).glob("*_m1.jsonl")):
        name = pred_path.name.replace("_m1.jsonl", "")
        mp4 = Path(args.analysis_dir) / f"{name}.mp4"
        if not mp4.exists():
            continue
        tids, means, row_tracks, rows = embed_tracks(
            mp4, io.read_frames(pred_path), embedder)
        np.savez(out_dir / f"{name}_m2.npz",
                 track_ids=np.array(tids), embeddings=means,
                 row_tracks=row_tracks, row_embeddings=rows)
        print(f"[OK] {name}: 트랙 {len(tids)}개 (샘플 {len(rows)}개) → {name}_m2.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
