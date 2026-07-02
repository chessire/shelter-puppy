"""P0 최소 정규화기 — 소스 영상 → 분석용 H.264 mp4 + 매핑(시간·공간).

설계서 P0 의 '분석 분기'만 구현(출력 분기는 M6 소관). 한 번의 ffmpeg 패스로:
  디코드(autorotate) → [HDR면 톤맵] → CFR(fps=30) → 다운스케일(긴 변 768) → H.264 mp4

분석 mp4 는 CVAT 라벨링과 M1 추론이 **같은 좌표계**를 쓰도록 만드는 기준물.
회전은 ffmpeg autorotate 가 픽셀에 구워져 출력엔 rotation 태그가 없다(OpenCV 안전).

매핑(.map.json): 분석 박스를 원본 출력 좌표로 되돌리는 공간 변환 + 시간 정보.
  scale = orig_long / analysis_long. 원본 좌표 ≈ 분석 좌표 / (1/scale) = 분석 × scale.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

LONG_SIDE = 768
FPS = 30
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}  # PQ, HLG


def probe(path: str | Path) -> dict:
    """ffprobe 로 비디오 스트림 핵심 메타 추출."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,codec_name,r_frame_rate,color_transfer,duration,nb_frames",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return s


def rotation_tag(path: str | Path) -> int | None:
    """소스의 회전 메타(display matrix rotation)를 읽는다. 없으면 None.

    우리 데이터는 저장 픽셀이 이미 올바른 방향인데 rotation 태그만 잘못 붙는 경우가
    있다(autorotate 가 똑바른 픽셀을 돌려 눕힘). normalize 는 저장 픽셀을 신뢰
    (-noautorotate)하되, 태그가 있으면 경고 로그로 남겨 비정상 입력을 조기에 드러낸다.
    """
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "side_data=rotation", "-of", "default=nk=1:nw=1", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    if not out:
        return None
    try:
        return int(float(out.splitlines()[0]))
    except ValueError:
        return None


def _build_vf(is_hdr: bool, long_side: int) -> str:
    chain = []
    if is_hdr:
        # BT.2020/PQ·HLG → 선형화 → 톤맵 → BT.709 (분석용만).
        chain.append(
            "zscale=t=linear:npl=100,format=gbrpf32le,"
            "zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
        )
    chain.append(f"fps={FPS}")
    chain.append(
        f"scale={long_side}:{long_side}:"
        f"force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
    return ",".join(chain)


def normalize(src: str | Path, out_dir: str | Path, long_side: int = LONG_SIDE) -> dict:
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = probe(src)
    is_hdr = (meta.get("color_transfer") or "").lower() in HDR_TRANSFERS
    orig_w, orig_h = int(meta["width"]), int(meta["height"])

    # 회전: 저장 픽셀을 신뢰(-noautorotate). 우리 소스는 저장 방향이 곧 올바른 방향이라
    # autorotate 가 (잘못 붙은) rotation 태그를 적용해 똑바른 화면을 눕히는 사고를 막는다.
    # 태그가 있으면 경고만 남긴다(=비정상 입력 신호; 가로픽셀+회전태그 정상 파일이면 검토).
    rot = rotation_tag(src)
    if rot:
        print(f"[normalize][경고] {src.name}: rotation={rot}° 태그 — 저장 픽셀 기준으로 처리"
              f"(-noautorotate). 화면이 눕는다면 이 소스는 가로저장+회전 파일일 수 있음.")

    mp4_path = out_dir / f"{src.stem}.mp4"
    vf = _build_vf(is_hdr, long_side)
    cmd = [
        "ffmpeg", "-y", "-noautorotate", "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-an",  # 분석엔 오디오 불필요
        "-metadata:s:v:0", "rotate=0",  # 출력에서 회전 태그 제거(굽힘 없이 저장 그대로)
        "-movflags", "+faststart",
        str(mp4_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    # 결과 mp4 메타 재측정 (실제 분석 좌표계 확정).
    a = probe(mp4_path)
    aw, ah = int(a["width"]), int(a["height"])
    nb = int(a.get("nb_frames") or 0)
    scale = max(orig_w, orig_h) / max(aw, ah)  # 분석→원본 배율

    mapping = {
        "source": src.name,
        "analysis_mp4": mp4_path.name,
        "fps": FPS,
        "is_hdr": is_hdr,
        "tonemapped": is_hdr,
        "orig_size": [orig_w, orig_h],
        "analysis_size": [aw, ah],
        "scale_analysis_to_orig": scale,  # 원본좌표 ≈ 분석좌표 × scale
        "num_frames": nb,
        "note": "분석 박스[x,y,w,h] → 원본 출력 좌표 복원: 각 값 × scale (회전은 굽혀 있어 추가 보정 없음)",
    }
    map_path = out_dir / f"{src.stem}.map.json"
    with map_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    return mapping


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="normalize", description="P0 분석용 정규화")
    p.add_argument("sources", nargs="+", help="소스 영상 경로들")
    p.add_argument("--out", default="data/dev/analysis", help="출력 디렉토리")
    p.add_argument("--long-side", type=int, default=LONG_SIDE, help="긴 변 px")
    args = p.parse_args(argv)
    for s in args.sources:
        m = normalize(s, args.out, args.long_side)
        print(
            f"[OK] {m['source']} → {m['analysis_mp4']}  "
            f"{m['orig_size']}→{m['analysis_size']}  "
            f"{'HDR→SDR' if m['tonemapped'] else 'SDR'}  frames={m['num_frames']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
