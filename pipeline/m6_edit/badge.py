"""AI 표시 배지 — AI 기본법 표시 의무 대응 (정책 개정 2026-07-02).

우상단 반투명 상시 배지 "AI 편집" + TTS 포함 렌더는 바로 아랫줄에
"내레이션: AI 음성"을 *상시* 표기한다(개정 전: 배지 문구 분기 + 초반 4초 안내 —
사라지는 안내보다 두 줄 고정이 보기 좋고 표시도 더 명확하다는 사용자 결정).
가상 음성은 법이 '사람이 명확히 인식할 수 있는 표시'를 요구하고, mp4 메타데이터는
플랫폼 재인코딩이 날려버려 단독으론 불충분 → 가시 배지 + 메타데이터 병행.
예술적·창의적 표현물 예외("향유 저해 않는 방식") 덕에 작은 반투명으로 충분하다.
위치는 쇼츠/릴스 UI safe-zone: 상단 10%·우측 5% 오프셋(가장자리 아이콘 회피).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_KFONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

BADGE_EDIT = "AI 편집"
NOTICE_TTS = "내레이션: AI 음성"
_META_COMMENT = "AI-generated content (AI 편집{tts})"


def _text_png(text: str, W: int, H: int, out: Path, *, rel_size: float,
              anchor: str, opacity: int) -> None:
    """배지/안내용 텍스트 PNG. anchor: 'topright'(배지) | 'topright2'(배지 아랫줄)."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(_KFONT, max(18, int(W * rel_size)))
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = int(W * 0.95) - tw                      # 우측 5% 오프셋
    y = int(H * 0.10)                           # 상단 10% 오프셋
    if anchor == "topright2":
        y += int(th * 2.2)                      # 배지 바로 아랫줄
    d.text((x - bb[0], y - bb[1]), text, font=font,
           fill=(255, 255, 255, opacity),
           stroke_width=2, stroke_fill=(0, 0, 0, opacity // 2))
    img.save(out)


def apply_badge(src: Path, out: Path, *, tts: bool, size: tuple[int, int]) -> None:
    """상시 배지 "AI 편집"(+TTS 는 아랫줄 "내레이션: AI 음성" 상시) + 메타데이터.
    오디오는 무재인코딩 통과."""
    W, H = size
    badge = out.with_name(out.stem + "_badge.png")
    _text_png(BADGE_EDIT, W, H, badge, rel_size=1 / 34, anchor="topright", opacity=170)

    inputs = ["-i", str(src), "-i", str(badge)]
    if tts:
        notice = out.with_name(out.stem + "_notice.png")
        _text_png(NOTICE_TTS, W, H, notice, rel_size=1 / 42, anchor="topright2", opacity=150)
        inputs += ["-i", str(notice)]
        fc = "[0:v][1:v]overlay=0:0[b];[b][2:v]overlay=0:0[v]"
    else:
        fc = "[0:v][1:v]overlay=0:0[v]"

    subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", fc,
                    "-map", "[v]", "-map", "0:a?", "-c:a", "copy",
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-metadata", "comment=" + _META_COMMENT.format(tts=" · AI 음성 내레이션" if tts else ""),
                    str(out)], check=True, capture_output=True)
    badge.unlink(missing_ok=True)
    if tts:
        notice.unlink(missing_ok=True)
