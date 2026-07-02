"""M5 단위테스트 — 정규화기·CER·조립기(합성 픽스처, 엔진 불필요).

메트릭·조립이 맞아야 엔진을 심판할 수 있다(M0 원칙). 엔진 자체는 venv-tts 통합
스모크(assemble CLI)로 검증 — 여기선 프로젝트 venv 만으로 도는 결정론 부분만.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from pipeline.m5_tts.assemble import assemble, cer
from pipeline.m5_tts.normalize_ko import native, normalize, sino


# ── 정규화기 ──────────────────────────────────────────────

@pytest.mark.parametrize("src,want", [
    ("토리는 3살 추정, 몸무게 7.2kg의 작은 믹스견이에요.",
     "토리는 세 살 추정, 몸무게 칠 점 이 킬로그램의 작은 믹스견이에요."),
    ("2026년 1월 22일, 보호소에서 처음 만났어요.",
     "이천이십육년 일월 이십이일, 보호소에서 처음 만났어요."),
    ("강아지 2마리와 고양이 1마리", "강아지 두 마리와 고양이 한 마리"),
    ("하루 30분씩 산책해요", "하루 삼십분씩 산책해요"),
    ("체중이 10.05kg입니다", "체중이 십 점 영 오 킬로그램입니다"),
    ("15번 버스", "열다섯 번 버스"),
    ("3개월 된 아기", "삼개월 된 아기"),   # 개월은 한자어 (개 counter 오매칭 방지)
    ("2시간 산책", "두 시간 산책"),        # 시간은 고유어 (시 counter 와 구분)
    ("숫자 없는 문장은 그대로", "숫자 없는 문장은 그대로"),
])
def test_normalize(src, want):
    assert normalize(src) == want


def test_sino_native():
    assert sino(0) == "영"
    assert sino(110) == "백십"          # 일십 아님
    assert sino(2026) == "이천이십육"
    assert sino(10000) == "일만"
    assert native(1) == "한"
    assert native(20) == "스물"
    assert native(34) == "서른네"
    assert native(100) == sino(100)     # 고유어 범위 밖 → 한자어 폴백


# ── CER ──────────────────────────────────────────────────

def test_cer():
    assert cer("우리 토리", "우리 토리") == 0.0
    assert cer("우리 토리!", "우리, 토리") == 0.0          # 문장부호 무시
    assert cer("토리는 최고", "프린은 최고") == pytest.approx(3 / 5)  # 토→프 리→린 는→은
    assert cer("", "아무거나") == 0.0                       # 빈 ref 가드


# ── 모드 판정 — 결정론 핀 (Gemma 불필요 계층) ───────────────

def test_pin_mode():
    from pipeline.m5_tts.interpret import pin_mode
    assert pin_mode("토리 소개 내레이션 넣어줘") == "narration"
    assert pin_mode("대본 읽어줘: 우리 토리…") == "narration"
    assert pin_mode("TTS로 소개해줘") == "narration"
    # 부정 핀이 긍정 핀("음성")보다 먼저 — "음성 없이"가 narration 으로 새면 안 된다
    assert pin_mode("음성 없이 편집만 해줘") == "edit"
    assert pin_mode("자막만 넣어줘") == "edit"
    assert pin_mode("신나는 장면 위주로 편집해줘") is None      # 애매 → Gemma 몫


def test_clean_keywords():
    from pipeline.m5_tts.interpret import _clean_keywords
    junk = ["카페", "all", "임보견 당겨찍기:foster, 전체:full],", "target_dur: 0,`, `",
            "산책", "아주아주아주아주 길어서 키워드일 리 없는 문자열"]
    assert _clean_keywords(junk) == ["카페", "산책"]


def test_attribute_directives():
    """리터럴 지시(N초·클로즈업) 거리 기반 결정론 귀속 — 실제 e2e 요청 패턴."""
    from pipeline.m5_tts.interpret import attribute_directives
    from pipeline.m6_edit import EditBlock
    req = ("먼저 토리 얼굴을 천천히 클로즈업으로 5초 보여주고 아래에 텍스트로 "
           "우리 토리를 소개합니다를 띄워주면서 읽어줘. 그 다음 애견카페에서 뛰어 노는 "
           "모습 10초 편집해서 똥꼬발랄한 우리 토리 띄워주고 읽어줘.")
    sents = ["우리 토리를 소개합니다", "똥꼬발랄한 우리 토리"]
    blocks = [EditBlock(select="static"), EditBlock(select="dynamic")]
    attribute_directives(req, sents, blocks)
    assert blocks[0].target_dur == 5.0 and blocks[0].zoom == "gradual" \
        and blocks[0].subject == "foster"
    assert blocks[1].target_dur == 10.0 and blocks[1].zoom == "none"
    # Gemma 가 이미 채운 값은 존중
    blocks2 = [EditBlock(target_dur=7.0), EditBlock()]
    attribute_directives(req, sents, blocks2)
    assert blocks2[0].target_dur == 7.0
    # 대본 문장이 원문에 없으면(모델 작성 대본) 건드리지 않음
    blocks3 = [EditBlock(), EditBlock()]
    attribute_directives(req, ["원문에 없는 문장", "이것도 없음"], blocks3)
    assert blocks3[0].target_dur is None


def test_editblock_narration_field():
    from pipeline.m6_edit import EditBlock
    b = EditBlock.from_dict({"select": "static", "narration": "우리 토리를 소개합니다."})
    assert b.narration == "우리 토리를 소개합니다."
    assert EditBlock.from_dict({"select": "all"}).narration == ""   # 모드 B 하위호환


# ── 조립기 — 경계는 조립이 정의한다 ─────────────────────────

def _fixture_wav(path: Path, sec: float, sr: int = 24000):
    data = (np.sin(np.linspace(0, 440 * 2 * np.pi * sec, int(sr * sec))) * 8000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def test_assemble_timeline(tmp_path):
    _fixture_wav(tmp_path / "a.wav", 1.0)
    _fixture_wav(tmp_path / "b.wav", 2.0)
    rows = [
        {"id": "p00", "text": "가", "wav": str(tmp_path / "a.wav"),
         "pause_after": 0.5, "asr_flag": False},
        {"id": "p01", "text": "나", "wav": str(tmp_path / "b.wav"),
         "pause_after": 0.3, "asr_flag": False},
    ]
    res = assemble(rows, tmp_path / "out.wav", lead_in=0.2)
    tl = res["timeline"]
    # 경계 = lead_in + 누적(구절길이 + 무음) — 측정이 아니라 산수
    assert tl[0]["start"] == pytest.approx(0.2)
    assert tl[0]["end"] == pytest.approx(1.2)
    assert tl[1]["start"] == pytest.approx(1.7)
    assert tl[1]["end"] == pytest.approx(3.7)
    assert res["total_sec"] == pytest.approx(0.2 + 1.0 + 0.5 + 2.0 + 0.3, abs=0.01)
    with wave.open(str(tmp_path / "out.wav")) as w:
        assert w.getframerate() == 24000
        assert w.getnframes() / 24000 == pytest.approx(res["total_sec"], abs=0.01)
