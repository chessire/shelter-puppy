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
    junk = ["카페", "all", "강아지 당겨찍기:foster, 전체:full],", "target_dur: 0,`, `",
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


def test_scene_consensus():
    """요청 주도 장면 추론 다수결 — 3장 중 2장 이상 일치만, 샘플 부족 시 하한 완화.

    보기(order)는 요청에서 오는 파라미터일 뿐 — 함수에 어휘 결합이 없음을 증명하기
    위해 픽스처도 추상 토큰으로 쓴다(특정 도메인 단어 금지).
    """
    from pipeline.m4_action.scene_auto import consensus
    order = ["kw_a", "kw_b", "kw_c", "kw_d"]
    assert consensus([["kw_a", "kw_c"], ["kw_a"], ["kw_c", "kw_b"]], order) == ["kw_a", "kw_c"]
    assert consensus([["kw_d"], [], []], order) == []              # 1/3 → 기각 (보수 게이트)
    assert consensus([["kw_b", "kw_a"]], order) == ["kw_a", "kw_b"]  # 샘플 1장 → 하한 완화
    assert consensus([], order) == []


def test_expand_paths(tmp_path):
    """--inputs/--dog-photos 폴더 확장 — 파일·폴더 혼용, 확장자 필터, 이름순."""
    from pipeline.job import _expand_paths, _IMG_SET, _VID_SET
    d = tmp_path / "vids"; d.mkdir()
    (d / "b.MOV").touch(); (d / "a.mp4").touch(); (d / "note.txt").touch()
    single = tmp_path / "x.mov"; single.touch()
    got = _expand_paths([d, single], _VID_SET, "영상")
    assert [p.name for p in got] == ["a.mp4", "b.MOV", "x.mov"]   # 폴더 이름순 + 파일 그대로
    p = tmp_path / "photos"; p.mkdir()
    (p / "dog.HEIC").touch(); (p / "dog.jpg").touch(); (p / "clip.mp4").touch()
    assert [q.name for q in _expand_paths([p], _IMG_SET, "사진")] == ["dog.HEIC", "dog.jpg"]
    empty = tmp_path / "empty"; empty.mkdir()
    with pytest.raises(SystemExit):        # 빈 폴더 = 명시적 에러(조용히 0개 방지)
        _expand_paths([empty], _VID_SET, "영상")


def test_fragments_and_margin():
    """추적 조각 합침 — 동시등장 없으면 같은 강아지, 격차는 진짜 경쟁자와만 (실측 형태)."""
    from pipeline.m2_reid.photo_anchor import fragments_and_margin
    # IMG_0066 꼴: 토리가 1·35로 조각(비동시), 11은 35와 동시등장(다른 강아지)
    sims = {1: 0.52, 35: 0.49, 11: 0.44}
    fs = {1: set(range(0, 77)), 35: set(range(100, 474)), 11: set(range(460, 608))}
    frag, margin = fragments_and_margin(sims, fs)
    assert frag == [1, 35] and margin == pytest.approx(0.08, abs=0.001)
    # IMG_9980 꼴: 두 강아지가 동시등장 + 유사도 격차 큼 → 조각 없음, 큰 margin
    frag, margin = fragments_and_margin({1: 0.63, 2: 0.26}, {1: set(range(200)), 2: set(range(200))})
    assert frag == [1] and margin == pytest.approx(0.37, abs=0.001)
    # 동시등장 1프레임(ID 스위치 노이즈)은 조각 허용
    frag, _ = fragments_and_margin({71: 0.41, 1: 0.37}, {71: {5, 6, 7}, 1: {7, 100, 101}})
    assert frag == [71, 1]
    assert fragments_and_margin({}, {}) == ([], 0.0)


def test_donor_dedupe():
    """앵커 전파 유사컷 배제 — 같은 영상 가중치 + 프레임 거리 가중치 (사용자 규칙)."""
    from pipeline.m2_reid.photo_anchor import dup_score, select_diverse
    a = {"video": "A", "frame": 100}
    assert dup_score({"video": "B", "frame": 100}, [a]) == 0.0        # 다른 영상 = 무페널티
    assert dup_score({"video": "A", "frame": 105}, [a]) > 0.9         # 인접 컷 ≈ 1.0
    assert dup_score({"video": "A", "frame": 400}, [a]) == 0.5        # 같은 영상, 먼 컷
    # 그리디: 랭킹순으로 훑되 인접 컷은 건너뛰고 다른 영상/먼 컷을 채움
    cands = [
        {"video": "A", "frame": 100, "margin": 1.0, "motion": 0.9},
        {"video": "A", "frame": 110, "margin": 1.0, "motion": 0.8},   # 인접 → 제외
        {"video": "B", "frame": 50,  "margin": 1.0, "motion": 0.7},
        {"video": "A", "frame": 400, "margin": 0.5, "motion": 0.6},   # 같은 영상, 멀리 → 허용
    ]
    got = select_diverse(cands, k=3)
    assert [(c["video"], c["frame"]) for c in got] == [("A", 100), ("B", 50), ("A", 400)]


def test_apply_reservation():
    """핀 소스 예약 — 남의 핀 소스는 폴백 블록이 못 쓴다(하이파이브 선점 사고)."""
    from pipeline.m6_edit.run import _apply_reservation
    src = [("cafe.mp4", "p1"), ("walk.mp4", "p2"), ("hi5.mp4", "p3")]
    pins = [set(), {"walk.mp4"}, {"hi5.mp4"}]   # 블록0=폴백, 블록1=산책 핀, 블록2=하이파이브 핀
    allowed = _apply_reservation(src, pins)
    assert [s[0] for s in allowed[0]] == ["cafe.mp4"]                 # 폴백은 무주공산만
    assert [s[0] for s in allowed[1]] == ["cafe.mp4", "walk.mp4"]     # 자기 핀 + 무주공산
    assert [s[0] for s in allowed[2]] == ["cafe.mp4", "hi5.mp4"]
    # 핀이 하나도 없으면 전 블록 전체 사용(기존 동작)
    assert _apply_reservation(src, [set(), set()]) == [src, src]


def test_photo_anchor_confident():
    """사진 앵커 결정 규칙 — 절대 유사도와 격차 둘 다 넘어야 자동 확정."""
    from pipeline.m2_reid.photo_anchor import confident
    ok = {"track": 1, "sim": 0.60, "margin": 0.33, "sims": {1: 0.60, 2: 0.26}}   # 9980 실측꼴
    low_sim = {"track": 1, "sim": 0.30, "margin": 0.30, "sims": {1: 0.30, 2: 0.0}}
    low_margin = {"track": 1, "sim": 0.48, "margin": 0.004, "sims": {1: 0.48, 2: 0.478}}  # 비숑 실측꼴
    none = {"track": None, "sim": 0.0, "margin": 1.0, "sims": {}}
    assert confident(ok)
    assert not confident(low_sim)      # 영상에 그 강아지가 없을 수도 → 카드
    assert not confident(low_margin)   # 유사견종 오귀속 방어 → 카드
    assert not confident(none)


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
