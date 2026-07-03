"""상비 관찰(observe) 단위테스트 — 결정론 부분만(센서 모델·ollama 불필요).

프로필 조립·gloss·모션 요약·매칭 소독이 맞아야 매칭 레시피를 심판할 수 있다(M0 원칙).
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from pipeline.m4_action.observe import (_cos, is_dark, match_keywords,
                                        motion_summary, profile_text,
                                        propagate_tags, same_place)
from pipeline.workspace import Workspace


# ── 휘도 밤/낮 (골든 마진: 밤 luma 41·암부 0.58 vs 나머지 126+·≤0.06) ──────

@pytest.mark.parametrize("luma,dark_frac,want", [
    (40.9, 0.58, True),    # IMG_0199 실측 — 밤
    (126.2, 0.06, False),  # IMG_0173 실측 — 낮(비 오는 날)
    (55.0, 0.10, True),    # 평균 어두움만으로도 밤
    (100.0, 0.60, True),   # 암부 비율만으로도 밤(역광 등)
    (60.0, 0.50, False),   # 경계값은 미달(초과만 밤)
])
def test_is_dark(luma, dark_frac, want):
    assert is_dark(luma, dark_frac) is want


# ── 프로필 → 관찰 기록 텍스트 (gloss = 순수 번역, 미등록 라벨은 원문 폴백) ──

def test_profile_text_gloss_and_fallback(monkeypatch, tmp_path):
    # 밀폐: 생성 gloss 자산을 tmp 에 크래프트 — 자산이 단일 출처(코드 상수 없음)
    from pipeline.m4_action import observe
    (tmp_path / "label_gloss_ko.json").write_text(json.dumps({
        "audio": {"Music": "음악 소리", "Speech": "말소리"},
        "places": {"veterinarians_office": "동물병원"},
    }), encoding="utf-8")
    monkeypatch.setattr(observe, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(observe, "_gloss_tables", {})
    prof = {
        "audio": [["Music", 0.68], ["Speech", 0.74], ["Didgeridoo", 0.5]],
        "places": {"indoor": True, "categories": [["veterinarians_office", 0.9],
                                                  ["zen_garden", 0.3]]},
        "luma": {"luma": 40.9, "dark_frac": 0.58, "dark": True},
        "caption": "실내에서 강아지 여러 마리가 논다.",
    }
    t = profile_text(prof)
    assert "음악 소리(0.68)" in t and "말소리(0.74)" in t
    assert "Didgeridoo(0.5)" in t                      # 표에 없음 → 원문 폴백
    assert "실내 확실" in t and "동물병원(0.9)" in t
    assert "zen_garden(0.3)" in t
    assert "어두움(밤/저조도)" in t
    assert t.endswith("묘사: 실내에서 강아지 여러 마리가 논다.")


def test_profile_text_no_gloss_asset(monkeypatch, tmp_path):
    # 자산 자체가 없으면 전부 원문 폴백(안전한 저하) — 크래시 없음
    from pipeline.m4_action import observe
    monkeypatch.setattr(observe, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(observe, "_gloss_tables", {})
    t = profile_text({"audio": [["Music", 0.68]]})
    assert "Music(0.68)" in t


def test_profile_text_empty_and_extra():
    t = profile_text({}, extra="동작측정: 거의 정지(움직임 0초/정지 8초)")
    assert t.startswith("소리: (신호 없음)")
    assert "동작측정: 거의 정지" in t
    # indoor 미확정(None)이면 '실내 확실'이 없어야 — 한쪽 방향 센서
    t2 = profile_text({"places": {"indoor": None, "categories": []}})
    assert "실내 확실" not in t2


# ── 모션 요약 (M4 군 시간 → 정성 표현; 라벨은 넣지 않는다) ─────────────────

def _write_m4(tmp_path, name, segments):
    preds = tmp_path / "preds"
    preds.mkdir(exist_ok=True)
    (preds / f"{name}_m4.json").write_text(
        json.dumps({"segments": segments}), encoding="utf-8")


@pytest.mark.parametrize("segs,phrase", [
    ([{"start_t": 0, "end_t": 8, "group": "static", "action": "앉기",
       "conf": 1.0, "uncertain": False}], "거의 정지"),
    ([{"start_t": 0, "end_t": 16, "group": "dynamic", "action": "걷기",
       "conf": 1.0, "uncertain": False}], "활발히 움직임"),
    ([{"start_t": 0, "end_t": 5, "group": "dynamic", "action": "걷기",
       "conf": 1.0, "uncertain": False},
      {"start_t": 5, "end_t": 10, "group": "static", "action": "앉기",
       "conf": 1.0, "uncertain": False}], "움직임 보통"),
])
def test_motion_summary(tmp_path, segs, phrase):
    _write_m4(tmp_path, "V", segs)
    line = motion_summary(Workspace(tmp_path), "V")
    assert line is not None and phrase in line
    assert "걷기" not in line and "앉기" not in line   # 거친 라벨 미노출

def test_motion_summary_missing(tmp_path):
    assert motion_summary(Workspace(tmp_path), "없는영상") is None


# ── 매칭 소독 (LLM 응답의 enum 재검증 — 초소형 출력 원칙의 결정론 짝) ──────

class _FakeOllama(types.ModuleType):
    """match_keywords 내부 `import ollama` 를 가로채는 스텁."""

    def __init__(self, replies):
        super().__init__("ollama")
        self._replies = list(replies)
        self.prompts: list[str] = []

    def chat(self, model, messages, options, format, think):  # noqa: A002
        self.prompts.append(messages[0]["content"])
        content = self._replies.pop(0)
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(message=msg)


def test_match_keywords_sanitize(monkeypatch):
    profiles = {"IMG_A": {"caption": "a"}, "IMG_B": {"caption": "b"}}
    # 환각 이름·중복은 소독, 깨진 JSON 은 빈 결과(빈 후보는 프루닝 호출 없음)
    fake = _FakeOllama([json.dumps({"videos": ["IMG_A", "IMG_A", "IMG_Z"]}),
                        json.dumps({"remove": []}),
                        "{깨진 json"])
    monkeypatch.setitem(sys.modules, "ollama", fake)
    out = match_keywords(profiles, ["카페", "산책"])
    assert out["카페"] == ["IMG_A"]
    assert out["산책"] == []
    assert len(fake.prompts) == 3                      # 매칭 2 + 프루닝 1(후보 있을 때만)
    assert "뒷받침하면 포함" in fake.prompts[0]        # 확정 레시피(완화 매칭)
    assert "명백히 반대" in fake.prompts[1]            # 확정 레시피(프루닝)


def test_match_keywords_prune(monkeypatch):
    """프루닝 = 검증 비대칭 — 제거만 가능, 환각 제거 이름은 무시."""
    profiles = {"IMG_A": {"caption": "a"}, "IMG_B": {"caption": "b"}}
    fake = _FakeOllama([json.dumps({"videos": ["IMG_A", "IMG_B"]}),
                        json.dumps({"remove": ["IMG_B", "IMG_Z"]})])
    monkeypatch.setitem(sys.modules, "ollama", fake)
    out = match_keywords(profiles, ["놀이"])
    assert out["놀이"] == ["IMG_A"]                    # B 프루닝, 환각 Z 무해


def test_match_keywords_extras_in_prompt(monkeypatch):
    profiles = {"IMG_A": {"caption": "a"}}
    fake = _FakeOllama([json.dumps({"videos": []})])
    monkeypatch.setitem(sys.modules, "ollama", fake)
    match_keywords(profiles, ["놀이"], extras={"IMG_A": "동작측정: 거의 정지"})
    assert "동작측정: 거의 정지" in fake.prompts[0]


def test_match_keywords_empty_inputs(monkeypatch):
    fake = _FakeOllama([])
    monkeypatch.setitem(sys.modules, "ollama", fake)
    assert match_keywords({}, ["카페"]) == {"카페": []}
    assert match_keywords({"IMG_A": {}}, []) == {}
    assert fake.prompts == []                          # 물을 게 없으면 호출 0


# ── 장면 전파 이중 AND 게이트 (같은 배경 AND 같은 소리풍경) ────────────────

def _prof(scene, audio):
    return {"scene_vec": scene, "audio_vec": audio}


def test_cos_degenerate():
    assert _cos([], [1.0]) == 0.0                      # 빈 벡터
    assert _cos([1.0, 0.0], [1.0]) == 0.0              # 길이 불일치
    assert _cos([0.0, 0.0], [1.0, 0.0]) == 0.0         # 영벡터
    assert _cos([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_same_place_and_gate():
    a = _prof([1.0, 0.0], [1.0, 0.0])
    both = _prof([0.9, 0.1], [0.9, 0.1])               # 둘 다 높음 → 통과
    vis_only = _prof([0.9, 0.1], [0.0, 1.0])           # 오디오가 차단(산책길 함정)
    aud_only = _prof([0.0, 1.0], [0.9, 0.1])           # 시각이 차단(말소리 함정)
    assert same_place(a, both)[0] is True
    assert same_place(a, vis_only)[0] is False
    assert same_place(a, aud_only)[0] is False


def test_propagate_tags():
    cafe_a = _prof([1.0, 0.0], [1.0, 0.0])             # 기증자(1단 확정)
    cafe_b = _prof([0.95, 0.05], [0.9, 0.1])           # 같은 장소 → 전파 대상
    other = _prof([0.0, 1.0], [0.9, 0.1])              # 소리만 닮음 → 차단
    profiles = {"A": cafe_a, "B": cafe_b, "C": other}
    adds = propagate_tags(profiles, {"카페": ["A"], "밤": []})
    assert [x[0] for x in adds["카페"]] == ["B"]        # C 차단, 기증자 A 자신 제외
    assert adds["카페"][0][1] == "A"                    # 근거(기증자) 기록
    assert "밤" not in adds                             # 기증자 없으면 전파 없음
