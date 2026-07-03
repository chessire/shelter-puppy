"""상비 관찰(observe) 단위테스트 — 결정론 부분만(센서 모델·ollama 불필요).

프로필 조립·gloss·모션 요약·매칭 소독이 맞아야 매칭 레시피를 심판할 수 있다(M0 원칙).
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from pipeline.m4_action.observe import (is_dark, match_keywords, motion_summary,
                                        profile_text)
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

def test_profile_text_gloss_and_fallback():
    prof = {
        "audio": [["Music", 0.68], ["Speech", 0.74], ["Didgeridoo", 0.5]],
        "places": {"indoor": True, "categories": [["veterinarians_office", 0.9],
                                                  ["zen_garden", 0.3]]},
        "luma": {"luma": 40.9, "dark_frac": 0.58, "dark": True},
        "caption": "실내에서 개 여러 마리가 논다.",
    }
    t = profile_text(prof)
    assert "음악(0.68)" in t and "사람 말소리(0.74)" in t
    assert "Didgeridoo(0.5)" in t                      # 미등록 → 원문
    assert "실내 확실" in t and "동물병원/동물시설(0.9)" in t
    assert "zen_garden(0.3)" in t
    assert "어두움(밤/저조도)" in t
    assert t.endswith("묘사: 실내에서 개 여러 마리가 논다.")


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
    # 환각 이름·중복은 소독, 깨진 JSON 은 빈 결과
    fake = _FakeOllama([json.dumps({"videos": ["IMG_A", "IMG_A", "IMG_Z"]}),
                        "{깨진 json"])
    monkeypatch.setitem(sys.modules, "ollama", fake)
    out = match_keywords(profiles, ["카페", "산책"])
    assert out["카페"] == ["IMG_A"]
    assert out["산책"] == []
    assert len(fake.prompts) == 2                      # 키워드당 1호출(초소형)
    assert "뒷받침하면 포함" in fake.prompts[0]        # 확정 레시피(완화 지시)


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
