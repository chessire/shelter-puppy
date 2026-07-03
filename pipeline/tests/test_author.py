"""저작 모드 단위테스트 — 결정론 부분(감지·소독·블록 소스 필터)만.

저작 출력 자체는 의도된 비결정(temperature 상향)이라 골든 채점 대상이 아니다 —
여기선 "언제 저작하나"와 "저작 출력이 안전하게 소독되나"를 심판한다.
"""

from __future__ import annotations

import json
import sys
import types

from pipeline.m6_edit import EditBlock, EditPlan
from pipeline.m6_edit.author import (_clean_text, author_plan, caption_forbidden,
                                     field_gaps, fill_plan, is_unstructured,
                                     script_invented)
from pipeline.m6_edit.run import _block_sources
from pipeline.workspace import Workspace


# ── 저작 발동 감지 (구조 검사 — 문자열 검사 아님) ──────────────────────────

def test_is_unstructured_default_plan():
    assert is_unstructured(EditPlan(blocks=[EditBlock()])) is True


def test_is_unstructured_structured_plans():
    assert is_unstructured(EditPlan(blocks=[EditBlock(), EditBlock()])) is False
    assert is_unstructured(EditPlan(blocks=[EditBlock(keywords=["카페"])])) is False
    assert is_unstructured(EditPlan(blocks=[EditBlock(caption="자막")])) is False
    assert is_unstructured(EditPlan(blocks=[EditBlock(target_dur=5.0)])) is False
    assert is_unstructured(EditPlan(blocks=[EditBlock(select="dynamic")])) is False
    assert is_unstructured(EditPlan(blocks=[EditBlock(zoom="gradual")])) is False
    assert is_unstructured(EditPlan(blocks=[EditBlock()], title="제목")) is False


def test_script_invented():
    req = "우리 토리를 소개합니다 를 띄우면서 읽어줘"
    given = EditPlan(blocks=[EditBlock(narration="우리 토리를 소개합니다")])
    blind = EditPlan(blocks=[EditBlock(narration="안녕하세요 토리예요")])
    silent = EditPlan(blocks=[EditBlock()])
    assert script_invented(given, req) is False    # 고객 대본 = 원문에 존재
    assert script_invented(blind, req) is True     # 장님 작문 → 저작으로 교체
    assert script_invented(silent, req) is False   # 대본 없음(모드 B 결)


# ── 텍스트 소독 (요청 복창·이모지 — 실측 유출 2종) ─────────────────────────

def test_clean_text_request_echo():
    req = "우리 토리 소개 영상 만들어줘"
    # 실측: 마지막 블록 caption 에 요청 전문 복창 — 복창만 걷고 창작은 살린다
    assert _clean_text("우리 토리 소개 영상 만들어줘 (가족이 되어주세요!)", req) \
        == "가족이 되어주세요!"
    assert _clean_text("우리 토리 소개 영상 만들어줘", req) == ""
    assert _clean_text("가족이 되어주세요!", req) == "가족이 되어주세요!"   # 무관 텍스트 보존


def test_clean_text_emoji():
    # 실측: title 에 🐾 유출 — PIL 기본 폰트가 두부(□)로 렌더
    assert _clean_text("🐾 강아지 '토리'를 찾아라! 🐾", "요청") == "강아지 '토리'를 찾아라!"
    cleaned = _clean_text("토리 ❤️ 최고", "요청")
    assert "❤" not in cleaned and "토리" in cleaned and "최고" in cleaned


# ── 저작 출력 소독 ─────────────────────────────────────────────────────────

class _FakeOllama(types.ModuleType):
    def __init__(self, replies):
        super().__init__("ollama")
        self._replies = list(replies)
        self.prompts: list[str] = []
        self.opts: list[dict] = []

    def chat(self, model, messages, options, format, think):  # noqa: A002
        self.prompts.append(messages[0]["content"])
        self.opts.append(options)
        return types.SimpleNamespace(
            message=types.SimpleNamespace(content=self._replies.pop(0)))


def _ws_with_profiles(tmp_path, names):
    ws = Workspace(tmp_path)
    ws.write_meta({"scene_profile": {n: {"caption": f"{n} 관찰"} for n in names}})
    return ws


def test_author_plan_sanitize(monkeypatch, tmp_path):
    names = ["IMG_A", "IMG_B"]
    ws = _ws_with_profiles(tmp_path, names)
    reply = json.dumps({"title": "우리 토리", "blocks": [
        {"sources": ["IMG_A", "IMG_A", "IMG_Z"], "select": "static",
         "dur": 99, "zoom": "gradual", "caption": "인사", "narration": "안녕"},
        {"sources": ["IMG_B"], "select": "dynamic", "dur": 1,
         "caption": "놀이", "narration": "신나요"},
    ] + [{"sources": ["IMG_A"], "select": "all", "caption": f"c{i}"}
         for i in range(9)]})
    fake = _FakeOllama([reply])
    monkeypatch.setitem(sys.modules, "ollama", fake)
    plan = author_plan("소개 영상 만들어줘", ws, names, narration=True)
    assert len(plan.blocks) == 6                       # MAX_BLOCKS 클램프
    b0, b1 = plan.blocks[0], plan.blocks[1]
    assert b0.sources == ["IMG_A"]                     # 환각 IMG_Z·중복 소독
    assert b0.target_dur == 12.0 and b1.target_dur == 2.0   # dur 클램프
    assert b0.subject == "foster"                      # 줌 블록 → 강아지 중심
    assert all(b.speed == 1.0 for b in plan.blocks)    # 저작은 배속 금지
    assert b0.narration == "안녕"
    assert plan.title == "우리 토리"
    assert fake.opts[0]["temperature"] > 0             # 의도된 비결정
    assert "소개 영상 만들어줘" in fake.prompts[0]     # 요청 느낌 전달
    assert "IMG_A 관찰" in fake.prompts[0]             # 관찰 프로필 전달


def test_author_plan_strips_narration_in_edit_mode(monkeypatch, tmp_path):
    names = ["IMG_A"]
    ws = _ws_with_profiles(tmp_path, names)
    reply = json.dumps({"blocks": [{"sources": ["IMG_A"], "select": "all",
                                    "caption": "c", "narration": "읽지마"}]})
    fake = _FakeOllama([reply])
    monkeypatch.setitem(sys.modules, "ollama", fake)
    plan = author_plan("소개 영상", ws, names, narration=False)
    assert plan.blocks[0].narration == ""              # 모드 B — TTS 없음


def test_author_plan_failure_returns_none(monkeypatch, tmp_path):
    names = ["IMG_A"]
    ws = _ws_with_profiles(tmp_path, names)
    fake = _FakeOllama(["{깨진", "{또깨진"])            # JSON 2회 실패
    monkeypatch.setitem(sys.modules, "ollama", fake)
    assert author_plan("소개 영상", ws, names, narration=False) is None
    fake2 = _FakeOllama([json.dumps({"blocks": []})] * 2)   # 빈 블록 — 재추첨 후 폴백
    monkeypatch.setitem(sys.modules, "ollama", fake2)
    assert author_plan("소개 영상", ws, names, narration=False) is None
    assert len(fake2.prompts) == 2


def test_author_plan_retries_on_textless(monkeypatch, tmp_path):
    """자막·대본 전무 = 구조 무효 → 재추첨(실측: 전 블록 caption 빈 값 복권)."""
    names = ["IMG_A"]
    ws = _ws_with_profiles(tmp_path, names)
    textless = json.dumps({"blocks": [{"sources": ["IMG_A"], "select": "all",
                                       "dur": 5, "caption": ""}]})
    good = json.dumps({"blocks": [{"sources": ["IMG_A"], "select": "all",
                                   "dur": 5, "caption": "안녕 토리"}]})
    fake = _FakeOllama([textless, good])
    monkeypatch.setitem(sys.modules, "ollama", fake)
    plan = author_plan("소개 영상", ws, names, narration=False)
    assert plan is not None and plan.blocks[0].caption == "안녕 토리"
    assert len(fake.prompts) == 2                      # 1차 기각 → 재추첨


# ── 부분 저작 (그라디언트 — 유저 뼈대 불변, 빈 필드만 병합) ────────────────

def test_caption_forbidden():
    assert caption_forbidden("자막 없이 만들어줘") is True
    assert caption_forbidden("텍스트없이 조용하게") is True
    assert caption_forbidden("자막으로 우리 토리 띄워줘") is False


def test_field_gaps_scan():
    full = EditPlan(blocks=[EditBlock(keywords=["산책"], caption="자막", target_dur=5.0)])
    assert field_gaps(full, narration=False) == []
    sketch = EditPlan(blocks=[EditBlock(keywords=["산책"]),          # 자막·초 빈칸
                              EditBlock(select="dynamic")])          # 전부 빈칸
    assert field_gaps(sketch, narration=False) == \
        ["블록0:caption+dur", "블록1:sources+caption+dur"]
    # 모드 A 는 sources 만 — dur 은 내레이션이 정하고 자막은 wants_caption 관례
    assert field_gaps(sketch, narration=True) == ["블록1:sources"]
    # 자막 거부 핀이면 caption 은 갭이 아니다
    assert field_gaps(sketch, narration=False, allow_caption=False) == \
        ["블록0:dur", "블록1:sources+dur"]


def test_fill_plan_no_gaps_no_llm(monkeypatch, tmp_path):
    ws = _ws_with_profiles(tmp_path, ["IMG_A"])
    fake = _FakeOllama([])
    monkeypatch.setitem(sys.modules, "ollama", fake)
    plan = EditPlan(blocks=[EditBlock(keywords=["산책"], caption="자막", target_dur=5.0)])
    assert fill_plan("풀 스펙 요청", ws, ["IMG_A"], plan, narration=False) is plan
    assert fake.prompts == []                          # 갭 없음 = 저작 0호출


def test_fill_plan_merge_preserves_user_fields(monkeypatch, tmp_path):
    names = ["IMG_A", "IMG_B"]
    ws = _ws_with_profiles(tmp_path, names)
    user_b0 = EditBlock(keywords=["산책"], caption="유저 자막", target_dur=5.0)
    empty_b1 = EditBlock(select="dynamic")
    plan = EditPlan(blocks=[user_b0, empty_b1])
    reply = json.dumps({"blocks": [
        {"caption": "덮어쓰기 시도", "dur": 9, "sources": ["IMG_B"]},   # 전부 무시돼야
        {"caption": "신나게 놀아요!", "dur": 7, "sources": ["IMG_B", "IMG_Z"]},
    ]})
    fake = _FakeOllama([reply])
    monkeypatch.setitem(sys.modules, "ollama", fake)
    out = fill_plan("소개 영상, 놀다가 산책", ws, names, plan, narration=False)
    b0, b1 = out.blocks
    assert (b0.caption, b0.target_dur) == ("유저 자막", 5.0) and not b0.sources  # 불변
    assert b1.caption == "신나게 놀아요!" and b1.target_dur == 7.0
    assert b1.sources == ["IMG_B"]                     # 환각 IMG_Z 소독
    assert "확정된 구성" in fake.prompts[0] and "유저 자막" in fake.prompts[0]


def test_fill_plan_respects_caption_pin(monkeypatch, tmp_path):
    ws = _ws_with_profiles(tmp_path, ["IMG_A"])
    plan = EditPlan(blocks=[EditBlock(keywords=["산책"], target_dur=5.0)])  # caption 만 빈칸
    fake = _FakeOllama([])
    monkeypatch.setitem(sys.modules, "ollama", fake)
    out = fill_plan("자막 없이 산책 5초", ws, ["IMG_A"], plan, narration=False)
    assert out.blocks[0].caption == "" and fake.prompts == []   # 갭 아님 → 호출 0


def test_fill_plan_json_failure_keeps_plan(monkeypatch, tmp_path):
    ws = _ws_with_profiles(tmp_path, ["IMG_A"])
    plan = EditPlan(blocks=[EditBlock(select="dynamic")])
    fake = _FakeOllama(["{깨진", "{또깨진"])
    monkeypatch.setitem(sys.modules, "ollama", fake)
    out = fill_plan("놀아줘", ws, ["IMG_A"], plan, narration=False)
    assert out is plan and out.blocks[0].caption == ""  # 기본값 렌더 폴백


def test_author_plan_caption_forbidden(monkeypatch, tmp_path):
    """전체 저작에서도 자막 거부 핀 — 무자막 플랜이 유효로 통과."""
    names = ["IMG_A"]
    ws = _ws_with_profiles(tmp_path, names)
    reply = json.dumps({"blocks": [{"sources": ["IMG_A"], "select": "all",
                                    "dur": 5, "caption": "지워질 자막"}]})
    fake = _FakeOllama([reply])
    monkeypatch.setitem(sys.modules, "ollama", fake)
    plan = author_plan("자막 없이 소개 영상 만들어줘", ws, names, narration=False)
    assert plan is not None and plan.blocks[0].caption == ""
    assert len(fake.prompts) == 1                      # 무자막이어도 재추첨 안 함


# ── 블록 소스 필터 (저작 직접 지정 우선, 전멸 시 키워드 폴백) ──────────────

def test_block_sources_direct_pick(tmp_path):
    sources = [("/x/IMG_A.mp4", "a.json"), ("/x/IMG_B.mp4", "b.json")]
    b = EditBlock(sources=["IMG_B"])
    picked, pinned = _block_sources(sources, b, Workspace(tmp_path))
    assert [m for m, _ in picked] == ["/x/IMG_B.mp4"]
    assert pinned == {"/x/IMG_B.mp4"}                  # 지정 = 핀(예약·면제)


def test_block_sources_fallback_when_no_match(tmp_path):
    sources = [("/x/IMG_A.mp4", "a.json")]
    b = EditBlock(sources=["IMG_Z"])                   # 전멸 → 키워드 경로 폴백
    picked, pinned = _block_sources(sources, b, Workspace(tmp_path))
    assert picked == sources and pinned == set()       # 키워드도 없음 = 전체·핀 없음
