"""텍스트 레이아웃·블록 걸침 카피 단위테스트 — 결정론 부분만.

배치 자체의 미감은 저작처럼 주관 축(취향 배터리) — 여기선 ①빈 영역 선택 산수
②가독성 산수 ③PlanText 소독 ④저작 출력 병합을 심판한다.
"""

from __future__ import annotations

from pipeline.m6_edit import AUTO_POS, EditBlock, EditPlan, PlanText
from pipeline.m6_edit import layout
from pipeline.m6_edit.author import _to_plan
from pipeline.m6_edit.run import _text_rect


# ── 영역-박스 겹침 산수 ─────────────────────────────────────────────────────

def test_rect_overlap():
    assert layout.rect_overlap((0, 0, 10, 10), (5, 5, 15, 15)) == 25.0
    assert layout.rect_overlap((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0   # 모서리 접촉
    assert layout.rect_overlap((0, 0, 10, 10), (20, 0, 30, 10)) == 0.0


def test_occupancy_worst_moment():
    rect = (0, 0, 10, 10)
    # max(프레임별 겹침) — 한 순간이라도 가리면 가린 것
    boxes = [(0, 0, 5, 10), (0, 0, 1, 1)]
    assert abs(layout.occupancy(rect, boxes) - 0.5) < 1e-9
    assert layout.occupancy(rect, []) == 0.0        # 박스 없음 = 제약 없음(안전한 저하)


def test_pick_region_avoids_subject():
    rects = {"bottom": (0, 80, 100, 100), "top": (0, 0, 100, 20),
             "left": (0, 40, 20, 60)}
    dog_bottom = [(10, 85, 90, 95)]                 # 주인공이 하단에
    assert layout.pick_region(rects, dog_bottom) == "top"
    assert layout.pick_region(rects, []) == "bottom"   # 박스 없음 → 관행 1순위


def test_pick_region_respects_taken_and_adjacency():
    rects = {p: (0, 0, 10, 10) for p in layout.AUTO_ORDER}
    # top 이 title 자리 → top 은 물론 인접 구석(top-left)도 제외(붙은 자리 금지)
    pos = layout.pick_region(rects, [], taken=["top"])
    assert pos == "bottom"
    pos = layout.pick_region(rects, [], taken=["top", "bottom"])
    assert pos not in (None, "top", "top-left", "bottom", "bottom-left", "bottom-right")


def test_pick_region_none_when_everything_conflicts():
    rects = {p: (0, 0, 10, 10) for p in layout.AUTO_ORDER}
    taken = ["top", "bottom", "left", "right"]      # 4방위 점유 → 구석 전부 인접
    assert layout.pick_region(rects, [], taken=taken) is None


def test_conflicts_adjacency_pairs():
    # 사용자 확정(2026-07-06): 상단+좌·우상단, 하단+좌·우하단, 좌단+좌상·좌하단,
    # 우단+우상·우하단은 같은 시간에 함께 못 쓴다. 대각(상단+좌하단 등)은 허용.
    assert layout.conflicts("top", "top-left") and layout.conflicts("top-left", "top")
    assert layout.conflicts("bottom", "bottom-right")
    assert layout.conflicts("left", "bottom-left")
    assert layout.conflicts("right", "top-right")
    assert layout.conflicts("top", "top")           # 같은 자리
    assert not layout.conflicts("top", "bottom-left")
    assert not layout.conflicts("top-left", "bottom-right")


def test_pick_region_all_occupied_falls_back_to_min():
    rects = {"bottom": (0, 80, 100, 100), "top": (0, 0, 100, 20)}
    boxes = [(0, 0, 100, 100), (0, 82, 100, 100)]   # 전 화면 + 하단 추가
    # 전부 임계 초과 → 겹침 최소 영역(동률이니 순서 첫 후보). 드롭은 이 함수 권한 아님.
    assert layout.pick_region(rects, boxes) in ("bottom", "top")


# ── 가독성 산수 ─────────────────────────────────────────────────────────────

def test_required_secs():
    assert layout.required_secs("가나다") == layout.TEXT_MIN_SHOW      # 하한
    assert abs(layout.required_secs("가" * 36) - 3.0) < 1e-9           # 36자/12cps


def test_resolve_window_span_kept():
    win = layout.resolve_window(0.0, 10.0, [0.2, 0.8], req=3.0)
    assert win == (2.0, 8.0)


def test_resolve_window_span_expanded_then_ok():
    # span 창(2s) < req(3s) → 범위 전체(10s)로 확장
    assert layout.resolve_window(0.0, 10.0, [0.0, 0.2], req=3.0) == (0.0, 10.0)


def test_resolve_window_drop_when_range_too_short():
    assert layout.resolve_window(0.0, 2.0, None, req=3.0) is None


def test_place_copy_default_dwells_then_disappears():
    # span 미지정 = 읽을 만큼 보였다 사라짐 — '심쿵 주의보!'(7자, req=하한 1.5s)가
    # 21초 범위 내내 떠 있던 실측 피드백(2026-07-06)의 수정.
    win = layout.place_copy(0.0, 21.0, None, "심쿵 주의보!", [])
    assert win is not None and win[0] == 0.0
    assert abs(win[1] - layout.TEXT_MIN_SHOW * layout.DWELL_FACTOR) < 1e-9


def test_place_copy_dwell_capped_by_range():
    win = layout.place_copy(0.0, 2.0, None, "짧은 카피", [])   # 범위가 체류보다 짧음
    assert win == (0.0, 2.0)


def test_place_copy_explicit_span_respected():
    # 명시 span 은 체류 기본값을 안 탄다 — [0,1] = 내내(저작이 상시를 의도한 것)
    assert layout.place_copy(0.0, 10.0, [0.0, 1.0], "카피", []) == (0.0, 10.0)
    assert layout.place_copy(0.0, 10.0, [0.5, 0.9], "카피", []) == (5.0, 9.0)


def test_place_copy_unreadable_drops():
    assert layout.place_copy(0.0, 2.0, None, "가" * 120, []) is None


def test_place_copy_waits_for_concurrency_gap():
    # 동시 상한 2(사용자 확정): title(상시) + 블록0 자막(0~5s)이 이미 2개 →
    # 카피는 자막이 끝나는 5s 에서야 뜬다(카피는 보조 — 항상 양보).
    windows = [(0.0, 21.0), (0.0, 5.0)]
    win = layout.place_copy(0.0, 21.0, None, "심쿵 주의보!", windows)
    assert win is not None and abs(win[0] - 5.0) < 1e-9


def test_place_copy_drops_when_no_gap():
    # title 상시 + 자막이 전 구간 연속 = 어디에 얹어도 3개 → 드롭
    windows = [(0.0, 10.0), (0.0, 5.0), (5.0, 10.0)]
    assert layout.place_copy(0.0, 10.0, None, "카피", windows) is None


def test_place_copy_explicit_span_clipped_to_gap():
    # 명시 span 창이 상한과 부분 충돌 → 자유 조각과의 교집합으로 잘려 배치
    windows = [(0.0, 10.0), (0.0, 4.0)]             # title + 앞 자막
    win = layout.place_copy(0.0, 10.0, [0.0, 0.8], "카피", windows)
    assert win is not None and abs(win[0] - 4.0) < 1e-9 and abs(win[1] - 8.0) < 1e-9


def test_title_window_dwell_and_cap():
    # title 도 소멸(2026-07-06 사용자) — 짧은 title 은 하한 1.5s × TITLE_DWELL_FACTOR
    w = layout.title_window("토리", 21.0)
    assert w == (0.0, layout.TEXT_MIN_SHOW * layout.TITLE_DWELL_FACTOR)
    assert layout.title_window("토리", 3.0) == (0.0, 3.0)   # 영상보다 길면 전체


def test_level3_concurrency_is_within_level():
    # 레벨 계약: L3 동시 상한은 같은 레벨끼리만 — L3 창 2개가 겹치는 구간은 회피
    l3 = [(0.0, 3.0), (1.0, 4.0)]                   # 1~3s 에 L3 두 개
    win = layout.place_copy(0.0, 10.0, None, "폴짝!", l3)
    assert win is not None and win[0] >= 3.0 - 1e-9  # 세 번째는 겹침 해소 후
    assert layout.fits_concurrency((5.0, 6.0), l3)
    assert not layout.fits_concurrency((1.5, 2.5), l3)


def test_l3_order_excludes_bands():
    # L3 무대는 중간 밴드 — top(L1)·bottom(L2)·top-right(배지)는 후보에 없다
    assert "top" not in layout.L3_ORDER
    assert "bottom" not in layout.L3_ORDER
    assert "top-right" not in layout.L3_ORDER


def test_peak_offset_snaps_to_jump_moment():
    import types
    from pipeline.m6_edit import EditBlock
    from pipeline.m6_edit.run import _foster_cache, _peak_offset
    from pipeline.workspace import Workspace
    ws = Workspace.dev()
    def box(cx):
        return types.SimpleNamespace(x=cx - 50, y=500, x2=cx + 50, y2=600)
    # 프레임 0~90: 정지(중심 400) → 프레임 45~60 사이 프레임당 30px 질주 → 정지
    boxes = {}
    cx = 400.0
    for i in range(91):
        if 45 <= i < 60:
            cx += 30.0
        boxes[i] = box(cx)
    _foster_cache[(str(ws.root), "FAKE_PEAK")] = boxes
    clips = [("/fake/FAKE_PEAK.mp4", 0.0, 3.0)]
    peak = _peak_offset([(10.0, EditBlock(), clips)], ws)
    assert peak is not None
    assert 10.0 + 45 / 30 - 0.1 <= peak <= 10.0 + 60 / 30 + 0.1   # 질주 창 안
    # 정지 소재(이동 0) → None(블록 시작 폴백)
    _foster_cache[(str(ws.root), "FAKE_STILL")] = {i: box(400) for i in range(91)}
    assert _peak_offset([(0.0, EditBlock(),
                          [("/fake/FAKE_STILL.mp4", 0.0, 3.0)])], ws) is None


# ── PlanText 소독 ──────────────────────────────────────────────────────────

def test_plantext_from_dict_sanitizes():
    t = PlanText.from_dict({"text": " 두 장면에 걸친 카피 ", "blocks": [3, 1],
                            "pos": "top-left", "span": [0.1, 0.9]})
    assert t.text == "두 장면에 걸친 카피"
    assert t.blocks == [1, 3]                       # 정렬
    assert t.pos == "top-left"
    assert t.span == [0.1, 0.9]


def test_plantext_invalid_fields_default():
    t = PlanText.from_dict({"text": "카피", "blocks": [2], "pos": "가운데",
                            "span": [0.9, 0.1]})
    assert t.blocks == [2, 2]                       # 단일 → 범위
    assert t.pos == AUTO_POS                        # 미지원 위치 → auto
    assert t.span is None                           # 역순 span → 전체
    assert PlanText.from_dict({"text": "카피", "blocks": "x"}).blocks is None


def test_editplan_from_dict_drops_invalid_texts():
    plan = EditPlan.from_dict({"blocks": [{"select": "all"}],
                               "texts": [{"text": "", "blocks": [0, 1]},
                                         {"text": "유효", "blocks": [0, 0]},
                                         {"text": "블록없음"}]})
    assert [t.text for t in plan.texts] == ["유효"]


def test_editblock_caption_pos_auto_accepted():
    assert EditBlock.from_dict({"select": "all", "caption_pos": "auto"}).caption_pos == AUTO_POS
    assert EditBlock.from_dict({"select": "all", "caption_pos": "없는곳"}).caption_pos == "bottom"


# ── 저작 출력 병합(_to_plan) ────────────────────────────────────────────────

def _raw(texts):
    return {"blocks": [{"sources": ["a"], "select": "all", "dur": 5,
                        "caption": "블록 자막"},
                       {"sources": ["b"], "select": "all", "dur": 5,
                        "caption": "둘째 자막"}],
            "texts": texts}


def test_to_plan_texts_kept_and_clamped():
    raw = _raw([{"text": "걸침 카피", "blocks": [0, 9]}])   # 끝 인덱스 초과
    plan = _to_plan(raw, "요청", ["a", "b"], narration=False)
    assert len(plan.texts) == 1
    assert plan.texts[0].blocks == [0, 1]                   # len(blocks)-1 로 클램프
    assert plan.texts[0].pos == AUTO_POS                    # 미지정 → auto


def test_to_plan_texts_capped_and_echo_dropped():
    req = "우리 토리 소개 영상 만들어줘"
    raw = _raw([{"text": req, "blocks": [0, 1]},            # 요청 복창 → 소독으로 빈 값
                {"text": "카피1", "blocks": [0, 0]},
                {"text": "카피2", "blocks": [1, 1]},
                {"text": "카피3", "blocks": [0, 1]}])       # MAX_TEXTS 초과
    plan = _to_plan(raw, req, ["a", "b"], narration=False)
    assert len(plan.texts) <= 2
    assert all(t.text and req not in t.text for t in plan.texts)


def test_to_plan_no_texts_when_caption_forbidden():
    raw = _raw([{"text": "카피", "blocks": [0, 1]}])
    plan = _to_plan(raw, "자막 없이 만들어줘", ["a", "b"],
                    narration=True, allow_caption=False)
    assert plan is not None and plan.texts == []


def test_to_plan_caption_is_level2_bottom():
    # 레벨 계약(2026-07-07): L2 설명은 하단 고정 — 저작 출력에 위치 필드가 없고,
    # raw 에 위치가 끼어 있어도 읽지 않는다(일관성이 자막의 미덕, 로밍 제거).
    raw = _raw([{"text": "카피", "blocks": [0, 1], "pos": "top"}])
    raw["title"] = "제목"
    raw["blocks"][0]["caption_pos"] = "top-left"
    plan = _to_plan(raw, "요청", ["a", "b"], narration=False)
    assert all(b.caption_pos == "bottom" for b in plan.blocks)
    assert plan.texts[0].pos == AUTO_POS            # L3 자리도 렌더러 몫


def test_to_plan_level3_char_contract():
    # L3 감탄: 공백 제외 8자 초과는 자르지 않고 통째로 드롭
    raw = _raw([{"text": "폴짝 폴짝 정말 신난다!", "blocks": [0, 0]},  # 비공백 10자 → 드롭
                {"text": "심쿵 주의보!", "blocks": [0, 1]}])           # 비공백 6자 → 유지
    plan = _to_plan(raw, "요청", ["a", "b"], narration=False)
    assert [t.text for t in plan.texts] == ["심쿵 주의보!"]


# ── 저작 재량 TTS (TTS 소유권 이진, 2026-07-07) ─────────────────────────────

def test_voice_discretion_gate():
    from pipeline.m5_tts.interpret import voice_discretion_allowed
    # TTS 무언급 + 자동 라우팅 edit → 재량 열림
    assert voice_discretion_allowed("edit", True, "우리 토리 소개 영상 만들어줘")
    # 유저가 목소리를 언급하면 방향 불문 꺼짐(긍정 핀=모드 A, 부정 핀=거부)
    assert not voice_discretion_allowed("edit", True, "음성 없이 만들어줘")
    assert not voice_discretion_allowed("narration", True, "대본 읽어줘")
    # 카드/수동/구버전 잡(출처 불명) → 유저 결정 취급, 꺼짐
    assert not voice_discretion_allowed("edit", False, "우리 토리 소개 영상 만들어줘")


def test_to_plan_voice_reads_caption_verbatim():
    raw = _raw([])
    raw["blocks"][0]["voice"] = True
    raw["blocks"][1]["voice"] = False
    plan = _to_plan(raw, "요청", ["a", "b"], narration=False, voice_choice=True)
    assert plan.blocks[0].narration == plan.blocks[0].caption == "블록 자막"
    assert plan.blocks[1].narration == ""


def test_to_plan_voice_ignored_without_choice():
    # 게이트 닫힘(유저가 TTS 언급) — voice 출력이 있어도 읽지 않는다
    raw = _raw([])
    raw["blocks"][0]["voice"] = True
    plan = _to_plan(raw, "요청", ["a", "b"], narration=False)
    assert all(not b.narration for b in plan.blocks)


def test_to_plan_voice_needs_caption():
    raw = _raw([])
    raw["blocks"][0]["caption"] = ""
    raw["blocks"][0]["voice"] = True
    plan = _to_plan(raw, "요청", ["a", "b"], narration=False, voice_choice=True)
    assert plan.blocks[0].narration == ""


def test_to_plan_voice_moderates_texts():
    raw = _raw([{"text": "카피1", "blocks": [0, 0]},
                {"text": "카피2", "blocks": [1, 1]}])
    raw["blocks"][0]["voice"] = True
    plan = _to_plan(raw, "요청", ["a", "b"], narration=False, voice_choice=True)
    assert len(plan.texts) == 1                     # 목소리 켜면 화면 글 절제


# ── 자막 사실 검수(프루닝 레시피의 자막판) — 결정론 병합부만 ────────────────

def test_apply_rewrites_merges_and_syncs_voice():
    from pipeline.m6_edit.author import _apply_rewrites
    plan = _to_plan(_raw([]), "요청", ["a", "b"], narration=False)
    plan.blocks[0].narration = plan.blocks[0].caption      # 재량 TTS(그대로 읽기)
    _apply_rewrites(plan, {0: "기록 범위의 새 자막", 1: ""}, "요청")
    assert plan.blocks[0].caption == "기록 범위의 새 자막"
    assert plan.blocks[0].narration == "기록 범위의 새 자막"  # 읽기 동기화
    assert plan.blocks[1].caption == "둘째 자막"             # 빈 재작성 = 무시


def test_apply_rewrites_sanitizes_and_bounds():
    from pipeline.m6_edit.author import _apply_rewrites
    req = "우리 토리 소개 영상 만들어줘"
    plan = _to_plan(_raw([]), req, ["a", "b"], narration=False)
    old = plan.blocks[0].caption
    _apply_rewrites(plan, {0: req, 7: "범위 밖", "x": "비정수"}, req)
    assert plan.blocks[0].caption == old                    # 복창 재작성 = 소독 후 무시


# ── 렌더러 기하(단일 출처) ──────────────────────────────────────────────────

def test_text_rect_regions_distinct():
    W, H = 1080, 1920
    r_bottom = _text_rect("자막", W, H, "bottom")
    r_top = _text_rect("자막", W, H, "top")
    r_tl = _text_rect("자막", W, H, "top-left")
    r_br = _text_rect("자막", W, H, "bottom-right")
    assert r_bottom[1] > H * 0.7 and r_top[3] < H * 0.5     # 상하 분리
    assert r_tl[0] < W * 0.3 and r_br[2] > W * 0.7          # 좌우 구석 분리
    assert layout.rect_overlap(r_bottom, r_top) == 0.0


def test_style_hierarchy_sizes():
    # 시각 위계: title > copy > caption (같은 텍스트의 rect 높이로 비교)
    W, H = 1080, 1920
    def h(style):
        r = _text_rect("토리", W, H, "bottom", style)
        return r[3] - r[1]
    assert h("title") > h("copy") > h("caption")


def test_pick_region_respects_blocked_rects():
    rects = {p: ((0, 0, 10, 10) if p == "bottom" else (20, 20, 30, 30))
             for p in layout.AUTO_ORDER}
    # bottom 자리가 이름 없는 카피 rect 와 겹침 → 다음 후보로
    pos = layout.pick_region(rects, [], blocked=[(5, 5, 15, 15)])
    assert pos == "top"


def test_beside_subject_picks_empty_side():
    from pipeline.m6_edit.run import _beside_subject
    W, H = 1080, 1920
    # 주인공이 왼쪽에 → 오른쪽 여백에, 얼굴 높이로, 피사체 쪽(왼쪽) 정렬
    boxes = [(100, 800, 400, 1200)] * 5
    got = _beside_subject("심쿵 주의보!", boxes, W, H)
    assert got is not None
    block, rect = got
    assert rect[0] > 400                                     # 박스 오른쪽
    assert block[-1] == "left"                               # 줄 정렬 = 피사체 쪽
    face_y = 800 + 0.35 * 400
    assert rect[1] < face_y < rect[3]                        # 얼굴 높이에 걸침


def test_beside_subject_none_when_no_room():
    from pipeline.m6_edit.run import _beside_subject
    W, H = 1080, 1920
    boxes = [(50, 400, 1030, 1500)] * 3                      # 화면을 꽉 채운 주인공
    assert _beside_subject("카피", boxes, W, H) is None
    assert _beside_subject("카피", [], W, H) is None         # 박스 없음 → 폴백
