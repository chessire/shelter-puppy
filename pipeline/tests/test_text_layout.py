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


def test_title_departure_frees_copy_slot():
    # title(0~4.5s) + 상시 자막(블록마다) 상황 — title 이 떠난 뒤 자막 1개뿐인
    # 구간이 생기고, 카피는 거기 들어간다(자막 span 없이도 카피가 사는 이유).
    title_w = layout.title_window("토리", 10.0)
    windows = [title_w, (0.0, 5.0), (5.0, 10.0)]    # title + 블록0·1 자막(상시)
    win = layout.place_copy(0.0, 10.0, None, "심쿵 주의보!", windows)
    assert win is not None and abs(win[0] - title_w[1]) < 1e-9


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


def test_to_plan_author_caption_defaults_auto():
    plan = _to_plan(_raw([]), "요청", ["a", "b"], narration=False)
    assert all(b.caption_pos == AUTO_POS for b in plan.blocks)


def test_to_plan_demotes_collisions_with_fixtures():
    # 상시 요소 충돌 소독 — title 있으면 top·top-left(인접), 항상 top-right → auto
    raw = _raw([{"text": "카피", "blocks": [0, 1], "pos": "top"}])
    raw["title"] = "제목"
    raw["blocks"][0]["caption_pos"] = "top-left"
    raw["blocks"][1]["caption_pos"] = "top-right"
    plan = _to_plan(raw, "요청", ["a", "b"], narration=False)
    assert plan.blocks[0].caption_pos == AUTO_POS
    assert plan.blocks[1].caption_pos == AUTO_POS
    assert plan.texts[0].pos == AUTO_POS


def test_to_plan_keeps_safe_explicit_pos():
    raw = _raw([])                                  # title 없음 → top 은 재량으로 허용
    raw["blocks"][0]["caption_pos"] = "top"
    raw["blocks"][1]["caption_pos"] = "bottom-left"
    plan = _to_plan(raw, "요청", ["a", "b"], narration=False)
    assert plan.blocks[0].caption_pos == "top"
    assert plan.blocks[1].caption_pos == "bottom-left"


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
