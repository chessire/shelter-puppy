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


def test_pick_region_respects_taken():
    rects = {"bottom": (0, 80, 100, 100), "top": (0, 0, 100, 20),
             "left": (0, 40, 20, 60)}
    dog_bottom = [(10, 85, 90, 95)]
    taken = [(0, 0, 100, 20)]                       # top 은 이미 title 자리
    assert layout.pick_region(rects, dog_bottom, taken) == "left"


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
    # 상시 요소 충돌 소독 — title 있으면 top, 항상 top-right → auto (실측 회귀)
    raw = _raw([{"text": "카피", "blocks": [0, 1], "pos": "top"}])
    raw["title"] = "제목"
    raw["blocks"][0]["caption_pos"] = "top"
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
