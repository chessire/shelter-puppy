"""RAG 결정론 파트 단위 테스트 — 롤업·인용 선택·L2 조립·게이트 (DB/ollama 불필요).

실행: python -m pytest rag/tests
"""

import pytest

from rag.db import vec_literal
from rag.ingest import CITE_CONF, _cite, foster_gate, l2_rows, rollup


def _seg(s, e, group="dynamic", conf=1.0, uncertain=False):
    return {"start_t": s, "end_t": e, "group": group, "conf": conf,
            "uncertain": uncertain}


# --------------------------------------------------------------------------- #
# rollup
# --------------------------------------------------------------------------- #
def test_rollup_ratio_and_total():
    r = rollup([[_seg(0, 6, "dynamic"), _seg(6, 8, "static")]])
    assert r["total_secs"] == pytest.approx(8.0)
    assert r["dynamic_ratio"] == pytest.approx(6 / 8)
    assert r["uncertain_pct"] == 0.0


def test_rollup_uncertain_half_weight():
    # uncertain dynamic 4초(0.5 가중=2) vs static 2초 → 비율 2/4
    r = rollup([[_seg(0, 4, "dynamic", conf=1.0, uncertain=True),
                 _seg(4, 6, "static")]])
    assert r["dynamic_ratio"] == pytest.approx(2 / 4)
    assert r["uncertain_pct"] == pytest.approx(4 / 6)


def test_rollup_conf_scales_with_observation():
    short = rollup([[_seg(0, 6)]])
    long = rollup([[_seg(0, 90)]])
    assert short["conf"] < 1.0
    assert long["conf"] == 1.0          # 상한 캡


def test_rollup_empty():
    r = rollup([])
    assert r["total_secs"] == 0.0 and r["dynamic_ratio"] == 0.0


# --------------------------------------------------------------------------- #
# _cite — 인용 구간 선택
# --------------------------------------------------------------------------- #
def test_cite_picks_best_conf():
    segs = [_seg(0, 2, conf=0.8), _seg(2, 5, conf=0.95)]
    assert _cite(segs)["start_t"] == 2


def test_cite_excludes_uncertain_and_low_conf():
    segs = [_seg(0, 2, conf=0.99, uncertain=True),
            _seg(2, 4, conf=CITE_CONF - 0.05)]
    assert _cite(segs) is None


# --------------------------------------------------------------------------- #
# l2_rows — 관찰 원문 결정론 조립 (재작성 없음 = 원문이 부분문자열로 보존)
# --------------------------------------------------------------------------- #
def test_l2_rows_assembles_behavior_and_caption():
    prof = {"behavior": "사람 손을 향해 앞발을 내밀며 상호작용",
            "caption": "실내 나무 바닥 위의 흰 강아지"}
    rows = l2_rows("IMG_1", prof, [_seg(1, 3, conf=0.9)], {"실내"})
    assert len(rows) == 2
    beh, cap = rows
    assert prof["behavior"] in beh["text"]          # 원문이 그대로 보존돼야 함
    assert "장면: 실내" in beh["text"]
    assert beh["t_start"] == 1 and beh["conf"] == pytest.approx(0.9)
    assert prof["caption"] in cap["text"]
    assert cap["t_start"] is None                   # 장면 서술엔 인용 구간 없음


def test_l2_rows_no_citation_without_confident_segment():
    rows = l2_rows("IMG_1", {"behavior": "달림"},
                   [_seg(0, 2, conf=0.5)], set())
    assert rows[0]["t_start"] is None and rows[0]["conf"] is None


def test_l2_rows_empty_profile():
    assert l2_rows("IMG_1", {}, [], set()) == []


# --------------------------------------------------------------------------- #
# 게이트·리터럴
# --------------------------------------------------------------------------- #
def test_foster_gate_blocks_uncertain():
    with pytest.raises(RuntimeError):
        foster_gate({"foster_uncertain": ["IMG_2"]})
    foster_gate({"foster_uncertain": []})           # 확정이면 통과


def test_vec_literal():
    assert vec_literal([0.5, -1.0, 0.25]) == "[0.5,-1,0.25]"
