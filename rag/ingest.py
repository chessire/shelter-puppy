"""인제스트 — 잡 산출물 → video_analysis + L2 청크(관찰 원문) + L1 청크(trait 요약).

텍스트 출처 규칙(설계 07-08 — 저장 위치별로 출처 고정):
  · L2 는 관찰 문장 원문을 코드로만 조립(원문 + 인용 타임스탬프 + 확정 장면).
    여기서 LLM 을 부르면 "L2=관찰 원문" 보장이 깨진다 — 문장 다듬기 금지.
  · L1 은 LLM 이 문장을 만드는 유일한 곳 — 주장 분해 사실 검수(파이프라인 자막
    검수 부품 이식)를 통과한 청크만 색인. 실패 trait 은 증거 제한 재작성 1회,
    그래도 실패하면 드롭(안전한 저하 — 요약이 비어도 L2·카드가 남는다).
  · 저작(author) 텍스트는 DB 에 넣지 않는다 — 같은 내용의 원천 관찰이 이미 여기
    있고, 자막 검수 통과는 "기록이 받친다"까지만 보증.
  · re-ID 게이트 — meta.foster_uncertain 이 남아 있으면 인제스트 거부(틀린 re-ID
    는 다른 개의 행동을 이 개 프로필에 조용히 기록한다).

L1 재집계는 잡 디렉토리가 아니라 DB 의 video_analysis 에서 — 영상이 누적될수록
요약이 두꺼워지고, 잡 디렉토리가 정리돼도 재집계 가능.
"""

from __future__ import annotations

import json

from pipeline.m6_edit.author import MODEL, _caption_claims, _claim_supported
from pipeline.workspace import Workspace

from . import TRAITS
from .db import connect, init_db, vec_literal
from .embed import embed_texts

UNCERTAIN_W = 0.5      # 설계: uncertain 구간 0.5 가중
CITE_CONF = 0.7        # 설계: L2 인용 타임스탬프 = conf≥0.7 & uncertain 아님
CONF_FULL_SECS = 60.0  # trait conf = min(1, 관찰초/60) [잠정 — 'trait conf 분포' 검증으로 보정]
SUMM_TEMP = 0.2        # L1 요약은 창작이 아니다 — 저작(0.9)과 달리 낮게


# --------------------------------------------------------------------------- #
# 결정론 — 게이트·롤업·L2 조립 (LLM 없음)
# --------------------------------------------------------------------------- #
def foster_gate(meta: dict) -> None:
    unc = meta.get("foster_uncertain") or []
    if unc:
        raise RuntimeError(
            f"re-ID 게이트: 임보견 미확정 영상 {sorted(unc)} — 카드 확정 후 인제스트")


def rollup(all_segs: "list[list[dict]]") -> dict:
    """M4 세그먼트 confidence 가중 동/정 롤업 — 활동성·독립성의 결정론 원천."""
    dyn = sta = tot = unc = 0.0
    for segs in all_segs:
        for s in segs:
            d = max(float(s["end_t"]) - float(s["start_t"]), 0.0)
            w = float(s.get("conf") or 0.0)
            if s.get("uncertain"):
                w *= UNCERTAIN_W
                unc += d
            tot += d
            if s.get("group") == "dynamic":
                dyn += d * w
            else:
                sta += d * w
    ratio = dyn / (dyn + sta) if (dyn + sta) > 0 else 0.0
    return {"total_secs": tot, "dynamic_ratio": ratio,
            "uncertain_pct": (unc / tot if tot else 0.0),
            "conf": min(1.0, tot / CONF_FULL_SECS)}


def _cite(segs: "list[dict]"):
    """인용 구간 — 고신뢰(conf≥0.7, uncertain 아님) 중 최고 conf. 없으면 None."""
    best = None
    for s in segs:
        if s.get("uncertain") or float(s.get("conf") or 0.0) < CITE_CONF:
            continue
        if best is None or float(s["conf"]) > float(best["conf"]):
            best = s
    return best


def l2_rows(name: str, prof: dict, segs: "list[dict]",
            tags: "set[str]") -> "list[dict]":
    """L2 결정론 조립 — behavior(행동 에피소드)·caption(장면 서술) 원문 그대로.

    행동 청크에만 인용 타임스탬프를 붙인다(구체 일화 인용용). 조립은 문자열
    연결뿐 — 어휘의 원천은 관찰 캡셔너와 확정 장면 태그(데이터이지 상수 아님).
    """
    ctx = " — 장면: " + ", ".join(sorted(tags)) if tags else ""
    rows = []
    cite = _cite(segs)
    beh = (prof.get("behavior") or "").strip()
    if beh:
        src = (f" ({name} 영상 {float(cite['start_t']):.0f}초)" if cite
               else f" ({name} 영상)")
        rows.append({"video_id": name, "text": beh + ctx + src,
                     "t_start": float(cite["start_t"]) if cite else None,
                     "t_end": float(cite["end_t"]) if cite else None,
                     "conf": float(cite["conf"]) if cite else None})
    cap = (prof.get("caption") or "").strip()
    if cap:
        rows.append({"video_id": name, "text": cap + ctx + f" ({name} 영상)",
                     "t_start": None, "t_end": None, "conf": None})
    return rows


def _read_segments(ws: Workspace, name: str) -> "list[dict]":
    p = ws.preds_m4(name)
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("segments") or []


def _record_lines(rows: "list[tuple]") -> "list[str]":
    """영상별 관찰 기록 한 줄 — L1 생성과 주장 검수의 단일 출처(DB 행 기반).

    author._record_lines 와 같은 조립 결이지만 원천이 잡 디렉토리가 아니라
    video_analysis 행(재집계가 잡 정리와 무관해야 하므로). 동작은 정성 어휘 없이
    측정 숫자만 — 정성 표현의 하중은 behavior·caption 원문이 진다.
    """
    from pipeline.m4_action.observe import profile_text
    lines = []
    for video_id, segs, prof in rows:
        prof = prof or {}
        dyn = sum(s["end_t"] - s["start_t"] for s in segs if s.get("group") == "dynamic")
        sta = sum(s["end_t"] - s["start_t"] for s in segs if s.get("group") != "dynamic")
        line = f"[{video_id}] " + profile_text(
            prof, f"동작측정: 움직임 {dyn:.0f}초/정지 {sta:.0f}초")
        if (prof.get("behavior") or "").strip():
            line += " | 행동: " + prof["behavior"].strip()
        if prof.get("scene_tags"):
            line += " | 확정 장면: " + ", ".join(prof["scene_tags"])
        lines.append(line)
    return lines


# --------------------------------------------------------------------------- #
# L1 — LLM 이 문장을 만드는 유일한 곳 (Gemma 요약 → 주장 분해 검수 → 통과분만 색인)
# --------------------------------------------------------------------------- #
def generate_l1(dog_name: str, records: "list[str]", roll: dict) -> dict:
    """trait 별 상담용 요약 — 지시만, 어휘의 원천은 관찰 기록(상수 금지)."""
    import ollama
    stat = (f"측정: 전체 관찰 {roll['total_secs']:.0f}초, 움직임 가중비율 "
            f"{roll['dynamic_ratio']:.0%}, 판정불확실 구간 {roll['uncertain_pct']:.0%}")
    prompt = (
        f"강아지 '{dog_name}' 영상들의 관찰 기록:\n" + "\n".join(records) +
        f"\n{stat}\n\n"
        "위 기록만 근거로, 입양 문의자 상담용 성격 요약을 trait 별로 써라.\n"
        "trait: " + ", ".join(TRAITS) + "\n"
        "규칙: trait 당 1~2문장. 기록에 있는 사실만 — 기록에 없는 행동·사건·장소를"
        " 지어내지 마라. 물건이 보인다는 기록은 그 물건을 쓰는 행동의 근거가"
        " 아니다. 관찰이 부족한 trait 은 지어내지 말고 빈 문자열로 둬라."
        " 단정 대신 관찰된 범위로 서술하라.")
    fmt = {"type": "object",
           "properties": {t: {"type": "string"} for t in TRAITS},
           "required": list(TRAITS)}
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                    options={"temperature": SUMM_TEMP, "num_predict": 512},
                    format=fmt, think=False)
    try:
        data = json.loads(r.message.content)
    except json.JSONDecodeError:
        return {}
    return {t: (data.get(t) or "").strip() for t in TRAITS}


def _rewrite(trait: str, text: str, records: "list[str]",
             bad: "list[str]") -> str:
    """증거 제한 재작성 — 파이프라인 자막 검수의 2라운드 결(재작성도 도망칠 수
    있어 재판정은 호출부가 한다)."""
    import ollama
    prompt = (
        "관찰 기록:\n" + "\n".join(records) + "\n\n"
        f"다음 문장에서 기록이 뒷받침하지 않는 주장({', '.join(bad)})을 빼고, "
        "기록이 뒷받침하는 내용까지만으로 다시 써라. 1~2문장. 남길 근거가 없으면 "
        f"빈 문자열.\n문장: {text}")
    fmt = {"type": "object", "properties": {"text": {"type": "string"}},
           "required": ["text"]}
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.4, "num_predict": 128},
                    format=fmt, think=False)
    try:
        return (json.loads(r.message.content).get("text") or "").strip()
    except json.JSONDecodeError:
        return ""


def _supported(claim: str, records: "list[str]") -> bool:
    """any-fit — L1 은 개 전체 요약이라 어느 영상 기록이든 받치면 근거. 블록
    자막의 all-fit(그 블록 모든 소스 위에 뜸)과 판정 기준이 다르다. 여러 기록을
    한 줄로 합쳐 판정하는 방식은 기각됨(파이프라인 실측 — 판정이 물러짐).
    기록별로 대조하되 첫 지지에서 멈춘다(호출 비용)."""
    return any(_claim_supported(claim, rec) for rec in records)


def verify_l1(texts: dict, records: "list[str]") -> dict:
    """주장 분해 검수 — 통과분만 반환. 실패 trait 은 재작성 1회 후 잔존이면 드롭."""
    passed = {}
    for trait, text in texts.items():
        if not text:
            continue
        bad = [c for c in _caption_claims(text) if not _supported(c, records)]
        if not bad:
            passed[trait] = text
            continue
        print(f"     [RAG 검수] {trait} 근거 미달 {bad} → 증거 제한 재작성")
        text2 = _rewrite(trait, text, records, bad)
        if text2 and all(_supported(c, records) for c in _caption_claims(text2)):
            passed[trait] = text2
        else:
            print(f"     [RAG 검수] ⚠️ {trait} 재작성도 미달 → 드롭(안전한 저하)")
    return passed


# --------------------------------------------------------------------------- #
# 인제스트 본체
# --------------------------------------------------------------------------- #
def ingest_job(job_id: str, dog_id: str, dog_name: "str | None" = None,
               data_root: "str | None" = None, skip_l1: bool = False) -> dict:
    """잡 1개 인제스트: video_analysis upsert → 해당 영상 L2 재조립 → L1 재집계.

    같은 영상(video_id=파일명) 재인제스트는 갱신 — 잡이 달라도 푸티지 정체성 기준.
    """
    ws = Workspace.job(job_id, data_root)
    meta = ws.read_meta()
    foster_gate(meta)
    profiles = meta.get("scene_profile") or {}
    tags = ws.scene_tags()
    names = meta.get("sources") or sorted(profiles)

    conn = connect()
    try:
        init_db(conn)
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO dogs(dog_id, name, status) VALUES (%s,%s,%s) "
                "ON CONFLICT (dog_id) DO NOTHING",
                (dog_id, dog_name or dog_id, "미입력"))

        ingested = []
        for name in names:
            prof = profiles.get(name)
            segs = _read_segments(ws, name)
            if not prof and not segs:
                print(f"     [RAG] {name}: 프로필·M4 태그 없음 — 건너뜀")
                continue
            store = dict(prof or {})
            if tags.get(name):
                store["scene_tags"] = sorted(tags[name])
            with conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO video_analysis(dog_id, video_id, job_id, segments, profile) "
                    "VALUES (%s,%s,%s,%s,%s) "
                    "ON CONFLICT (dog_id, video_id) DO UPDATE SET "
                    "job_id=EXCLUDED.job_id, segments=EXCLUDED.segments, "
                    "profile=EXCLUDED.profile, analyzed_at=now()",
                    (dog_id, name, meta.get("job_id"),
                     json.dumps(segs, ensure_ascii=False),
                     json.dumps(store, ensure_ascii=False)))
            ingested.append(name)

        # L2 재조립 — 이번에 들어온 영상만 (관찰 원문 결정론 조립)
        n_l2 = 0
        rows = []
        for name in ingested:
            prof = profiles.get(name) or {}
            rows += l2_rows(name, prof, _read_segments(ws, name),
                            tags.get(name) or set())
        if rows:
            vecs = embed_texts([r["text"] for r in rows])
            with conn, conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunks WHERE dog_id=%s AND layer='L2' "
                    "AND video_id = ANY(%s)", (dog_id, ingested))
                for r, v in zip(rows, vecs):
                    cur.execute(
                        "INSERT INTO chunks(dog_id, layer, conf, video_id, "
                        "t_start, t_end, text, embedding) "
                        "VALUES (%s,'L2',%s,%s,%s,%s,%s,%s::vector)",
                        (dog_id, r["conf"], r["video_id"], r["t_start"],
                         r["t_end"], r["text"], vec_literal(v)))
            n_l2 = len(rows)

        n_l1 = 0
        if not skip_l1:
            n_l1 = reaggregate_l1(conn, dog_id, dog_name or dog_id)
        return {"videos": len(ingested), "l2": n_l2, "l1": n_l1}
    finally:
        conn.close()


def reaggregate_l1(conn, dog_id: str, dog_name: str) -> int:
    """L1 재집계 — DB 의 video_analysis 전체에서(설계: 새 영상마다 증분 재집계)."""
    with conn.cursor() as cur:
        cur.execute("SELECT video_id, segments, profile FROM video_analysis "
                    "WHERE dog_id=%s ORDER BY video_id", (dog_id,))
        db_rows = cur.fetchall()
    if not db_rows:
        return 0
    roll = rollup([segs for _, segs, _ in db_rows])
    records = _record_lines(db_rows)
    texts = generate_l1(dog_name, records, roll)
    passed = verify_l1(texts, records)
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE dog_id=%s AND layer='L1'", (dog_id,))
        if passed:
            vecs = embed_texts(list(passed.values()))
            for (trait, text), v in zip(passed.items(), vecs):
                cur.execute(
                    "INSERT INTO chunks(dog_id, layer, trait, conf, text, embedding) "
                    "VALUES (%s,'L1',%s,%s,%s,%s::vector)",
                    (dog_id, trait, roll["conf"], text, vec_literal(v)))
    return len(passed)
