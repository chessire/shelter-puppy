"""검색 — dog_id 고정 + vis 게이팅 + 코사인 top-k. 카드는 검색 없이 통째 로드.

응대봇 계약(설계): 카드(A)는 세션 시작 시 주입, 질문은 임베딩 → chunks 검색.
staff 필드·청크는 응대봇 컨텍스트에서 제외("담당자 상담 연결" 트리거는 봇 몫).
인용 화법은 layer 로 분기 — 카드="보호소·임보자 진술" / L1="관찰 기록 요약" /
L2="영상 속 장면". layer 로 텍스트 출처가 정해지므로 별도 출처 컬럼이 없다.
"""

from __future__ import annotations

from .db import connect, vec_literal
from .embed import embed_texts


def load_card(dog_id: str, staff: bool = False) -> dict:
    """정형 카드 — vis 필터 적용본. public_soft 는 저장된 순화본만 노출(순화본이
    비어 있으면 노출하지 않음 — 실시간 순화 없음, 설계 확정)."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name, status FROM dogs WHERE dog_id=%s", (dog_id,))
            row = cur.fetchone()
            if not row:
                raise KeyError(f"미등록 dog_id: {dog_id}")
            cur.execute("SELECT field, value, soft_value, vis FROM dog_fields "
                        "WHERE dog_id=%s ORDER BY id", (dog_id,))
            fields = []
            for field, value, soft, vis in cur.fetchall():
                if staff:
                    fields.append({"field": field, "value": value, "vis": vis})
                elif vis == "public":
                    fields.append({"field": field, "value": value, "vis": vis})
                elif vis == "public_soft":
                    if soft:
                        fields.append({"field": field, "value": soft, "vis": vis})
                    else:
                        print(f"     [RAG] ⚠️ public_soft '{field}' 순화본 없음 — 미노출")
        return {"dog_id": dog_id, "name": row[0], "status": row[1],
                "fields": fields}
    finally:
        conn.close()


def search(dog_id: str, query: str, k: int = 5,
           include_staff: bool = False) -> "list[dict]":
    """질문 → bge-m3 임베딩 → WHERE dog_id AND vis≠staff → 코사인 top-k."""
    qv = vec_literal(embed_texts([query])[0])
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT layer, trait, conf, video_id, t_start, t_end, text, "
                "1 - (embedding <=> %s::vector) AS score "
                "FROM chunks WHERE dog_id=%s "
                + ("" if include_staff else "AND vis <> 'staff' ") +
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (qv, dog_id, qv, k))
            cols = ("layer", "trait", "conf", "video_id", "t_start", "t_end",
                    "text", "score")
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()
