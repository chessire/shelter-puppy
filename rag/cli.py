"""RAG CLI — 스키마 초기화·카드 입력·잡 인제스트·검색 확인.

  python -m rag.cli init
  python -m rag.cli card <dog_id> --file card.json
  python -m rag.cli ingest <job_id> --dog <dog_id> [--dog-name 이름]
                    [--data-root ./jobs] [--skip-l1]
  python -m rag.cli search <dog_id> "질문" [-k 5] [--card]

card.json 형식(사람 입력 — 데이터 A):
  {"name": "토리", "status": "입양가능",
   "fields": [{"field": "견종", "value": "진도믹스", "vis": "public"},
              {"field": "경미 질환", "value": "슬개골 1기",
               "soft_value": "가벼운 관절 관리가 필요해요", "vis": "public_soft"}]}
"""

from __future__ import annotations

import argparse
import json


def _cmd_init(_args) -> None:
    from .db import init_db
    init_db()
    print("[RAG] 스키마 적용 완료")


def _cmd_card(args) -> None:
    from .db import connect, init_db
    data = json.loads(open(args.file, encoding="utf-8").read())
    conn = connect()
    try:
        init_db(conn)
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO dogs(dog_id, name, status) VALUES (%s,%s,%s) "
                "ON CONFLICT (dog_id) DO UPDATE SET name=EXCLUDED.name, "
                "status=EXCLUDED.status",
                (args.dog_id, data["name"], data["status"]))
            cur.execute("DELETE FROM dog_fields WHERE dog_id=%s", (args.dog_id,))
            for f in data.get("fields") or []:
                cur.execute(
                    "INSERT INTO dog_fields(dog_id, field, value, soft_value, vis) "
                    "VALUES (%s,%s,%s,%s,%s)",
                    (args.dog_id, f["field"], f["value"],
                     f.get("soft_value"), f["vis"]))
        print(f"[RAG] 카드 저장: {args.dog_id} 필드 {len(data.get('fields') or [])}개")
    finally:
        conn.close()


def _cmd_ingest(args) -> None:
    from .ingest import ingest_job
    n = ingest_job(args.job_id, args.dog, dog_name=args.dog_name,
                   data_root=args.data_root, skip_l1=args.skip_l1)
    print(f"[RAG] 인제스트 완료: 영상 {n['videos']}개, L2 {n['l2']}청크, "
          f"L1 {n['l1']}청크")


def _cmd_search(args) -> None:
    from .retrieve import load_card, search
    if args.card:
        card = load_card(args.dog_id)
        print(f"— 카드: {card['name']} ({card['status']})")
        for f in card["fields"]:
            print(f"  · {f['field']}: {f['value']}")
    for r in search(args.dog_id, args.query, k=args.k):
        loc = f" @{r['video_id']}" if r["video_id"] else ""
        conf = f" conf={r['conf']:.2f}" if r["conf"] is not None else ""
        trait = f"[{r['trait']}] " if r["trait"] else ""
        print(f"  {r['score']:.3f} {r['layer']}{loc}{conf} {trait}{r['text']}")


def main() -> None:
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")

    c = sub.add_parser("card")
    c.add_argument("dog_id")
    c.add_argument("--file", required=True)

    i = sub.add_parser("ingest")
    i.add_argument("job_id")
    i.add_argument("--dog", required=True)
    i.add_argument("--dog-name")
    i.add_argument("--data-root")
    i.add_argument("--skip-l1", action="store_true")

    s = sub.add_parser("search")
    s.add_argument("dog_id")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--card", action="store_true")

    args = p.parse_args()
    {"init": _cmd_init, "card": _cmd_card,
     "ingest": _cmd_ingest, "search": _cmd_search}[args.cmd](args)


if __name__ == "__main__":
    main()
