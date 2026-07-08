"""DB 연결·초기화 — psycopg2 + pgvector(벡터는 리터럴 캐스트, 별도 어댑터 없음)."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2

from . import DB_URL_ENV, DEFAULT_DB_URL

_SCHEMA = Path(__file__).parent / "schema.sql"


def connect():
    return psycopg2.connect(os.environ.get(DB_URL_ENV, DEFAULT_DB_URL))


def init_db(conn=None) -> None:
    """schema.sql 적용 — 전부 IF NOT EXISTS 라 재실행 안전."""
    own = conn is None
    conn = conn or connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(_SCHEMA.read_text())
    finally:
        if own:
            conn.close()


def vec_literal(v) -> str:
    """float 리스트 → pgvector 리터럴('[0.1,0.2,...]'). SQL 에서 ::vector 캐스트."""
    return "[" + ",".join(f"{x:.6g}" for x in v) + "]"
