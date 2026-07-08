-- 강아지 RAG 스키마 — 설계: Notion '강아지 RAG 저장·검색 — 최종 설계'(07-08 개정)
-- 저장 위치별로 텍스트 출처 고정: dogs/dog_fields=사람 입력, video_analysis=측정
-- 원본, chunks.layer L1=LLM 생성(검수 통과분만), L2=관찰 문장 원문(코드 조립).
CREATE EXTENSION IF NOT EXISTS vector;

-- 데이터 A — 정형 프로필 카드 (사람 입력, 임베딩 없음, 세션 시작 시 통째 주입)
CREATE TABLE IF NOT EXISTS dogs (
    dog_id     TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 카드 필드 — vis 3단(public / public_soft / staff). public_soft 는 사람이 카드
-- 작성 시 순화본(soft_value)을 함께 입력한다(실시간 LLM 순화 없음 — 설계 확정).
CREATE TABLE IF NOT EXISTS dog_fields (
    id         BIGSERIAL PRIMARY KEY,
    dog_id     TEXT NOT NULL REFERENCES dogs(dog_id) ON DELETE CASCADE,
    field      TEXT NOT NULL,
    value      TEXT NOT NULL,
    soft_value TEXT,
    vis        TEXT NOT NULL CHECK (vis IN ('public', 'public_soft', 'staff'))
);
CREATE INDEX IF NOT EXISTS dog_fields_dog ON dog_fields(dog_id);

-- 원시 분석 보관 — 영상 1개 = 행 1개. segments 는 실제 M4 출력 그대로,
-- profile 은 관찰 프로필 원본(caption·behavior·audio·places·luma + 벡터 2종).
-- video_id = 소스 파일명(푸티지 정체성) — 같은 영상 재인제스트는 갱신.
CREATE TABLE IF NOT EXISTS video_analysis (
    dog_id      TEXT NOT NULL REFERENCES dogs(dog_id) ON DELETE CASCADE,
    video_id    TEXT NOT NULL,
    job_id      TEXT,
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    segments    JSONB NOT NULL,
    profile     JSONB,
    PRIMARY KEY (dog_id, video_id)
);

-- 검색 청크 — layer 로 텍스트 출처가 정해진다(L1=LLM 생성, L2=관찰 원문). 별도
-- 출처 컬럼은 두지 않는다(같은 정보를 두 곳에 기록하지 않음) — 대신 L2 에 LLM
-- 수정이 끼어들면 이 전제가 깨지므로 금지.
CREATE TABLE IF NOT EXISTS chunks (
    id        BIGSERIAL PRIMARY KEY,
    dog_id    TEXT NOT NULL REFERENCES dogs(dog_id) ON DELETE CASCADE,
    layer     TEXT NOT NULL CHECK (layer IN ('L1', 'L2')),
    trait     TEXT,
    conf      REAL,
    vis       TEXT NOT NULL DEFAULT 'public' CHECK (vis IN ('public', 'staff')),
    video_id  TEXT,
    t_start   REAL,
    t_end     REAL,
    text      TEXT NOT NULL,
    embedding vector(1024) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_dog ON chunks(dog_id);
